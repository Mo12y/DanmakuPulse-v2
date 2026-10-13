#!/usr/bin/env python3
"""
m4_evolution.py — M4 热榜纵向演化：日弹幕时序 + 演化四维 + K-means 分型 + 生命周期（08 §7）

输入：dwd_danmaku_clean（~spam）+ dim_video（pubdate/last_batch/first_batch）+ dws_hawkes_params
输出：
  ① {wh}/dws_video_daily —— 每 (bvid, batch_date, send_date) 一行：cnt / wcnt（日弹幕时序）
  ② {wh}/dws_evolution_feats —— 每 (bvid, batch_date) 一行（门控 D≥min_days）：
     演化四维 9 特征 + cluster_id/cluster_type（k=3：爆发-衰减/多波震荡/迟发峰值）+ 轮廓
  ③ {wh}/dws_lifecycle —— 每 bvid 一行（最新快照）：观测天数/峰值/半衰期（Hawkes 或序列描述）
  ④ 报告 data-out/m4_evolution_report.json（DoD：轮廓≥0.4；三型占比合计 100% 且每型≥5%）

口径（08 §7；「Entropy 规格」四维）：
  - 序列 = 视频发布后 [0, 21d] 窗口内按日聚合（与 L3 窗口一致）；门控 D≥--min-days（v0=5，
    ⚠️ 08 未指定，实测中位数仅 3 天——参数化留痕）
  - 趋势（升段比/转折数/极差比）｜峰值（显著峰数〔复用 l2_common.platform_peaks〕/峰距/突出度）
    ｜统计（偏度/峰度）｜极值（Max_Index）；标准化后 MLlib K-means++（k=3, seed=42）
  - 三型命名 = 按质心规则程序化判定（迟发=质心 max_idx 最大；多波=余下质心显著峰数最大；
    余=爆发-衰减）——规则固定、可复现，命名是描述性归纳（沿用旧项目「形态图谱」诚实边界）
  - 生命周期半衰期优先级：Hawkes/幂律（dws_hawkes_params.decay_halflife_d）→ 否则序列描述
    （峰值后首日 cnt ≤ 峰值/2 的天数）
"""
import argparse
import json
import os

import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

ENGINE_VERSION = 'b2-v1'
FEATS = ['up_frac', 'n_turns', 'range_ratio', 'peak_cnt_sig', 'peak_gap_avg',
         'peak_prominence', 'skew', 'kurt', 'max_idx_norm']
TYPES = ['爆发-衰减', '多波震荡', '迟发峰值']


def series_features(c):
    """演化四维（9 特征）。c: 日计数序列（np 数组，升序）。"""
    from scipy.stats import kurtosis, skew
    import l2_common
    c = np.asarray(c, dtype=float)
    D = c.size
    diffs = np.diff(c)
    up_frac = float(np.mean(diffs > 0)) if diffs.size else 0.0
    nz = diffs[diffs != 0]
    n_turns = int(np.sum(np.diff(np.sign(nz)) != 0)) if nz.size >= 2 else 0
    med = float(np.median(c))
    mad = float(np.median(np.abs(c - med))) * 1.4826
    rng = float(c.max() - c.min())
    mean = float(c.mean())
    pk = l2_common.platform_peaks(list(c))
    sig = [i for i in pk if c[i] > med + mad]
    peak_cnt_sig = len(sig)
    if len(sig) >= 2:
        peak_gap_avg = float(np.mean(np.diff(sig)))
    else:
        peak_gap_avg = float(D - 1)          # 无显著峰或单峰 → 记窗口全宽（文档化）
    peak_prominence = float(c.max() / (med + 1.0))
    return {
        'up_frac': up_frac, 'n_turns': n_turns, 'range_ratio': float(rng / (mean + 1e-9)),
        'peak_cnt_sig': peak_cnt_sig, 'peak_gap_avg': peak_gap_avg,
        'peak_prominence': peak_prominence,
        'skew': float(skew(c)) if D >= 3 else 0.0,
        'kurt': float(kurtosis(c)) if D >= 3 else 0.0,
        'max_idx_norm': float(np.argmax(c) / max(D - 1, 1)),
    }


def name_types(centers):
    """质心 → 三型命名（规则固定）：迟发= max_idx_norm 最大；多波=余下 peak_cnt_sig 最大；余=爆发-衰减。"""
    idx = list(range(len(centers)))
    late = max(idx, key=lambda i: centers[i][FEATS.index('max_idx_norm')])
    rest = [i for i in idx if i != late]
    multi = max(rest, key=lambda i: centers[i][FEATS.index('peak_cnt_sig')])
    burst = [i for i in rest if i != multi][0]
    return {late: TYPES[2], multi: TYPES[1], burst: TYPES[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--min-days', type=int, default=5)
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('m4_evolution')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean').filter(~F.col('is_spam'))
           .select('bvid', 'batch_date', 'send_date', 'send_time', 'attr_weight', 'pubdate'))
    dim = (spark.read.parquet(f'{wh}/dim_video')
           .select('bvid', F.col('pubdate').cast('long').alias('pub_p'), 'last_batch', 'first_batch'))

    # ── ① dws_video_daily（全量日时序；窗口裁剪由下游/演化层做）──
    daily = (dwd.groupBy('bvid', 'batch_date', 'send_date')
             .agg(F.count('*').alias('cnt'), F.sum('attr_weight').alias('wcnt'))
             .withColumnRenamed('send_date', 'day')
             .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    daily.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_video_daily')
    n_daily = daily.count()

    # ── ② 演化特征（窗口 [pubdate, +21d]，门控 D≥min_days）──
    d = dwd.join(dim, 'bvid').filter(F.col('pub_p').isNotNull())
    dd = (d.withColumn('day_idx', F.floor((F.col('send_time') - F.col('pub_p')) / 86400.0).cast('int'))
          .filter((F.col('day_idx') >= 0) & (F.col('day_idx') < 21)))
    g = (dd.groupBy('bvid', 'batch_date')
         .agg(F.collect_list('day_idx').alias('days')))
    n_all = g.count()

    def run_one(r):
        c = np.bincount(np.asarray(r['days'], dtype=int), minlength=21)[:21]
        nz = np.nonzero(c)[0]
        D = int(nz.max()) + 1 if nz.size else 0
        if D < args.min_days:
            return None
        return (r['bvid'], r['batch_date'], series_features(c[:D]), D)

    rows_all = g.rdd.map(run_one).filter(lambda x: x is not None).collect()
    n_gate = len(rows_all)
    flat = [(bv, bd, D) + tuple(fx[f] for f in FEATS) for bv, bd, fx, D in rows_all]
    feats_df = spark.createDataFrame(flat, ['bvid', 'batch_date', 'n_days'] + FEATS).cache()
    feats_df.count()

    # 标准化 + K-means++（k=3）
    asm = VectorAssembler(inputCols=FEATS, outputCol='features')
    sc = StandardScaler(inputCol='features', outputCol='features_scaled', withMean=True, withStd=True)
    ass = asm.transform(feats_df)
    scaled = sc.fit(ass).transform(ass).cache()
    scaled.count()
    km = KMeans(featuresCol='features_scaled', predictionCol='cluster_id', k=3, seed=42)
    model = km.fit(scaled)
    pred = model.transform(scaled).cache()
    pred.count()
    sil = float(ClusteringEvaluator(featuresCol='features_scaled', predictionCol='cluster_id')
                .evaluate(pred))
    centers = np.array(model.clusterCenters())
    tmap = name_types(centers)

    typed = (pred.withColumn('cluster_type',
                             F.create_map(*[x for i in range(len(tmap))
                                            for x in (F.lit(i), F.lit(tmap[i]))])[
                                 F.col('cluster_id')])
             .withColumn('k', F.lit(3)).withColumn('silhouette', F.lit(sil))
             .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    typed.select('bvid', 'batch_date', *FEATS, 'n_days', 'cluster_id', 'cluster_type',
                 'k', 'silhouette', 'engine_version') \
        .write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_evolution_feats')

    sizes = {int(r['cluster_id']): r['count'] for r in
             typed.groupBy('cluster_id').count().collect()}
    type_sizes = {tmap[k]: v for k, v in sizes.items()}
    type_frac = {t: round(v / n_gate, 4) for t, v in type_sizes.items()}

    # ── ③ dws_lifecycle（每 bvid：最新快照；半衰期 = Hawkes/幂律 → 序列描述）──
    lw = (dd.filter(F.col('batch_date') == F.col('last_batch'))
          .groupBy('bvid').agg(F.collect_list('day_idx').alias('days')))
    hawkes = None
    try:
        hawkes = (spark.read.parquet(f'{wh}/dws_hawkes_params')
                  .select('bvid', F.col('decay_halflife_d').alias('hk_halflife'),
                          F.col('method').alias('hk_method')))
    except Exception:
        pass

    def life_one(r):
        c = np.bincount(np.asarray(r['days'], dtype=int), minlength=21)[:21]
        nz = np.nonzero(c)[0]
        if not nz.size:
            return None
        D = int(nz.max()) + 1
        peak_day = int(np.argmax(c))
        peak_cnt = int(c.max())
        half = None
        for d2 in range(peak_day + 1, D):
            if c[d2] <= peak_cnt / 2.0:
                half = int(d2 - peak_day)
                break
        return (r['bvid'], D, int(c[:D].sum()), peak_day, peak_cnt, half)

    lrows = lw.rdd.map(life_one).filter(lambda x: x is not None).collect()
    ldf = spark.createDataFrame(lrows, ['bvid', 'n_days_obs', 'total_cnt', 'peak_day',
                                        'peak_cnt', 'half_decay_days'])
    if hawkes is not None:
        ldf = ldf.join(hawkes, 'bvid', 'left')
    else:
        ldf = (ldf.withColumn('hk_halflife', F.lit(None).cast('double'))
               .withColumn('hk_method', F.lit(None).cast('string')))
    ldf = (ldf.withColumn('halflife_d', F.coalesce(F.col('hk_halflife'),
                                                   F.col('half_decay_days').cast('double')))
           .withColumn('halflife_src', F.when(F.col('hk_halflife').isNotNull(),
                                              F.concat(F.lit('hawkes/'), F.col('hk_method')))
                       .when(F.col('half_decay_days').isNotNull(), F.lit('series'))
                       .otherwise(F.lit('none')))
           .withColumn('halflife_rank', F.when(F.col('halflife_d').isNotNull(),
                        F.rank().over(Window.orderBy(F.col('halflife_d').desc()))))
           .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    ldf.write.mode('overwrite').parquet(f'{wh}/dws_lifecycle')

    report = {
        'job': 'm4_evolution', 'engine_version': ENGINE_VERSION,
        'min_days_gate': args.min_days,
        'n_daily_rows': n_daily,
        'n_video_batches_all': n_all, 'n_gate': n_gate,
        'k': 3, 'silhouette': round(sil, 6),
        'type_sizes': type_sizes, 'type_frac': type_frac,
        'dod_silhouette_ge_040': sil >= 0.40,
        'dod_each_type_ge_5pct': all(v >= 0.05 for v in type_frac.values()),
        'type_rule': '迟发=max_idx 最大；多波=余下显著峰数最大；余=爆发-衰减（质心规则，可复现）',
        'note': '门控 D≥min_days 为 v0 选择（08 未指定；实测日序列中位数 3 天）',
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm4_evolution_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m4_evolution：daily {n_daily} 行；演化 {n_gate}/{n_all}（门控≥{args.min_days}d）；'
          f'silhouette={sil:.4f}；三型 {type_frac}')
    spark.stop()


if __name__ == '__main__':
    main()
