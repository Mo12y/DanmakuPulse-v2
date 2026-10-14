#!/usr/bin/env python3
"""
m4_evolution.py — M4 演化 v2（跨批指标轨迹分型 + 生命周期）（2026-10-14 收口批次 ④）

⛔ v1（勿引用其数字）：基于「快照内日弹幕序列」的演化分型。2026-10-14 实锤：B 站弹幕接口对
饱和视频返回固定容量「最近 N 条」滑动窗（danmaku_got 量子化在 600 的倍数；同一视频跨批
reported 增长而 got 不变、「峰值日」漂移）⇒ 该序列是窗的落点、形状不可复现，全量换变量。

v2 口径：
  - 轨迹 = dws_metric_track（daily）按采集时刻排序、剔除批次倒挂观测（见 m4_track_common）
  - 特征全部来自可观测指标（g_play/g_like/g_reported/rank_delta/rank_last/eng_shift/sat_last；
    pool_saturation = 1 − coverage 本身是信息）；**禁用 danmaku_got 做水平/形状/半衰期**
  - 半衰期：hawkes（既有，带已知口径注记）→ 序列口径·仅「可行批」（coverage ≥ 0.95，池≈全量）；
    metric 增长半衰期（v0 实验列，观测 ≥4 后可用）

输出：
  ① {wh}/dws_video_daily —— 每 (bvid, batch_date, day) 一行：日弹幕时序（**快照口径**，
     饱和视频为该批窗口落点——仅描述用，不得用于跨批水平/形状）
  ② {wh}/dws_evolution_feats —— 每 bvid 一行：轨迹特征 + K-means(k=3) 三型
     （质心规则命名：快增型/稳增型/缓增型，按 g_play 质心降序）
  ③ {wh}/dws_lifecycle —— 每 bvid 一行：观测跨度/最新水平/增长率 + 半衰期 + 饱和标记
  ④ 报告 data-out/m4_evolution_report.json（DoD：轮廓 + 每型占比；v1 数字作废）
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
import pyspark.sql.types as T

import m4_track_common as mtc

ENGINE_VERSION = 'm4-v2'


def parse_epoch(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts)).timestamp()
    except Exception:
        try:
            return datetime.strptime(str(ts)[:19], '%Y-%m-%d %H:%M:%S').timestamp()
        except Exception:
            return None


def epoch_to_iso(t):
    return None if t is None else datetime.fromtimestamp(t).isoformat(timespec='seconds')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--min-days', type=int, default=5, help='序列口径日观测门（v0=5，仅可行批用）')
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

    # ── ① dws_video_daily（快照口径日时序；保留为描述性事实表）──
    dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean').filter(~F.col('is_spam'))
           .select('bvid', 'batch_date', 'send_date', 'send_time', 'attr_weight', 'pubdate'))
    daily = (dwd.groupBy('bvid', 'batch_date', 'send_date')
             .agg(F.count('*').alias('cnt'), F.sum('attr_weight').alias('wcnt'))
             .withColumnRenamed('send_date', 'day')
             .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    daily.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_video_daily')
    n_daily = daily.count()

    # ── ② 轨迹（driver 侧组装；dws_metric_track daily）──
    track = (spark.read.parquet(f'{wh}/dws_metric_track')
             .filter(F.col('cadence') == 'daily')
             .select('bvid', 'batch_date', 'video_fetched_at', 'play', 'like',
                     'danmaku_reported', 'rank_proxy', 'pool_saturation', 'danmaku_got'))
    rows = track.collect()
    by_video = {}
    for r in rows:
        by_video.setdefault(r['bvid'], []).append({
            'batch': int(r['batch_date']), 't': parse_epoch(r['video_fetched_at']),
            'play': r['play'], 'like': r['like'], 'reported': r['danmaku_reported'],
            'rank': r['rank_proxy'], 'sat': r['pool_saturation'], 'got': r['danmaku_got']})

    n_dropped = n_assumed = 0
    feats_rows, life_raw, last_pairs = [], [], []
    for bvid, obs in by_video.items():
        kept, dropped, assumed = mtc.build_trajectory(obs)
        n_dropped += dropped
        n_assumed += int(assumed)
        if not kept:
            continue
        first, last = kept[0], kept[-1]
        fx = mtc.trajectory_features(kept)
        cov_last = (1.0 - last['sat']) if last['sat'] is not None else None
        adm = bool(cov_last is not None and cov_last >= 0.95)
        mt_tau, mt_n = mtc.metric_halflife(kept, 'play')
        life_raw.append({
            'bvid': bvid, 'n_obs': len(kept),
            'first_batch': first['batch'], 'last_batch': last['batch'],
            'first_fetched_at': epoch_to_iso(first['t']),
            'last_fetched_at': epoch_to_iso(last['t']),
            'span_days': (fx['span_days'] if fx else None),
            'play_last': last['play'], 'like_last': last['like'],
            'danmaku_reported_last': last['reported'], 'rank_last': last['rank'],
            'danmaku_got_last': last['got'], 'coverage_last': cov_last,
            'pool_saturation_last': last['sat'],
            'series_admissible': adm,
            'g_play_daily': (fx['g_play'] if fx else None),
            'g_reported_daily': (fx['g_reported'] if fx else None),
            'metric_halflife_d': mt_tau, 'metric_halflife_n_rates': mt_n,
        })
        if fx is not None:
            feats_rows.append({'bvid': bvid, 'last_batch': last['batch'], 'fx': fx})
        last_pairs.append((bvid, str(last['batch']), adm))

    # ── ③ 序列口径半衰期（仅可行批；峰后 cnt ≤ 峰值/2 的天数）──
    series_map = {}
    if last_pairs:
        dim = (spark.read.parquet(f'{wh}/dim_video')
               .select('bvid', F.col('pubdate').cast('long').alias('pub_p')))
        lp = spark.createDataFrame(last_pairs, ['bvid', 'last_batch', 'adm'])
        dd = (dwd.join(dim, 'bvid').filter(F.col('pub_p').isNotNull())
              .withColumn('day_idx',
                          F.floor((F.col('send_time') - F.col('pub_p')) / 86400.0).cast('int'))
              .filter((F.col('day_idx') >= 0) & (F.col('day_idx') < 21)))
        gg = (dd.join(F.broadcast(lp),
                      (dd['bvid'] == lp['bvid']) & (dd['batch_date'] == lp['last_batch']))
              .groupBy(lp['bvid'], 'adm')
              .agg(F.collect_list('day_idx').alias('days')))
        for r in gg.collect():
            if not r['adm']:
                continue
            c = np.bincount(np.asarray(r['days'], dtype=int), minlength=21)[:21]
            nz = np.nonzero(c)[0]
            if not nz.size:
                continue
            D = int(nz.max()) + 1
            peak_day = int(np.argmax(c))
            peak_cnt = int(c.max())
            half = None
            for d2 in range(peak_day + 1, D):
                if c[d2] <= peak_cnt / 2.0:
                    half = int(d2 - peak_day)
                    break
            series_map[r['bvid']] = {
                'n_days_obs': D, 'total_cnt': int(c[:D].sum()),
                'peak_day': peak_day, 'peak_cnt': peak_cnt, 'half_decay_days': half}

    # ── ④ K-means 三型（仅「全特征齐备」轨迹；partial 行保留、cluster 置空）──
    # 说明：n=2 批时，缺少任一侧采集时刻的视频 g_* 不可算（assumed 序）——不填零参与聚类，
    # 否则「无信息」会被聚成一型（2026-10-14 首跑实测教训：fillna(0) 产生假「缓增型」）。
    full_rows, partial_rows = [], []
    for fr in feats_rows:
        vals = [fr['fx'].get(f) for f in mtc.FEATS_V2]
        row = ([fr['bvid'], int(fr['last_batch'])] +
               [None if v is None else float(v) for v in vals] +
               [int(fr['fx']['n_obs']), float(fr['fx']['span_days'] or 0.0)])
        (full_rows if all(v is not None for v in vals) else partial_rows).append(tuple(row))

    n_feats_all, n_full, n_partial = len(feats_rows), len(full_rows), len(partial_rows)
    result = {'clustered': False}
    if n_full >= 10:
        schema = T.StructType(
            [T.StructField('bvid', T.StringType()), T.StructField('last_batch', T.IntegerType())]
            + [T.StructField(f, T.DoubleType()) for f in mtc.FEATS_V2]
            + [T.StructField('n_obs', T.IntegerType()),
               T.StructField('span_days', T.DoubleType())])
        df = spark.createDataFrame(full_rows, schema)
        asm = VectorAssembler(inputCols=mtc.FEATS_V2, outputCol='features')
        sc = StandardScaler(inputCol='features', outputCol='features_scaled',
                            withMean=True, withStd=True)
        ass = asm.transform(df)
        scaled = sc.fit(ass).transform(ass).cache()
        scaled.count()
        km = KMeans(featuresCol='features_scaled', predictionCol='cluster_id', k=3, seed=42)
        model = km.fit(scaled)
        pred = model.transform(scaled).cache()
        pred.count()
        sil = float(ClusteringEvaluator(featuresCol='features_scaled',
                                        predictionCol='cluster_id').evaluate(pred))
        tmap = mtc.name_types_v2(model.clusterCenters())
        typed = (pred
                 .withColumn('cluster_type',
                             F.create_map(*[x for i in range(len(tmap))
                                            for x in (F.lit(i), F.lit(tmap[i]))])
                             [F.col('cluster_id')])
                 .withColumn('k', F.lit(3)).withColumn('silhouette', F.lit(sil))
                 .withColumn('engine_version', F.lit(ENGINE_VERSION)))
        cols_out = ['bvid', 'last_batch', 'n_obs', 'span_days', *mtc.FEATS_V2,
                    'cluster_id', 'cluster_type', 'k', 'silhouette', 'engine_version']
        typed_all = typed.select(*cols_out)
        if n_partial:
            pdf = (spark.createDataFrame(partial_rows, schema)
                   .withColumn('cluster_id', F.lit(None).cast('int'))
                   .withColumn('cluster_type', F.lit(None).cast('string'))
                   .withColumn('k', F.lit(3)).withColumn('silhouette', F.lit(sil))
                   .withColumn('engine_version', F.lit(ENGINE_VERSION)))
            typed_all = typed_all.unionByName(pdf.select(*cols_out))
        typed_all.write.mode('overwrite').parquet(f'{wh}/dws_evolution_feats')

        from statistics import median
        by_c = {}
        for r in pred.select('cluster_id', *mtc.FEATS_V2).collect():
            by_c.setdefault(int(r['cluster_id']), []).append(r)
        type_sizes = {tmap[c]: len(v) for c, v in by_c.items()}
        type_frac = {t: round(v / n_full, 4) for t, v in type_sizes.items()}
        type_medians = {tmap[c]: {f: round(float(median([rr[f] for rr in v])), 6)
                                  for f in mtc.FEATS_V2} for c, v in by_c.items()}
        result = {'clustered': True, 'silhouette': round(sil, 6),
                  'type_sizes': type_sizes, 'type_frac': type_frac,
                  'type_medians': type_medians}
    else:
        print(f'[warn] 轨迹样本 {n_feats} < 10，跳过聚类（轨迹表照常输出）')

    # ── ⑤ dws_lifecycle ──
    life_schema = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('n_obs', T.IntegerType()),
        T.StructField('first_batch', T.IntegerType()), T.StructField('last_batch', T.IntegerType()),
        T.StructField('first_fetched_at', T.StringType()),
        T.StructField('last_fetched_at', T.StringType()),
        T.StructField('span_days', T.DoubleType()),
        T.StructField('play_last', T.LongType()), T.StructField('like_last', T.LongType()),
        T.StructField('danmaku_reported_last', T.IntegerType()),
        T.StructField('rank_last', T.IntegerType()),
        T.StructField('danmaku_got_last', T.IntegerType()),
        T.StructField('coverage_last', T.DoubleType()),
        T.StructField('pool_saturation_last', T.DoubleType()),
        T.StructField('series_admissible', T.BooleanType()),
        T.StructField('g_play_daily', T.DoubleType()),
        T.StructField('g_reported_daily', T.DoubleType()),
        T.StructField('metric_halflife_d', T.DoubleType()),
        T.StructField('metric_halflife_n_rates', T.IntegerType()),
        T.StructField('n_days_obs', T.IntegerType()), T.StructField('total_cnt', T.LongType()),
        T.StructField('peak_day', T.IntegerType()), T.StructField('peak_cnt', T.LongType()),
        T.StructField('half_decay_days', T.IntegerType()),
    ])
    life_rows = []
    for lr in life_raw:
        s = series_map.get(lr['bvid'], {})
        life_rows.append((
            lr['bvid'], lr['n_obs'], lr['first_batch'], lr['last_batch'],
            lr['first_fetched_at'], lr['last_fetched_at'], lr['span_days'],
            lr['play_last'], lr['like_last'], lr['danmaku_reported_last'], lr['rank_last'],
            lr['danmaku_got_last'], lr['coverage_last'], lr['pool_saturation_last'],
            lr['series_admissible'], lr['g_play_daily'], lr['g_reported_daily'],
            lr['metric_halflife_d'], lr['metric_halflife_n_rates'],
            s.get('n_days_obs'), s.get('total_cnt'), s.get('peak_day'),
            s.get('peak_cnt'), s.get('half_decay_days')))
    ldf = spark.createDataFrame(life_rows, life_schema)

    hawkes = None
    try:
        hawkes = (spark.read.parquet(f'{wh}/dws_hawkes_params')
                  .select('bvid', F.col('decay_halflife_d').alias('hk_halflife'),
                          F.col('method').alias('hk_method')))
    except Exception:
        pass
    if hawkes is not None:
        ldf = ldf.join(hawkes, 'bvid', 'left')
    else:
        ldf = (ldf.withColumn('hk_halflife', F.lit(None).cast('double'))
               .withColumn('hk_method', F.lit(None).cast('string')))
    ldf = (ldf.withColumn('halflife_d', F.coalesce(F.col('hk_halflife'),
                                                   F.col('half_decay_days').cast('double')))
           .withColumn('halflife_src',
                       F.when(F.col('hk_halflife').isNotNull(),
                              F.concat(F.lit('hawkes/'), F.col('hk_method')))
                       .when(F.col('half_decay_days').isNotNull(), F.lit('series'))
                       .otherwise(F.lit('none')))
           .withColumn('halflife_rank',
                       F.when(F.col('halflife_d').isNotNull(),
                              F.rank().over(Window.orderBy(F.col('halflife_d').desc()))))
           .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    ldf.write.mode('overwrite').parquet(f'{wh}/dws_lifecycle')
    n_life = ldf.count()
    n_adm = ldf.filter(F.col('series_admissible')).count()
    n_series = ldf.filter(F.col('half_decay_days').isNotNull()).count()
    n_mhl = ldf.filter(F.col('metric_halflife_d').isNotNull()).count()

    report = {
        'job': 'm4_evolution', 'engine_version': ENGINE_VERSION,
        'spec': '跨批指标轨迹（play/like/reported/rank）+ pool_saturation；danmaku_got 仅进 coverage',
        'min_days_gate_note': f'序列口径半衰期仅可行批（coverage≥0.95）；--min-days 仅存参（{args.min_days}）',
        'n_daily_rows': n_daily,
        'n_track_daily_rows': len(rows), 'n_videos': len(by_video),
        'n_dropped_out_of_order': n_dropped, 'n_assumed_order_videos': n_assumed,
        'n_trajectory_feats': n_feats_all, 'n_feats_full': n_full,
        'n_feats_partial': n_partial,
        'clustering_note': '聚类仅用全特征齐备样本（n_feats_full）；partial 行 cluster_id=NULL',
        'k': 3,
        'silhouette': result.get('silhouette'),
        'type_sizes': result.get('type_sizes'), 'type_frac': result.get('type_frac'),
        'type_medians': result.get('type_medians'),
        'dod_silhouette_ge_040': (result.get('silhouette') or -1) >= 0.40,
        'dod_each_type_ge_5pct': bool(result.get('type_frac')) and
                                 all(v >= 0.05 for v in result.get('type_frac', {}).values()),
        'type_rule': '快增/稳增/缓增型 = 按 g_play 质心降序（规则固定、可复现）',
        'n_lifecycle': n_life, 'n_series_admissible': n_adm,
        'n_series_halflife': n_series, 'n_metric_halflife_v0': n_mhl,
        'v1_deprecated': ('v1（快照内日弹幕序列分型：124 行 / silhouette 0.4128）因 danmaku_got '
                          '滑动窗机制作废——不得引用；本表为 v2 全量重算'),
        'note': ('n=2~3 批时为「增速+名次+饱和度」画像（速率分层）；批次数增长后按同框架复跑。'
                 'metric 半衰期为 v0 实验列（≥4 观测后启用），暂不进主口径 halflife_d'),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm4_evolution_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m4_evolution v2：轨迹 {n_feats_all} 个视频（全特征 {n_full}）；'
          f'silhouette={result.get("silhouette")}；三型 {result.get("type_frac")}；'
          f'倒挂剔除 {n_dropped}')
    spark.stop()


if __name__ == '__main__':
    main()
