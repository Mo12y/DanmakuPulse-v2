#!/usr/bin/env python3
"""
ads_build.py — B2.3 生产链 ADS 层（08 §10.1）：dws/ods → PostgreSQL ads 表（JDBC overwrite 全量重建）

覆盖 3 页数据流（运营总览 / 视频节奏详情 / 分区画像）+ M4（演化分型）：
  ads_video_rhythm_curve     (bvid,batch_date,bucket 0-99: cnt,wcnt,cal_cnt,sent_avg,event_id)
  ads_video_curve_5s         (bvid,batch_date,bucket5s: cnt,wcnt,cal_cnt) —— 详情页懒加载（08 §10.3）
  ads_video_highlights       (event 级：start/end/score/sync/summary/top_danmaku/sent_mix)
  ads_video_fingerprint      (16 特征 + 16 z 列 + cluster_id/cluster_type + tname)
  ads_partition_fingerprint  (tname：中位曲线+包络 + 特征 med/mad + n + low_support)
  ads_dual_axis              (bvid：offset_bin(20) × lag_bin(5) 计数；最新快照)
  ads_hotlist_latest / _history（上榜追踪）
  ads_up_profile             (UP 档案)
  ads_evolution_feats        (演化四维 + 三型，页面⑤备用)
  ads_batch_meta / ads_video_metrics / ads_video_curve（B0 min 链产物**不动**——对账锚点页依赖）

口径备注：
  - cal_cnt（1pct）：由 5s 的 cal_cnt 按「桶中点落位」聚合（cal 仅在 5s 网格定义——m2_l2_lag）；
    与精确 cnt 存在 ≤1 个 5s 桶的粒度差（展示用「原始/校正」切换，不参与论文口径）
  - event_id：事件 [start_sec,end_sec] 覆盖的桶 = 桶中点时间落入事件区间；重叠取小 event_id
  - summary：v0 = 事件窗内权重前 3 弹幕拼接（截 40 字）——TextRank 摘要属 P1 增项（列已在，注释标明）
  - top_danmaku：窗内 attr_weight×长度 前 5 条（原始文本）
"""
import argparse
import json
import os

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

ENGINE_VERSION = 'b2-v1'
FEAT_COLS = ['dm_per_min', 'wdm_per_min', 'rhythm_entropy', 'burst_cv', 'w_burst_cv',
             'early_lag_ratio', 'color_ratio', 'top_mode_ratio',
             'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio', 'avg_text_len',
             'opening_5pct', 'ending_5pct', 'mid_ratio']
LAG_ORDER = ['<1h', '1-6h', '6-24h', '1-3d', '>3d']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--pg-host', default='postgres')
    ap.add_argument('--pg-db', default='ads')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true', help='本地模式（跳过 PG 写入，只做构建演练）')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('ads_build')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    url = f'jdbc:postgresql://{args.pg_host}:5432/{args.pg_db}'
    props = {'user': os.environ.get('PG_USER', 'danmaku'),
             'password': os.environ.get('PG_PASSWORD', ''),
             'driver': 'org.postgresql.Driver'}
    written = {}

    def write_pg(df, name, order=None):
        if args.local:
            written[name] = df.count()
            print(f'  [local] {name}: {df.count()} 行（跳过 PG）')
            return
        d = df.coalesce(2)
        d.write.jdbc(url, name, mode='overwrite', properties=props)
        n = df.count()
        written[name] = n
        print(f'  ✅ {name}: {n} 行 → PG')

    # ── 基础读入 ──
    d1 = spark.read.parquet(f'{wh}/dws_density_1pct')
    d5 = spark.read.parquet(f'{wh}/dws_density_5s')
    fp = spark.read.parquet(f'{wh}/dws_video_rhythm_features')
    sent = spark.read.parquet(f'{wh}/dws_sentiment_curve_1pct')
    ev = spark.read.parquet(f'{wh}/dws_rhythm_events')
    clus = spark.read.parquet(f'{wh}/dws_curve_clusters')
    pr = spark.read.parquet(f'{wh}/dws_partition_rhythm')
    onl = spark.read.parquet(f'{wh}/dws_onlist_track')
    up = spark.read.parquet(f'{wh}/dws_up_profile')
    evo = spark.read.parquet(f'{wh}/dws_evolution_feats')
    dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean').filter(~F.col('is_spam'))
           .select('bvid', 'batch_date', 'time_offset', 'text', 'attr_weight', 'text_len',
                   'is_ad', 'is_meaningless', 'offset_pct', 'lag_bucket'))
    dim = spark.read.parquet(f'{wh}/dim_video').select('bvid', 'tname', 'last_batch')

    dur = fp.select('bvid', 'batch_date', F.col('duration').cast('double').alias('duration'))

    # ── ① 曲线（1%）：cnt/wcnt + cal_cnt（5s 聚合）+ sent_avg + event_id ──
    cal1 = (d5.select('bvid', 'batch_date', 'bucket5s', 'cal_cnt')
            .join(dur, ['bvid', 'batch_date'])
            .withColumn('b1', F.least(F.floor((F.col('bucket5s') * 5.0 + 2.5)
                                              / F.col('duration') * 100.0), F.lit(99)).cast('int'))
            .groupBy('bvid', 'batch_date', 'b1').agg(F.sum('cal_cnt').alias('cal_cnt')))
    ev_b = (ev.select('bvid', 'batch_date', 'event_id', 'start_sec', 'end_sec')
            .join(dur, ['bvid', 'batch_date'])
            .withColumn('b_lo', F.least(F.floor(F.col('start_sec') / F.col('duration') * 100.0),
                                        F.lit(99)).cast('int'))
            .withColumn('b_hi', F.least(F.floor(F.col('end_sec') / F.col('duration') * 100.0),
                                        F.lit(99)).cast('int'))
            .withColumn('b', F.explode(F.sequence('b_lo', 'b_hi')))
            .groupBy('bvid', 'batch_date', 'b').agg(F.min('event_id').alias('event_id')))
    curve = (d1.select('bvid', 'batch_date', F.col('bucket1pct').alias('bucket'), 'cnt', 'wcnt')
             .join(sent.select('bvid', 'batch_date', F.col('bucket').alias('bucket'),
                               F.col('sent_avg')), ['bvid', 'batch_date', 'bucket'], 'left')
             .join(cal1.select('bvid', 'batch_date', F.col('b1').alias('bucket'), 'cal_cnt'),
                   ['bvid', 'batch_date', 'bucket'], 'left')
             .join(ev_b.select('bvid', 'batch_date', F.col('b').alias('bucket'), 'event_id'),
                   ['bvid', 'batch_date', 'bucket'], 'left')
             .select('bvid', 'batch_date', 'bucket', 'cnt', 'wcnt',
                     F.coalesce('cal_cnt', F.col('cnt')).alias('cal_cnt'),
                     'sent_avg', 'event_id'))
    write_pg(curve, 'ads_video_rhythm_curve')

    # ── ② 5s 曲线（懒加载）──
    write_pg(d5.select('bvid', 'batch_date', 'bucket5s', 'cnt', 'wcnt', 'cal_cnt'),
             'ads_video_curve_5s')

    # ── ③ 高光（事件级）──
    evs = (ev.select('bvid', 'batch_date', 'event_id', 'start_sec', 'end_sec', 'score_fused',
                     'score_d1', 'score_d2', 'score_d3', 'score_d4', 'n_methods',
                     'sync_score', 'sync_p', 'n_window', 'delta_hat_sec', 'calibrated_on'))
    ev_d = (dwd.filter(~F.col('is_ad') & ~F.col('is_meaningless'))
            .join(evs, ['bvid', 'batch_date'])
            .filter((F.col('time_offset') >= F.col('start_sec'))
                    & (F.col('time_offset') <= F.col('end_sec'))))
    wtop = Window.partitionBy('bvid', 'batch_date', 'event_id') \
        .orderBy(F.desc('attr_weight'), F.desc('text_len'))
    top = (ev_d.withColumn('rn', F.row_number().over(wtop)).filter(F.col('rn') <= 5)
           .groupBy('bvid', 'batch_date', 'event_id')
           .agg(F.sort_array(F.collect_list(F.struct('rn', 'text'))).alias('_ts')))
    top = top.withColumn('top_danmaku', F.transform('_ts', lambda x: x['text'])).drop('_ts')
    top3 = (ev_d.withColumn('rn', F.row_number().over(wtop)).filter(F.col('rn') <= 3)
            .groupBy('bvid', 'batch_date', 'event_id')
            .agg(F.sort_array(F.collect_list(F.struct('rn', 'text'))).alias('_t3')))
    top3 = top3.withColumn('summary', F.concat_ws(' / ',
                                                  F.transform('_t3', lambda x: F.substring(x['text'], 1, 40)))).drop('_t3')
    sraw = (spark.read.parquet(f'{wh}/dwd_danmaku_sent')
            .filter(~F.col('is_ultra_short'))
            .select('bvid', 'batch_date', 'sent_state', 'offset_pct'))
    sm = (sraw.join(dur, ['bvid', 'batch_date'])
          .withColumn('offset_sec', F.col('offset_pct') * F.col('duration'))
          .join(evs.select('bvid', 'batch_date', 'event_id', 'start_sec', 'end_sec'),
                ['bvid', 'batch_date'])
          .filter((F.col('offset_sec') >= F.col('start_sec'))
                  & (F.col('offset_sec') <= F.col('end_sec')))
          .groupBy('bvid', 'batch_date', 'event_id', 'sent_state').count()
          .groupBy('bvid', 'batch_date', 'event_id')
          .pivot('sent_state', ['pos', 'neg', 'neu', 'objective']).sum('count'))
    sm = (sm.withColumnRenamed('pos', 'n_pos').withColumnRenamed('neg', 'n_neg')
          .withColumnRenamed('neu', 'n_neu').withColumnRenamed('objective', 'n_obj'))
    hi = (evs.join(top, ['bvid', 'batch_date', 'event_id'], 'left')
          .join(top3, ['bvid', 'batch_date', 'event_id'], 'left')
          .join(sm, ['bvid', 'batch_date', 'event_id'], 'left')
          .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    write_pg(hi, 'ads_video_highlights')

    # ── ④ 指纹（16 特征 + z + cluster + tname）──
    fpd = (fp.join(dim.select('bvid', 'tname'), 'bvid', 'left')
           .join(clus.select('bvid', 'batch_date', 'cluster_id', 'k', 'silhouette',
                             F.col('cluster_id_kshape'), F.col('ari_kmeans_kshape')),
                 ['bvid', 'batch_date'], 'left')
           .join(evo.select('bvid', F.col('cluster_type').alias('evo_type')),
                 'bvid', 'left')
           .withColumn('cluster_method', F.lit('kmeans'))
           .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    sel = (['bvid', 'batch_date', 'tname', 'duration', 'n_danmaku'] + FEAT_COLS
           + [f'{c}_z_part' for c in FEAT_COLS]
           + ['cluster_id', 'cluster_method', 'k', 'silhouette', 'cluster_id_kshape',
              'ari_kmeans_kshape', 'evo_type', 'engine_version'])
    write_pg(fpd.select(*sel), 'ads_video_fingerprint')

    # ── ⑤ 分区基准 ──
    psel = (['tname', 'n_videos', 'low_support', 'curve_med', 'curve_p25', 'curve_p75']
            + [f'{c}_med' for c in FEAT_COLS] + [f'{c}_mad' for c in FEAT_COLS])
    write_pg(pr.select(*psel), 'ads_partition_fingerprint')

    # ── ⑥ 双时间轴（offset_bin × lag_bin；每 bvid 最新快照）──
    # ⚠️ lag→index 映射用 when 链（实测 array_position(lit(list), ...) 在容器 PySpark
    #    抛 NOT_ITERABLE——2026-10-13）；映射表仍以 LAG_ORDER 为单一事实源
    lag_expr = None
    for _i, _v in enumerate(LAG_ORDER):
        cond = F.col('lag_bucket') == _v
        lag_expr = F.when(cond, _i) if lag_expr is None else lag_expr.when(cond, _i)
    lag_expr = lag_expr.otherwise(F.lit(None).cast('int'))
    da = (dwd.join(dim.select('bvid', F.col('last_batch')), 'bvid')
          .filter(F.col('batch_date') == F.col('last_batch'))
          .filter(F.col('offset_pct').isNotNull() & F.col('lag_bucket').isNotNull())
          .withColumn('offset_bin', F.least(F.floor(F.col('offset_pct') * 20.0), F.lit(19)).cast('int'))
          .withColumn('lag_bin', lag_expr)
          .groupBy('bvid', 'offset_bin', 'lag_bin').agg(F.count('*').alias('cnt')))
    write_pg(da, 'ads_dual_axis')

    # ── ⑦ 榜单（latest / history）──
    hsel = ['bvid', 'batch_date', 'cadence', 'rank_proxy', 'title', 'tname', 'up_name',
            'play', 'danmaku_reported', 'source', 'prev_batch_date', 'rank_delta',
            'play_delta', 'danmaku_delta', 'video_fetched_at', 'days_elapsed', 'pair_note']
    write_pg(onl.select(*hsel), 'ads_hotlist_history')
    latest_bd = onl.agg(F.max('batch_date')).collect()[0][0]
    write_pg(onl.filter(F.col('batch_date') == latest_bd).select(*hsel), 'ads_hotlist_latest')

    # ── ⑧ UP 档案 / ⑨ 演化分型 / ⑩ 生命周期 ──
    write_pg(up, 'ads_up_profile')
    write_pg(evo.select('bvid', 'last_batch', 'n_obs', 'span_days', 'g_play', 'g_like',
                        'g_reported', 'rank_last', 'rank_delta', 'eng_shift',
                        'sat_last', 'cluster_id', 'cluster_type', 'silhouette',
                        'engine_version'), 'ads_evolution_feats')
    life = spark.read.parquet(f'{wh}/dws_lifecycle')
    write_pg(life.select('bvid', 'n_obs', 'first_batch', 'last_batch', 'span_days',
                         'play_last', 'danmaku_reported_last', 'rank_last',
                         'coverage_last', 'pool_saturation_last', 'series_admissible',
                         'g_play_daily', 'g_reported_daily',
                         'metric_halflife_d', 'metric_halflife_n_rates',
                         'n_days_obs', 'total_cnt', 'peak_day', 'peak_cnt',
                         'half_decay_days', 'halflife_d', 'halflife_src',
                         'halflife_rank', 'engine_version'), 'ads_lifecycle')

    report = {'job': 'ads_build', 'engine_version': ENGINE_VERSION,
              'tables': written, 'pg_target': f'{args.pg_host}/{args.pg_db}',
              'note': 'B0 min 链 ads_video_metrics/ads_video_curve/ads_batch_meta 未动（锚点页依赖）'}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'ads_build_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ ads_build：{len(written)} 张表 → PG')
    spark.stop()


if __name__ == '__main__':
    main()
