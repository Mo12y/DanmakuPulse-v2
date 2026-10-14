#!/usr/bin/env python3
"""
m4_track.py — M4 跨批轨迹（dws_metric_track；2026-10-14 收口批次 ④⑤）

背景（实锤机制，2026-10-14）：B 站弹幕接口对饱和视频返回固定容量「最近 N 条」滑动窗
（danmaku_got 全库量子化在 600 的整数倍：1200×100、3600×56、1800×39…；同一视频跨批
reported 增长而 got 不变）。danmaku_got 不是池的有偏样本、是窗的落点，无法靠「减掉积累」
修正 ⇒ 跨批水平/形状/半衰期一律改用可观测量：
  play / like / coin / favorite / share / reply / rank_proxy / danmaku_reported。

输入：ods_hotlist_video（榜单快照：rank/指标）+ ods_batch_meta（cadence）
      + dwd_video_metric_snapshot（danmaku_got → coverage / pool_saturation；采集到的视频才有）
      + ods_video_fetch_times（视频级采集时刻；缺失批回落 batch 级假设并标注）
输出：{wh}/dws_metric_track —— 每 (bvid, batch_date) 一行：
  rank/7 指标/danmaku_got/coverage/pool_saturation + video_fetched_at/fetched_at_src
  + prev_batch_date / days_elapsed / pair_valid / pair_note
  + d_*（7 指标增量）+ g_*（几何日均增长率）+ rank_delta / rank_rate
口径：
  - **delta 一律按采集时刻记账**（days_elapsed = Δvideo_fetched_at / 86400），不用 batch_date；
  - pair 仅在同 cadence 的紧邻日频批对间计算；**采集时刻倒挂（t_cur ≤ t_prev）→ pair_valid=false、
    增量置空** —— 20261012 第三窗口（10-13 17:43–18:45 续采的 45 个视频）由此自动排除出跨批差分；
  - weekly_legacy 不做跨批差分（冻结决策不变）；danmaku_got 只进 coverage/saturation，不进指标列。
"""
import argparse
import json
import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

ENGINE_VERSION = 'm4-v2'
METRICS = ['play', 'like', 'coin', 'favorite', 'share', 'reply', 'danmaku_reported']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('m4_track')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # ── 1) 榜单快照 + cadence ──
    hot_cols = ['bvid', 'batch_date', 'rank_proxy', 'play', 'like', 'coin', 'favorite',
                'share', 'reply', 'danmaku_reported', 'title', 'tname', 'up_mid',
                'up_name', 'source']
    hot = (spark.read.parquet(f'{wh}/ods_hotlist_video').select(*hot_cols)
           .withColumn('batch_date', F.col('batch_date').cast('int')))
    meta = (spark.read.parquet(f'{wh}/ods_batch_meta')
            .select(F.col('batch_date').cast('int').alias('batch_date'),
                    F.coalesce(F.col('cadence'), F.lit('daily')).alias('cadence')))
    hot = hot.join(meta, 'batch_date', 'left').withColumn(
        'cadence', F.coalesce(F.col('cadence'), F.lit('daily')))

    # ── 2) coverage / pool_saturation（danmaku_got 只作此用途）──
    try:
        snap = (spark.read.parquet(f'{wh}/dwd_video_metric_snapshot')
                .select('bvid', F.col('batch_date').cast('int').alias('batch_date'),
                        F.col('danmaku_got').alias('danmaku_got'),
                        F.col('danmaku_reported').alias('rpt_snap')))
        snap = (snap.withColumn('coverage',
                                F.least(F.col('danmaku_got') / F.greatest(F.col('rpt_snap'), F.lit(1)),
                                        F.lit(1.0)))
                .withColumn('pool_saturation', F.greatest(F.lit(1.0) - F.col('coverage'),
                                                          F.lit(0.0)))
                .drop('rpt_snap'))
        hot = hot.join(snap, ['bvid', 'batch_date'], 'left')
    except Exception as e:
        hot = (hot.withColumn('danmaku_got', F.lit(None).cast('int'))
               .withColumn('coverage', F.lit(None).cast('double'))
               .withColumn('pool_saturation', F.lit(None).cast('double')))
        print(f'[warn] metric_snapshot 不可读（{type(e).__name__}）→ coverage/saturation 置空')

    # ── 3) 视频级采集时刻（缺失批回落并标注）──
    try:
        ft = (spark.read.parquet(f'{wh}/ods_video_fetch_times')
              .select('bvid', F.col('batch_date').cast('int').alias('batch_date'),
                      F.col('fetched_at').alias('video_fetched_at'),
                      F.col('src').alias('fetched_at_src')))
        hot = (hot.join(ft, ['bvid', 'batch_date'], 'left')
               .withColumn('t_cur', F.unix_timestamp(F.to_timestamp('video_fetched_at'))))
        hot = hot.withColumn('fetched_at_src', F.coalesce(F.col('fetched_at_src'),
                                                          F.lit('none')))
        print('[ft] ods_video_fetch_times 已接入')
    except Exception as e:
        hot = (hot.withColumn('video_fetched_at', F.lit(None).cast('string'))
               .withColumn('t_cur', F.lit(None).cast('long'))
               .withColumn('fetched_at_src', F.lit('none')))
        print(f'[ft] 无采集时刻表（{type(e).__name__}）→ 批级回退（pair_note=assumed_batch_order）')

    # ── 4) 日频紧邻批对 ──
    batches = sorted([(r['batch_date'], r['cadence'])
                      for r in hot.select('batch_date', 'cadence').distinct().collect()])
    by_cad = {}
    for bd, cad in batches:
        by_cad.setdefault(cad, []).append(bd)
    prev_pairs = []
    for cad, bds in by_cad.items():
        if cad != 'daily':
            continue                      # 冻结决策：legacy 不做跨批差分
        bds = sorted(bds)
        for i in range(1, len(bds)):
            prev_pairs.append((bds[i], bds[i - 1]))

    if prev_pairs:
        pairs = spark.createDataFrame(prev_pairs, ['batch_date', 'prev_batch_date'])
        prev = hot.select(
            'bvid', F.col('batch_date').alias('prev_batch_date'),
            *[F.col(m).alias('p_' + m) for m in METRICS],
            F.col('danmaku_got').alias('p_got'),
            F.col('rank_proxy').alias('p_rank'), F.col('t_cur').alias('p_t'))
        cur = (hot.join(pairs, 'batch_date', 'left')
               .join(prev, ['bvid', 'prev_batch_date'], 'left'))
    else:
        cur = hot.withColumn('prev_batch_date', F.lit(None).cast('int'))
        for c in [f'p_{m}' for m in METRICS] + ['p_got', 'p_t']:
            cur = cur.withColumn(c, F.lit(None).cast('long'))
        cur = cur.withColumn('p_rank', F.lit(None).cast('int'))

    # ── 5) 有效性（采集时刻倒挂排除）+ 增量/增长率（按采集时刻记账）──
    pair_exists = F.col('p_rank').isNotNull()
    t_both = F.col('t_cur').isNotNull() & F.col('p_t').isNotNull()
    order_bad = t_both & (F.col('t_cur') <= F.col('p_t'))
    usable = pair_exists & (~order_bad)

    out = cur
    for m in METRICS:
        out = out.withColumn('d_' + m,
                             F.when(usable, F.col(m) - F.col('p_' + m)))
        endp = F.col(m).cast('double')
        stp = F.col('p_' + m).cast('double')
        span = (F.col('t_cur') - F.col('p_t')) / F.lit(86400.0)
        out = out.withColumn(
            'g_' + m,
            F.when(usable & (span > 0) & (stp > 0),
                   F.pow(endp / stp, F.lit(1.0) / span) - F.lit(1.0)))
    out = out.withColumn('d_danmaku_got',
                         F.when(usable, F.col('danmaku_got') - F.col('p_got')))
    out = (out
           .withColumn('rank_delta', F.when(usable, F.col('p_rank') - F.col('rank_proxy')))
           .withColumn('days_elapsed',
                       F.when(~pair_exists, F.lit(None).cast('double'))
                       .when(order_bad, F.lit(None).cast('double'))
                       .when(t_both, (F.col('t_cur') - F.col('p_t')) / F.lit(86400.0))
                       .otherwise((F.col('batch_date') - F.col('prev_batch_date')).cast('double')))
           .withColumn('rank_rate', F.when(usable & (F.col('days_elapsed') > 0),
                                           F.col('rank_delta') / F.col('days_elapsed')))
           .withColumn('pair_valid', usable)
           .withColumn('pair_note',
                       F.when(~pair_exists, F.lit('no_prev'))
                       .when(order_bad, F.lit('excluded_fetch_order'))
                       .when(t_both, F.lit('ok'))
                       .otherwise(F.lit('assumed_batch_order')))
           .withColumn('prev_batch_date', F.when(pair_exists, F.col('prev_batch_date')))
           .withColumn('engine_version', F.lit(ENGINE_VERSION)))

    cols = (['bvid', 'batch_date', 'cadence'] + METRICS +
            ['danmaku_got', 'coverage', 'pool_saturation', 'video_fetched_at',
             'fetched_at_src', 'prev_batch_date', 'days_elapsed', 'pair_valid', 'pair_note',
             'rank_proxy', 'rank_delta', 'rank_rate'] +
            ['d_' + m for m in METRICS] + ['d_danmaku_got'] +
            ['g_' + m for m in METRICS] +
            ['title', 'tname', 'up_mid', 'up_name', 'source', 'engine_version'])
    out = out.select(*cols)
    out.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_metric_track')
    n = out.count()

    per_batch = {str(r['batch_date']): r['count']
                 for r in out.groupBy('batch_date').count().collect()}
    n_valid = out.filter(F.col('pair_valid')).count()
    n_ok = out.filter(F.col('pair_note') == 'ok').count()
    n_excl = out.filter(F.col('pair_note') == 'excluded_fetch_order').count()
    n_assumed = out.filter(F.col('pair_note') == 'assumed_batch_order').count()
    # 机制回显（证据自检）：reported 涨而 got 平的 pair 数
    n_flat_got = out.filter(F.col('pair_valid') & (F.col('d_danmaku_got') == 0)
                            & (F.col('d_danmaku_reported') > 0)).count()
    days_stats = out.filter(F.col('days_elapsed').isNotNull()) \
                    .agg(F.min('days_elapsed'), F.max('days_elapsed')).collect()[0]
    report = {
        'job': 'm4_track', 'engine_version': ENGINE_VERSION,
        'n_rows': n, 'per_batch': per_batch,
        'prev_pairs_daily': prev_pairs,
        'n_pair_valid': n_valid, 'n_pair_ok': n_ok,
        'n_excluded_fetch_order': n_excl,
        'n_assumed_batch_order': n_assumed,
        'pair_semantics': 'n_pair_valid = ok + assumed（可用差分）；n_pair_ok = 双方采集时刻齐备且顺序正确',
        'n_reported_up_got_flat': n_flat_got,
        'days_elapsed_min': days_stats[0], 'days_elapsed_max': days_stats[1],
        'note': ('delta 按采集时刻记账（Δvideo_fetched_at）；采集时刻倒挂 pair 置空；'
                 'weekly_legacy 不做差分；danmaku_got 只进 coverage/pool_saturation'),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm4_track_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m4_track：{n} 行；有效 pair {n_valid}；倒挂排除 {n_excl}；'
          f'reported↑got 平 {n_flat_got}')
    spark.stop()


if __name__ == '__main__':
    main()
