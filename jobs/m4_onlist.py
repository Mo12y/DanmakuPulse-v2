#!/usr/bin/env python3
"""
m4_onlist.py — M4 上榜追踪（08 §7）

输入：ods_hotlist_video（快照 rank_proxy/指标）+ ods_batch_meta（cadence 自描述）
输出：{wh}/dws_onlist_track —— 每 (bvid, batch_date) 一行：
  rank_proxy / play / like / coin / favorite / share / reply / danmaku_reported
  + title / tname / up_mid / up_name / source / cadence
  + prev_batch_date / rank_delta / play_delta / danmaku_delta（增速）

⚠️ 冻结决策纪律（10 §2.3 / 11 §2「历史批次用途」）：4 批周频（cadence=weekly_legacy）
   不参与任何时序计算 ⇒ 跨批差分**仅在同 cadence 连续批对间计算**（当前=仅 daily）；
   legacy 行的 delta 列显式 NULL——rank 事实本身保留（与 dwd_video_metric_snapshot 同性质），
   消费方按 cadence 列自行过滤。
差异口径：rank_delta = prev.rank_proxy − cur.rank_proxy（>0 = 名次上升）；
   play_danmaku 增速 = cur − prev（数据天然递增，负值=缩水/删档，保留不裁剪）。
"""
import argparse
import json
import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

ENGINE_VERSION = 'b2-v1'
KEEP = ['bvid', 'batch_date', 'rank_proxy', 'play', 'like', 'coin', 'favorite',
        'share', 'reply', 'danmaku_reported', 'title', 'tname', 'up_mid', 'up_name', 'source']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('m4_onlist')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    hot = (spark.read.parquet(f'{wh}/ods_hotlist_video')
           .select(*KEEP)
           .withColumn('batch_date', F.col('batch_date').cast('int')))
    meta = (spark.read.parquet(f'{wh}/ods_batch_meta')
            .select(F.col('batch_date').cast('int').alias('batch_date'),
                    F.coalesce(F.col('cadence'), F.lit('daily')).alias('cadence')))
    hot = hot.join(meta, 'batch_date', 'left').withColumn(
        'cadence', F.coalesce(F.col('cadence'), F.lit('daily')))

    # ── 批序：同 cadence 类内的紧邻前批（仅 daily 计算差分；weekly_legacy 显式不做）──
    batches = sorted([(r['batch_date'], r['cadence'])
                      for r in hot.select('batch_date', 'cadence').distinct().collect()])
    prev_pairs = []       # (cur_bd, prev_bd) —— 同 cadence 类内紧邻
    by_cad = {}
    for bd, cad in batches:
        by_cad.setdefault(cad, []).append(bd)
    for cad, bds in by_cad.items():
        if cad != 'daily':
            continue           # 冻结决策：legacy 不做跨批差分
        bds = sorted(bds)
        for i in range(1, len(bds)):
            prev_pairs.append((bds[i], bds[i - 1]))

    cur = hot
    if prev_pairs:
        pairs = spark.createDataFrame(prev_pairs, ['batch_date', 'prev_batch_date'])
        prev = hot.select(
            'bvid', F.col('batch_date').alias('prev_batch_date'),
            F.col('rank_proxy').alias('p_rank'), F.col('play').alias('p_play'),
            F.col('danmaku_reported').alias('p_dm'))
        cur = (cur.join(pairs, 'batch_date', 'left')
               .join(prev, ['bvid', 'prev_batch_date'], 'left'))
    else:
        cur = (cur.withColumn('prev_batch_date', F.lit(None).cast('int'))
               .withColumn('p_rank', F.lit(None).cast('int'))
               .withColumn('p_play', F.lit(None).cast('long'))
               .withColumn('p_dm', F.lit(None).cast('long')))

    out = (cur.withColumn('rank_delta', F.col('p_rank') - F.col('rank_proxy'))
           .withColumn('play_delta', F.col('play') - F.col('p_play'))
           .withColumn('danmaku_delta', F.col('danmaku_reported') - F.col('p_dm'))
           # prev_batch_date 仅在「前批确有该视频（可比较）」时保留（新上榜 → NULL）
           .withColumn('prev_batch_date',
                       F.when(F.col('p_rank').isNotNull(), F.col('prev_batch_date')))
           .withColumn('engine_version', F.lit(ENGINE_VERSION))
           .select('bvid', 'batch_date', 'cadence', 'rank_proxy', 'play', 'like', 'coin',
                   'favorite', 'share', 'reply', 'danmaku_reported', 'title', 'tname',
                   'up_mid', 'up_name', 'source', 'prev_batch_date', 'rank_delta',
                   'play_delta', 'danmaku_delta', 'engine_version'))
    out.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_onlist_track')
    n = out.count()

    per_batch = {int(r['batch_date']): r['count'] for r in
                 out.groupBy('batch_date').count().collect()}
    n_delta = out.filter(F.col('rank_delta').isNotNull()).count()
    report = {
        'job': 'm4_onlist', 'engine_version': ENGINE_VERSION,
        'n_rows': n, 'per_batch': per_batch,
        'n_delta_computed': n_delta,
        'prev_pairs_daily': prev_pairs,
        'note': ('冻结决策：weekly_legacy 不做跨批差分（delta 列 NULL）；'
                 'delta 仅在同 cadence 连续批对间计算（当前 daily 批对为空 → 全部 NULL，'
                 '待 daily 批次（20261012/13…）进入 ods_hotlist_video 后自动生效）'),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm4_onlist_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m4_onlist：{n} 行（{per_batch}）；delta 计算 {n_delta} 行')
    spark.stop()


if __name__ == '__main__':
    main()
