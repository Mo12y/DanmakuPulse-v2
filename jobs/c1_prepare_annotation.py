#!/usr/bin/env python3
"""
c1_prepare_annotation.py — C1 任务 B1/B3：样本清单 + 30 支分层抽样 + 标注材料（任务 B 前置）

输入（全部为 Spark 产物，不回读原始 CSV）：
  {wh}/dws_rhythm_60s_meta   视频粒度（含 c1_eligible / exclude_reason）
  {wh}/dws_rhythm_60s        桶粒度 60s 序列（含 bucket_state，区分"没有弹幕"与"采集后未知"）

输出（data-out/c1/，均不进 git；本脚本与协议进 git）：
  c1_pool.csv            全部单元 + included + exclude_reason（纳入/排除清单，任务 A 要求）
  sample30.csv           30 支分层抽样（bvid/tname/duration/got/桶数/video_url）
  curves.csv             每支样本的 60s 序列（辅助标注；以实际观看为准）
  ann_A.csv / ann_B.csv  双人独立标注表模板（空，列固定）
  manifest.json          抽样种子、分层、计数——可复现

抽样纪律：
  - 只用元数据（tname/duration/got/桶数）分层，**不接触任何算法分段输出**（防锚定）
  - 分层=按 tname 比例分配（大余数法）+ 每层至少 2 支（不足 2 支的层全取）
  - 层内按 bvid 排序取前 k（确定性顺序，结果可复现）
  - 同一 bvid 多批次只取 1 支（取 n_danmaku 最大的批次，即最完整快照）
"""
import csv
import json
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

SEED = 20261028
N_TARGET = 30
# ⚠️ spark-master 容器的默认 FS 是 HDFS：宿主路径必须带 file: 前缀，否则写进 HDFS
# （export_v3_danmaku.py 头注同款坑，2026-10-27 已踩过一次）
OUT = 'file:///opt/work/data-out/c1'
OUT_LOCAL = '/opt/work/data-out/c1'          # Python open() 用（driver 本地 = 宿主挂载）


def main():
    b = (SparkSession.builder.appName('c1_prepare_annotation')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai'))
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    wh = '/warehouse'

    meta = spark.read.parquet(f'{wh}/dws_rhythm_60s_meta')
    buckets = spark.read.parquet(f'{wh}/dws_rhythm_60s')

    # ---- 1) 全量纳入/排除清单 ----
    pool = (meta.withColumn('included', F.col('c1_eligible'))
            .withColumn('video_url', F.concat(F.lit('https://www.bilibili.com/video/'), F.col('bvid'))))
    pool.coalesce(1).select('bvid', 'batch_date', 'tname', 'duration', 'danmaku_got', 'part_count',
                            'is_multi_part', 'n_buckets_expected', 'n_buckets_nonzero', 'n_zero_observed',
                            'n_post_snapshot', 'n_interior_holes', 'coverage_ratio', 'fetch_available',
                            'included', 'exclude_reason', 'video_url') \
        .orderBy(F.desc('included'), 'tname', 'bvid') \
        .write.mode('overwrite').option('header', True).csv(f'{OUT}/c1_pool_csv')

    # ---- 2) 同 bvid 去重（取 n_danmaku 最大的批次）----
    elig = (pool.filter(F.col('included'))
            .withColumn('rk', F.row_number().over(
                Window.partitionBy('bvid').orderBy(F.desc('n_danmaku'))))
            .filter(F.col('rk') == 1).drop('rk'))
    n_elig_videos = elig.count()

    # ---- 3) 分层抽样（只用元数据）----
    layers = (elig.groupBy('tname').agg(F.count('*').alias('n')).orderBy(F.desc('n')))
    layers_local = [(r['tname'], int(r['n'])) for r in layers.collect()]
    total = sum(n for _, n in layers_local)
    # 配额：纯比例（floor + 大余数法），总数恰为 N_TARGET。
    # 不设"每层至少 2"：tname 长尾极长（数十个分区、多数只有 1–3 支），强制下限会超配 2 倍以上。
    # 长尾分区按比例自然获得 0/1 支——这是分层抽样的标准行为，如实记录。
    quota = {t: N_TARGET * n / total for t, n in layers_local}
    alloc = {t: int(quota[t] // 1) for t, _ in layers_local}
    rem = N_TARGET - sum(alloc.values())
    for t, _ in sorted(layers_local, key=lambda x: -(quota[x[0]] - int(quota[x[0]]))):
        if rem <= 0:
            break
        alloc[t] += 1
        rem -= 1
    assert sum(alloc.values()) == N_TARGET, f'配额总数 {sum(alloc.values())} != {N_TARGET}'
    assert all(alloc[t] <= n for t, n in layers_local), '配额超过层容量'

    parts = [elig.filter(F.col('tname') == t).orderBy('bvid').limit(k) for t, k in alloc.items() if k > 0]
    sampled = parts[0]
    for p in parts[1:]:
        sampled = sampled.unionByName(p)
    sampled = sampled.orderBy('tname', 'bvid')

    sampled.coalesce(1).select('bvid', 'batch_date', 'tname', 'duration', 'danmaku_got',
                               'n_buckets_expected', 'n_buckets_nonzero', 'n_zero_observed',
                               'n_post_snapshot', 'n_interior_holes', 'video_url') \
        .write.mode('overwrite').option('header', True).csv(f'{OUT}/sample30_csv')

    # ---- 4) 每支样本的 60s 曲线（辅助标注）----
    curves = (buckets.join(sampled.select('bvid', 'batch_date'), ['bvid', 'batch_date'], 'inner')
              .orderBy('bvid', 'bucket60'))
    curves.coalesce(1).select('bvid', 'bucket60', 't_start_s', 't_end_s', 'cnt', 'wcnt',
                              'bucket_state') \
        .write.mode('overwrite').option('header', True).csv(f'{OUT}/curves_csv')

    # ---- 5) 标注表模板（列冻结，与协议 v1.0 一致）----
    os.makedirs(OUT_LOCAL, exist_ok=True)
    hdr = ['bvid', 'seg_id', 'start_bucket', 'end_bucket', 'category', 'confidence', 'note']
    for who in ('A', 'B'):
        with open(f'{OUT_LOCAL}/ann_{who}.csv', 'w', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow(hdr)

    manifest = {
        'seed': SEED, 'n_target': N_TARGET,
        'eligible_units': int(pool.filter(F.col('included')).count()),
        'eligible_videos': int(n_elig_videos),
        'sampled': int(sampled.count()),
        'alloc': alloc,
        'layers_top20': layers_local[:20],
        'rules': '分层=tname 比例（大余数法）+每层≥2；层内按 bvid 排序取前 k；'
                 '同 bvid 多批取 n_danmaku 最大批；抽样只用元数据，未接触任何算法输出',
        'sheet_columns': hdr,
        'category_set': ['L', 'R', 'H', 'B', 'D', 'X'],
        'protocol': 'docs/c1_annotation_protocol.md v1.0（冻结于查看任何算法分段输出之前）',
    }
    with open(f'{OUT_LOCAL}/manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: manifest[k] for k in ('seed', 'eligible_units', 'eligible_videos', 'sampled', 'alloc')},
                     ensure_ascii=False, indent=1))
    print('✅ c1_prepare_annotation 完成 →', OUT_LOCAL)
    spark.stop()


if __name__ == '__main__':
    main()
