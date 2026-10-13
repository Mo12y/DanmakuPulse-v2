#!/usr/bin/env python3
"""
m2_l1_baseline.py — L1 分区基准库构建器（08 §3.3）

输入：dwd_danmaku_clean（tname + is_spam=false）+ dws_video_rhythm_features（指纹）
输出：{wh}/dws_partition_rhythm —— 每 tname 一行：
  - curve_med / curve_p25 / curve_p75：100 维（1% 桶）中位曲线与包络（跨视频每桶 cnt 的 percentile）
  - 各指纹特征的 <feat>_med / <feat>_mad（稳健 z 的分母，MAD×1.4826）
  - n_videos、low_support（<20 → z 值置空的依据）、engine_version
口径（08 §3.3）：滚动全量重算、幂等；percentile 用精确 `percentile(..., 0.5/0.25/0.75)`
（DoD「基准与 pandas 复算一致率 100%」——复算脚本用相同 percentile 定义）。
"""
import argparse

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b2-v1'
FEAT_COLS = ['dm_per_min', 'wdm_per_min', 'rhythm_entropy', 'burst_cv', 'w_burst_cv',
             'early_lag_ratio', 'color_ratio', 'top_mode_ratio',
             'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio', 'avg_text_len',
             'opening_5pct', 'ending_5pct', 'mid_ratio']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--min-videos', type=int, default=20)
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('m2_l1_baseline')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean')
           .filter(~F.col('is_spam'))
           .select('bvid', 'batch_date', 'tname', 'offset_pct'))

    # ── 中位曲线（每视频每桶 cnt → 分区内每桶 percentile）──
    c2 = (dwd.filter(F.col('offset_pct').isNotNull())
          .withColumn('bucket1pct', F.least(F.floor(F.col('offset_pct') * 100), F.lit(99)).cast('int'))
          .groupBy('tname', 'bvid', 'bucket1pct')
          .agg(F.count('*').alias('cnt')))
    curve = (c2.groupBy('tname', 'bucket1pct')
             .agg(F.expr('percentile(cnt, 0.5)').alias('med'),
                  F.expr('percentile(cnt, 0.25)').alias('p25'),
                  F.expr('percentile(cnt, 0.75)').alias('p75')))
    curve_arr = (curve.groupBy('tname')
                 .agg(F.sort_array(F.collect_list(F.struct('bucket1pct', 'med'))).alias('_cm'),
                      F.sort_array(F.collect_list(F.struct('bucket1pct', 'p25'))).alias('_c25'),
                      F.sort_array(F.collect_list(F.struct('bucket1pct', 'p75'))).alias('_c75'))
                 .withColumn('curve_med', F.col('_cm.med'))
                 .withColumn('curve_p25', F.col('_c25.p25'))
                 .withColumn('curve_p75', F.col('_c75.p75'))
                 .select('tname', 'curve_med', 'curve_p25', 'curve_p75'))

    # ── 指纹特征分区 median/MAD ──
    tn = dwd.select('bvid', 'batch_date', 'tname').distinct()
    ft = spark.read.parquet(f'{wh}/dws_video_rhythm_features').join(tn, ['bvid', 'batch_date'])
    med_aggs = [F.expr(f'percentile({c}, 0.5)').alias(f'{c}_med') for c in FEAT_COLS]
    m1 = ft.groupBy('tname').agg(*med_aggs)
    f2 = ft.join(m1, 'tname')
    mad_aggs = [F.expr(f'percentile(abs({c} - {c}_med), 0.5)').alias(f'{c}_mad') for c in FEAT_COLS]
    m2 = f2.groupBy('tname').agg(*mad_aggs)
    # MAD×1.4826 → 稳健尺度
    m2 = m2.select('tname', *[F.col(f'{c}_mad').alias(f'{c}_mad') for c in FEAT_COLS])
    for c in FEAT_COLS:
        m2 = m2.withColumn(f'{c}_mad', F.col(f'{c}_mad') * 1.4826)

    # ── n_videos + low_support ──
    nv = dwd.select('bvid', 'tname').distinct().groupBy('tname').agg(F.count('*').alias('n_videos'))

    out = (curve_arr.join(m1, 'tname', 'left')
           .join(m2, 'tname', 'left')
           .join(nv, 'tname', 'left')
           .withColumn('low_support', F.col('n_videos') < args.min_videos)
           .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    out.write.mode('overwrite').parquet(f'{wh}/dws_partition_rhythm')

    n = out.count()
    n_low = out.filter('low_support').count()
    print(f'✅ m2_l1_baseline：{n} 个分区（low_support {n_low} 个 <{args.min_videos} 视频）')
    spark.stop()


if __name__ == '__main__':
    main()
