#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp58_export.py — 导出 per-unit 节奏特征 + up_mid（分区基准偏倚审计用）

输出：file:///opt/work/data-out/exp58/units_feats.csv
列：bvid, batch_date, tname, up_mid, n_danmaku + 6 个审计指标
口径：dws_video_rhythm_features 全量 2,948 单元（不提前去重——B0 必须按原样复现），
     up_mid 取自 dim_video（每 bvid 一行）。
"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

OUT = 'file:///opt/work/data-out/exp58'
FEATS = ['dm_per_min', 'rhythm_entropy', 'burst_cv', 'opening_5pct', 'mid_ratio', 'avg_text_len']

spark = (SparkSession.builder.appName('exp58_export')
         .config('spark.sql.shuffle.partitions', '8').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

f = spark.read.parquet('/warehouse/dws_video_rhythm_features')
dim = spark.read.parquet('/warehouse/dim_video').select('bvid', 'up_mid', 'tname')  # 特征表无 tname
d = (f.join(dim, 'bvid', 'left')
     .select('bvid', 'batch_date', F.col('tname'), 'up_mid', 'n_danmaku', *FEATS))
n = d.count()
(miss := d.filter(F.col('up_mid').isNull()).count())
d.coalesce(1).write.mode('overwrite').option('header', True).csv(f'{OUT}/units_feats_csv')
print(f'✅ exp58_export：rows={n} up_mid_missing={miss} → {OUT}')
spark.stop()
