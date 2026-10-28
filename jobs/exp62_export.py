#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp62_export.py — 导出多批次视频的 1% 桶计数矩阵（跨快照信号用）

范围：出现于 ≥2 个批次的 bvid（576 个），全部批次全量。
输出：file:///opt/work/data-out/exp62/curves_multi.csv（bvid, batch_date, bucket1pct, cnt）
     + batch_meta.csv（bvid, batch_date, age_h 供排序）
"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

OUT = 'file:///opt/work/data-out/exp62'
spark = (SparkSession.builder.appName('exp62_export')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

d = spark.read.parquet('/warehouse/dwd_danmaku_clean').filter(~F.col('is_spam'))
rel = F.col('send_time').cast('long') - F.col('pubdate').cast('long')
cnt = (d.filter(F.col('offset_pct').isNotNull())
       .withColumn('bucket1pct', F.least(F.floor(F.col('offset_pct') * 100), F.lit(99)).cast('int'))
       .withColumn('age_h', F.round(rel / 3600.0, 2)))
per = (cnt.groupBy('bvid', 'batch_date')
       .agg(F.count('*').alias('n_pool'), F.min('age_h').alias('age_min_h'),
            F.max('age_h').alias('age_max_h')))
multi = per.groupBy('bvid').count().filter(F.col('count') >= 2).select('bvid')
keep = per.join(multi, 'bvid', 'inner')
cur = (cnt.join(keep.select('bvid', 'batch_date'), ['bvid', 'batch_date'], 'inner')
       .groupBy('bvid', 'batch_date', 'bucket1pct').agg(F.count('*').alias('cnt')))
cur.coalesce(1).write.mode('overwrite').option('header', True).csv(f'{OUT}/curves_multi_csv')
keep.coalesce(1).write.mode('overwrite').option('header', True).csv(f'{OUT}/batch_meta_csv')
print('bvids=', keep.select('bvid').distinct().count(), 'units=', keep.count(), 'curve_rows=', cur.count())
spark.stop()
