#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""c1_export_text.py — 导出 30 支标注视频的逐条弹幕（文本 + 时间位置），供 exp54 内容×节奏分析

输入：/warehouse/dwd_danmaku_clean（Spark 清洗产物，is_spam=false）
     file:/opt/work/data-out/c1/sample30_csv/（C1 30 支清单，bvid+batch_date）
输出：file:/opt/work/data-out/c1/text30/  单 CSV：bvid,batch_date,bucket60,text,user_hash,text_len,mode
范围：仅 sample30 的 30 个 (bvid,batch)；~1.4 万行，量小。
口径：bucket60 = floor(time_offset/60)，与 dws_rhythm_60s 一致；不截断（分析侧按视频桶数裁剪）。
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

OUT = 'file:///opt/work/data-out/c1/text30'

spark = (SparkSession.builder.appName('c1_export_text')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

sample = (spark.read.option('header', True).csv('file:///opt/work/data-out/c1/sample30_csv')
          .select('bvid', 'batch_date'))
n_sample = sample.count()

d = (spark.read.parquet('/warehouse/dwd_danmaku_clean')
     .filter(~F.col('is_spam'))
     .join(sample, ['bvid', 'batch_date'], 'inner')
     .filter(F.col('time_offset').isNotNull() & F.col('text').isNotNull())
     .withColumn('bucket60', F.floor(F.col('time_offset') / 60).cast('long'))
     .withColumn('txt', F.regexp_replace(F.col('text'), r'[\r\n\t]+', ' '))
     .select('bvid', 'batch_date', 'bucket60', 'txt', 'user_hash', 'text_len', 'mode'))

n = d.count()
nv = d.select('bvid').distinct().count()
(d.coalesce(1).write.mode('overwrite').option('header', True)
 .option('quote', '"').option('escape', '"').csv(OUT))
print(f'✅ c1_export_text：{n:,} 行 / {nv} 支（sample30={n_sample}）→ {OUT}')
spark.stop()
