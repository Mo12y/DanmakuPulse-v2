#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp61_audit_e1_censoring.py — D1 关键特征 c0 是否被"弹幕池只留最近 N 条"截断

问题：D1 的 c0 = [0,6h) 内发送的弹幕数。若池只保留最近 N 条，则最早发送的 [0,6h) 弹幕
     最可能被挤出，c0 被系统性低估——而 Q5（高弹幕量）恰恰误差最大（MAE 525）。
检验：per video 计算 share6 = n(send_time-pubdate ≤6h) / n_pool，按 got 分位看 share6 是否随量下降。
输出：file:///opt/work/data-out/exp61/censor.csv
"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

OUT = 'file:///opt/work/data-out/exp61'
spark = (SparkSession.builder.appName('exp61_e1')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

d = spark.read.parquet('/warehouse/dwd_danmaku_clean').filter(~F.col('is_spam'))
rel = F.col('send_time').cast('long') - F.col('pubdate').cast('long')
g = (d.filter(F.col('send_time').isNotNull() & F.col('pubdate').isNotNull())
     .groupBy('bvid', 'batch_date')
     .agg(F.first('danmaku_got').cast('long').alias('got'),
          F.count('*').alias('n_pool'),
          F.sum(F.when(rel <= 6 * 3600, 1).otherwise(0)).alias('n6'),
          F.sum(F.when(rel <= 24 * 3600, 1).otherwise(0)).alias('n24'),
          F.max(rel).alias('max_rel_s')))
g = (g.withColumn('share6', F.round(F.col('n6') / F.col('n_pool'), 5))
     .withColumn('share24', F.round(F.col('n24') / F.col('n_pool'), 5)))
g.coalesce(1).write.mode('overwrite').option('header', True).csv(f'{OUT}/censor_csv')
print('rows=', g.count())
spark.stop()
