#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp62b_rollcheck.py — 用弹幕身份交集直接判定"池是否滚动"（P0-2 修复）

身份 = (bvid, cid, send_time, text)：跨批次稳定（row_id 语义未确认，不用）。
对每个多批次 bvid：E=最早批、L=最晚批；overlap_frac = |E∩L| / |E|。
  overlap_frac == 1  ⇒ 纯增量（delta = L−E 就是真实新增）
  overlap_frac < 1   ⇒ 有滚动/删除（delta 被污染）
输出：file:///opt/work/data-out/exp62/rollcheck.csv（bvid, nE, nL, overlap_frac）
"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

OUT = 'file:///opt/work/data-out/exp62'
spark = (SparkSession.builder.appName('exp62b_rollcheck')
         .config('spark.sql.shuffle.partitions', '16')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

d = spark.read.parquet('/warehouse/dwd_danmaku_clean').filter(~F.col('is_spam'))
rel = F.col('send_time').cast('long') - F.col('pubdate').cast('long')
base = (d.filter(F.col('send_time').isNotNull() & F.col('pubdate').isNotNull() & F.col('time_offset').isNotNull())
        .withColumn('age_max_h', F.round(rel / 3600.0, 2)))

# 多批次 bvid + 每支的最早/最晚批
per = base.groupBy('bvid', 'batch_date').agg(F.count('*').alias('n_pool'),
                                             F.max('age_max_h').alias('age_max_h'))
multi = per.groupBy('bvid').count().filter(F.col('count') >= 2).select('bvid')
w = Window.partitionBy('bvid').orderBy(F.col('age_max_h'))
be = per.join(multi, 'bvid', 'inner')
E = (be.withColumn('rk', F.row_number().over(w)).filter(F.col('rk') == 1)
     .select('bvid', F.col('batch_date').alias('b_e'), F.col('n_pool').alias('nE')))
L = (be.withColumn('rk', F.row_number().over(Window.partitionBy('bvid').orderBy(F.desc('age_max_h'))))
     .filter(F.col('rk') == 1)
     .select('bvid', F.col('batch_date').alias('b_l'), F.col('n_pool').alias('nL')))
pairs = E.join(L, 'bvid', 'inner')
print('pairs:', pairs.count(), flush=True)

rows = (base.join(pairs, 'bvid', 'inner')
        .filter((F.col('batch_date') == F.col('b_e')) | (F.col('batch_date') == F.col('b_l')))
        .withColumn('key', F.concat_ws('\x01', 'cid', 'send_time', 'time_offset', 'text'))
        .withColumn('side', F.when(F.col('batch_date') == F.col('b_e'), 'E').otherwise('L'))
        .select('bvid', 'key', 'side', 'b_e', 'b_l', 'nE', 'nL'))

# 每 (bvid,key)：是否同时在 E 与 L
both = (rows.groupBy('bvid', 'key')
        .agg(F.sum(F.when(F.col('side') == 'E', 1).otherwise(0)).alias('inE'),
             F.sum(F.when(F.col('side') == 'L', 1).otherwise(0)).alias('inL'),
             F.first('b_e').alias('b_e'), F.first('b_l').alias('b_l'),
             F.first('nE').alias('nE'), F.first('nL').alias('nL'))
        .withColumn('inE', F.col('inE').cast('int')).withColumn('inL', F.col('inL').cast('int')))

agg = (both.groupBy('bvid', 'b_e', 'b_l', 'nE', 'nL')
       .agg(F.sum(F.when((F.col('inE') > 0) & (F.col('inL') > 0), 1).otherwise(0)).alias('overlap'),
            F.sum(F.when(F.col('inE') > 0, 1).otherwise(0)).alias('nE_keys'))
       .withColumn('overlap_frac', F.round(F.col('overlap') / F.greatest(F.col('nE_keys'), F.lit(1)), 4)))
agg.coalesce(1).write.mode('overwrite').option('header', True).csv(f'{OUT}/rollcheck_csv')
print('rollcheck rows:', agg.count())
spark.stop()
