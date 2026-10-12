#!/usr/bin/env python3
"""probe_dedup_keys.py — 去重键对比实验（B1.1 前置决策，2026-10-12）

对比两种去重键在历史批次数据上的结果（宿主机 venv，local 模式，只读）：
  A（B0/旧口径）: (batch_date, bvid, text, time_offset)
  B（08 §1.4 新键）: (batch_date, bvid, cid, row_id)
输出：各键下行数 + row_id/cid 的 null/唯一性诊断 + 旧键重复组样例。
"""
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = f'file://{ROOT}/data/raw-local'

spark = (SparkSession.builder.master('local[2]').appName('probe_dedup_keys')
         .config('spark.driver.memory', '2g')
         .config('spark.sql.shuffle.partitions', '8').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

df = (spark.read.option('header', True).option('quote', '"').option('escape', '"')
      .csv(f'{RAW}/batch_date=*/danmaku.csv'))
c0 = df.columns[0]
if c0.lstrip('\ufeff') != c0:
    df = df.withColumnRenamed(c0, c0.lstrip('\ufeff'))
df = (df.withColumn('time_offset', F.col('time_offset').cast('double'))
      .filter(F.col('time_offset').isNotNull())
      .withColumn('batch_date', F.regexp_extract(F.input_file_name(), r'batch_date=(\d{8})', 1)))

ok = df.filter(F.col('status') == 'ok')
n_raw = ok.count()
a = ok.dropDuplicates(['batch_date', 'bvid', 'text', 'time_offset']).count()
b = ok.dropDuplicates(['batch_date', 'bvid', 'cid', 'row_id']).count()
print('══════════ 去重键对比（历史 4 批）══════════')
print(f'raw(status=ok, offset 非空)      = {n_raw:,}')
print(f'A 旧键 (bd,bvid,text,offset)     = {a:,}   （去重 -{n_raw - a:,}）')
print(f'B 新键 (bd,bvid,cid,row_id)      = {b:,}   （去重 -{n_raw - b:,}）')
print(f'差（B - A）                       = {b - a:+,}')

print('══════════ row_id / cid 诊断 ══════════')
for col in ['row_id', 'cid']:
    n_null = ok.filter(F.col(col).isNull()).count()
    n_dist = ok.select(col).distinct().count()
    n_empty = ok.filter(F.col(col) == '').count()
    print(f'{col}: null={n_null:,} empty={n_empty:,} distinct={n_dist:,}')

# 旧键下的"内容重复组"（同 bd,bvid,text,offset 多于 1 行）
w = Window.partitionBy('batch_date', 'bvid', 'text', 'time_offset')
dup = (ok.withColumn('k_cnt', F.count('*').over(w))
       .filter(F.col('k_cnt') > 1)
       .withColumn('rk', F.row_number().over(w.orderBy('row_id'))))
n_dup_rows = dup.count()
n_dup_groups = dup.filter(F.col('rk') == 1).count()
print('══════════ 旧键重复组统计 ══════════')
print(f'重复组数 = {n_dup_groups:,}   组内总行数 = {n_dup_rows:,}   被旧键丢弃 = {n_dup_rows - n_dup_groups:,}')

# 抽查：这些"内容重复"里，row_id 是否也真的重复？
g = (dup.groupBy('batch_date', 'bvid', 'text', 'time_offset')
     .agg(F.count('*').alias('n_rows'), F.countDistinct('row_id').alias('n_rid'),
          F.countDistinct('cid').alias('n_cid')))
print('── 重复组里 row_id 的唯一性：n_rows == n_rid 表示内容重复但弹幕 ID 不同（真实的不同弹幕）──')
g.selectExpr('n_rows', 'n_rid', 'n_cid',
             'CASE WHEN n_rows = n_rid THEN 1 ELSE 0 END AS rid_all_unique') \
 .groupBy('n_rows', 'n_rid', 'n_cid', 'rid_all_unique').count() \
 .orderBy(F.desc('count')).show(20, truncate=False)

print('── 样例（前 10 组）──')
dup.select('batch_date', 'bvid', 'text', 'time_offset', 'row_id', 'rk', 'k_cnt') \
   .orderBy('batch_date', 'bvid', 'time_offset', 'rk').show(25, truncate=40)
spark.stop()
