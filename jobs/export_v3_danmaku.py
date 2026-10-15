# -*- coding: utf-8 -*-
'''v3 证据钻取供数：把弹幕正文导出给前端（33 号交互精修 · 证据钻取）

为什么走 PG 而不是塞进 v3data/*.json：
  快照覆盖 888 视频-批次 / 111.5 万条弹幕，JSONL 约 44MB —— 进 git 不合适，
  全量读进 256MB 的 fastapi 容器内存里更不合适。
  落到 PostgreSQL（fastapi 本来就连着）+ (bvid, t) 索引，查某时刻是毫秒级。

输出：/opt/work/data-out/v3_danmaku/danmaku.csv（单文件，供 COPY 进 PG）
只读 HDFS，不改管线、不重跑实验。
'''
import subprocess
from pyspark.sql import SparkSession, functions as F

OUT = 'file:///opt/work/data-out/v3_danmaku'   # 必须带 file://：Spark 默认 FS 是 HDFS，不带就写进 HDFS 了
spark = (SparkSession.builder.appName('export_v3_danmaku')
         .config('spark.sql.shuffle.partitions', '8').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

d = (spark.read.parquet('/warehouse/dwd_danmaku_clean')
     .filter(~F.col('is_spam'))
     .select('bvid', 'batch_date', 'time_offset', 'text'))
d = d.filter(F.col('time_offset').isNotNull() & F.col('text').isNotNull())
# 清掉会破坏行结构的字符（换行/制表/回车），并截断超长文本
d = d.withColumn('txt', F.regexp_replace(F.col('text'), r'[\r\n\t]+', ' '))
d = d.withColumn('txt', F.substring(F.col('txt'), 1, 120))
d = d.filter(F.length(F.trim(F.col('txt'))) > 0)
d = d.withColumn('t', F.round(F.col('time_offset'), 1).cast('float'))
d = d.select('bvid', 'batch_date', 't', 'txt')

n = d.count()
print('导出弹幕行数', n)
(d.coalesce(1).write.mode('overwrite')
 .option('header', False).option('quote', '"').option('escape', '"')
 .option('nullValue', '').csv(OUT))
spark.stop()
print('写出', OUT)

