# -*- coding: utf-8 -*-
'''容器内：分层抽 20 + 导出曲线（纯 Spark，不用 pandas）'''
from pyspark.sql import SparkSession, functions as F
OUT = '/opt/work/data-out/blind_peaks'
spark = SparkSession.builder.appName('bp_src').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
fp = spark.read.parquet('/warehouse/dws_video_rhythm_features').select('bvid','batch_date','n_danmaku','duration').filter(F.col('n_danmaku') >= 100)
lo, hi = fp.approxQuantile('n_danmaku', [1/3, 2/3], 0.0)
print('门槛内', fp.count(), '| 三分位', int(lo), int(hi))
f = fp.withColumn('stratum', F.when(F.col('n_danmaku') < lo, 'low').when(F.col('n_danmaku') < hi, 'mid').otherwise('high'))
parts = []
for name, k in [('low',7),('mid',7),('high',6)]:
    p = f.filter(F.col('stratum') == name).orderBy(F.rand(42)).limit(k)
    parts.append(p); print('  层', name, '抽', p.count())
sel = parts[0]
for p in parts[1:]: sel = sel.unionByName(p)
sel = sel.withColumn('no', F.monotonically_increasing_id() + 1)
sel.write.mode('overwrite').option('header', True).csv(OUT + '/sample20')
keys = sel.select('bvid','batch_date')
cur = spark.read.parquet('/warehouse/dws_density_5s').select('bvid','batch_date','bucket5s','wcnt').join(keys, ['bvid','batch_date'])
cur.write.mode('overwrite').option('header', True).csv(OUT + '/curves20')
print('sel', sel.count(), '| curve rows', cur.count())
spark.stop()