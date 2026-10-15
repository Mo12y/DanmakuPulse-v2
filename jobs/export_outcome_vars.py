# -*- coding: utf-8 -*-
import csv
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('chk').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
WH='/warehouse'
dim = spark.read.parquet(WH+'/dim_video').select('bvid','pubdate','tname','n_batches')
lc = spark.read.parquet(WH+'/dws_lifecycle').select('bvid','n_days_obs','half_decay_days','hk_halflife','metric_halflife_d','span_days','series_admissible')
mt = spark.read.parquet(WH+'/dws_metric_track').select('bvid','cadence','play','like','coin','favorite','share','reply','danmaku_reported','danmaku_got','coverage','pool_saturation')
print('metric_track 行', mt.count(), '| bvid', mt.select('bvid').distinct().count())
j = dim.join(lc, 'bvid', 'left').join(mt, 'bvid', 'left')
rows = j.collect()
print('join 行', len(rows))
cols = ['bvid','cadence','pubdate','tname','n_batches','n_days_obs','half_decay_days','hk_halflife','metric_halflife_d','span_days','series_admissible','play','like','coin','favorite','share','reply','danmaku_reported','danmaku_got','coverage','pool_saturation']
with open('/opt/work/data-out/outcome_vars.csv','w',newline='',encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(cols)
    for r in rows:
        d = r.asDict()
        w.writerow([d.get(c) for c in cols])
print('-> data-out/outcome_vars.csv')
spark.stop()