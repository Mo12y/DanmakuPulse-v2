# -*- coding: utf-8 -*-
'''导出对照实验所需数据：101点曲线 + 标签 + up_mid + 结果变量'''
import csv
from collections import defaultdict
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.appName('exp').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
WH = '/warehouse'
dim = spark.read.parquet(WH+'/dim_video').select('bvid','tname','up_mid')
tmap = {r['bvid']: (r['tname'], r['up_mid']) for r in dim.collect()}
# 结果变量：在榜天数
lc = spark.read.parquet(WH+'/dws_lifecycle').select('bvid','n_days_obs','half_decay_days')
lmap = {r['bvid']: (r['n_days_obs'], r['half_decay_days']) for r in lc.collect()}
fp = spark.read.parquet(WH+'/dws_video_rhythm_features').select('bvid','batch_date','duration','n_danmaku','dm_per_min','wdm_per_min','opening_5pct','ending_5pct','mid_ratio','rhythm_entropy','burst_cv','early_lag_ratio','peak_cnt','avg_text_len')
frows = fp.collect()
d1 = spark.read.parquet(WH+'/dws_density_1pct').select('bvid','batch_date','bucket1pct','wcnt').collect()
g = defaultdict(dict)
for r in d1: g[(r['bvid'], r['batch_date'])][int(r['bucket1pct'])] = float(r['wcnt'] or 0.0)
OUT = '/opt/work/data-out/exp_data.csv'
n = 0
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['bvid','batch_date','tname','up_mid','n_days_obs','half_decay','duration','n_danmaku','dm_per_min','wdm_per_min','opening_5pct','ending_5pct','mid_ratio','rhythm_entropy','burst_cv','early_lag_ratio','peak_cnt','avg_text_len'] + ['c%d' % i for i in range(101)])
    for r in frows:
        k = (r['bvid'], r['batch_date'])
        if k not in g: continue
        t, u = tmap.get(r['bvid'], (None, None))
        nd, hd = lmap.get(r['bvid'], (None, None))
        cur = [g[k].get(i, 0.0) for i in range(101)]
        mx = max(cur) or 1.0
        w.writerow([r['bvid'], r['batch_date'], t, u, nd, hd, r['duration'], r['n_danmaku'], r['dm_per_min'], r['wdm_per_min'],
                    r['opening_5pct'], r['ending_5pct'], r['mid_ratio'], r['rhythm_entropy'], r['burst_cv'], r['early_lag_ratio'], r['peak_cnt'], r['avg_text_len']] + [round(v/mx, 6) for v in cur])
        n += 1
print('导出', n, '行 ->', OUT)
spark.stop()