# -*- coding: utf-8 -*-
'''分桶合理性诊断：泊松噪声 vs 真实结构（Fano 因子）'''
import numpy as np
from collections import defaultdict
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('bucket_diag').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
rows = spark.read.parquet('/warehouse/dws_density_5s').select('bvid','batch_date','bucket5s','cnt').collect()
print('总行', len(rows))
g = defaultdict(dict)
for r in rows: g[(r['bvid'], r['batch_date'])][int(r['bucket5s'])] = float(r['cnt'] or 0)
print('video-batch', len(g))
lam, fano, nb = [], [], []
for k, d in g.items():
    v = np.array(list(d.values()))
    if len(v) < 5 or v.mean() <= 0: continue
    lam.append(v.mean()); fano.append(v.var()/v.mean()); nb.append(len(v))
lam = np.array(lam); fano = np.array(fano); nb = np.array(nb)
print()
print('=== 5s 桶泊松诊断 ===')
print('%-14s %7s %7s %7s %7s %7s' % ('指标','P10','P25','P50','P75','P90'))
for name, arr in [('桶内均值 lam', lam), ('Fano=var/mean', fano), ('桶数', nb)]:
    q = np.percentile(arr, [10,25,50,75,90])
    print('%-14s %7.2f %7.2f %7.2f %7.2f %7.2f' % (name, q[0],q[1],q[2],q[3],q[4]))
print()
print('=== 按桶内均值分档 ===')
for lo, hi, tag in [(0,3,'lam<3 极噪'), (3,10,'3-10 噪'), (10,30,'10-30 可用'), (30,1e12,'lam>=30 良好')]:
    m = (lam >= lo) & (lam < hi)
    if m.sum() == 0: continue
    print('  %-14s %5.1f%% (%4d 个)  Fano中位 %6.2f  噪声 1/sqrt(lam) 中位 %3.0f%%'
          % (tag, 100*m.mean(), m.sum(), float(np.median(fano[m])), 100/float(np.sqrt(np.median(lam[m])))))
print()
print('=== Fano 分布（Fano≈1 为纯泊松噪声；越大越有真实结构）===')
for t in [1.0, 1.2, 1.5, 2.0, 3.0, 5.0]:
    print('  Fano > %4.1f : %5.1f%% (%d 个)' % (t, 100*(fano>t).mean(), int((fano>t).sum())))