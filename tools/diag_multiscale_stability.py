# -*- coding: utf-8 -*-
'''重分桶稳定性 v2：用事件区间内的峰值桶判局部极大（修正 v1 用 start_bucket 的错误）'''
import numpy as np, math
from collections import defaultdict
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('rebin2').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
d5 = spark.read.parquet('/warehouse/dws_density_5s').select('bvid','batch_date','bucket5s','cnt').collect()
ev = spark.read.parquet('/warehouse/dws_rhythm_events').select('bvid','batch_date','start_bucket','end_bucket','score_fused').collect()
G = defaultdict(dict)
for r in d5: G[(r['bvid'], r['batch_date'])][int(r['bucket5s'])] = float(r['cnt'] or 0)
E = defaultdict(list)
for r in ev: E[(r['bvid'], r['batch_date'])].append((int(r['start_bucket']), int(r['end_bucket']), float(r['score_fused'])))
print('事件总数', sum(len(v) for v in E.values()))
print()
print('%-7s %7s %11s %11s %11s' % ('聚合','事件数','峰桶=局部极大','超2sqrt(lam)','超3sqrt(lam)'))
for w in [1, 3, 6, 12]:
    tot = loc = s2 = s3 = skip = 0
    for k, evs in E.items():
        d = G.get(k)
        if not d: continue
        mx = max(d); c = np.zeros(mx + 1)
        for b, v in d.items(): c[b] = v
        n = len(c) // w
        if n < 3: continue
        coarse = c[:n*w].reshape(n, w).sum(axis=1)
        lam = float(c.mean())
        for (sb, eb, sc) in evs:
            seg = c[int(sb):int(eb)+1]
            if len(seg) == 0: skip += 1; continue
            pb = int(sb) + int(np.argmax(seg))      # 区间内峰值桶
            cb = min(pb // w, n - 1)
            lo, hi = max(0, cb - 1), min(n, cb + 2)
            base = float(np.median(coarse[max(0,cb-5):min(n,cb+6)]))
            tot += 1
            if coarse[cb] >= coarse[lo:hi].max() - 1e-9: loc += 1
            if coarse[cb] - base > 2*math.sqrt(max(lam*w, 1e-9)): s2 += 1
            if coarse[cb] - base > 3*math.sqrt(max(lam*w, 1e-9)): s3 += 1
    p = lambda x: '%10.1f%%' % (100*x/max(tot,1))
    print('%-7s %7d %11s %11s %11s' % ('%ds'%(5*w), tot, p(loc), p(s2), p(s3)))
print()
print('（跳过：区间为空或粗桶不足）')