# -*- coding: utf-8 -*-
'''标注者 A2：泊松下界规则（修正版）
修正：
  1. 合并间距 50s -> 10s（原值照抄了检测器的 d3_gap_min，对短视频把峰全并掉）
  2. 局部窗自适应（clip(n//8, 4, 24) 桶），原固定 ±12 桶对短视频几乎等于全局
'''
import math, csv
import numpy as np
from collections import defaultdict
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('annA2').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
rows = spark.read.parquet('/warehouse/dws_density_5s').select('bvid','batch_date','bucket5s','cnt').collect()
G = defaultdict(dict)
for r in rows: G[(r['bvid'], r['batch_date'])][int(r['bucket5s'])] = float(r['cnt'] or 0)
BS, K, MIN_GAP_B = 5, 2.0, 2      # 5s/桶, 2*sqrt(lam), 合并间距 2 桶 = 10s
out = []
for (bv, bd), d in sorted(G.items()):
    if len(d) < 5: continue
    mx = max(d); c = np.zeros(mx + 1)
    for b, v in d.items(): c[b] = v
    n = len(c)
    win = int(min(max(n // 8, 4), 24))
    peaks = []
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        nb = [c[j] for j in range(lo, hi) if j != i]
        if not nb: continue
        base = float(np.percentile(nb, 25))
        lam = float(np.mean(nb))
        if c[i] <= base: continue
        if c[i] >= max(c[lo:hi]) - 1e-9 and (c[i] - base) > K * math.sqrt(max(lam, 1e-9)):
            peaks.append((i, float(c[i]), base, lam))
    peaks.sort(key=lambda x: -x[1])
    kept = []
    for p in peaks:
        if all(abs(p[0] - q[0]) >= MIN_GAP_B for q in kept): kept.append(p)
    kept.sort()
    for (i, v, base, lam) in kept:
        out.append({'bvid': bv, 'batch_date': bd, 'start_sec': i*BS, 'end_sec': (i+1)*BS,
                    'peak_cnt': int(v), 'local_base': round(base,2), 'lambda': round(lam,2)})
print('A2 峰数:', len(out), '| 覆盖 video-batch', len({(x['bvid'],x['batch_date']) for x in out}))
k = {(x['bvid'],x['batch_date']) for x in out}
print('每 video-batch 峰数 中位 %.1f' % np.median([sum(1 for y in out if (y['bvid'],y['batch_date'])==kk) for kk in k]))
with open('/opt/work/data-out/annotator_A2_poisson.csv','w',newline='',encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print('-> data-out/annotator_A2_poisson.csv')
spark.stop()