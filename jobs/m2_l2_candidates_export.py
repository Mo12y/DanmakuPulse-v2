# -*- coding: utf-8 -*-
'''m2_l2_candidates_export.py — 导出 theta 截断前的全候选（21 号 §3）
输出 data-out/l2_candidates.csv: bvid,batch_date,source,start_sec,end_sec,score
处理逻辑严格照抄 m2_l2_events.main()：params['detect'] / wcnt / n_danmaku 门槛 / 不归一化。
'''
import sys, os, csv, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, yaml
from pyspark.sql import SparkSession, functions as F
import m2_l2_events as M

ap = argparse.ArgumentParser()
ap.add_argument('--wh-root', default='/warehouse')
ap.add_argument('--config', default='/opt/work/config/params.yaml')
ap.add_argument('--out', default='/opt/work/data-out/l2_candidates.csv')
ap.add_argument('--theta', type=float, default=0.0)
a = ap.parse_args()

params = yaml.safe_load(open(a.config, encoding='utf-8'))
d = params['detect']                      # <-- 关键修正 1
min_dm = int(params.get('l2', {}).get('min_danmaku', 100))
bucket_sec = int(d['bucket_sec'])
w = dict(zip(M.METHODS, d['fusion_w']))
print('fusion_w', w, '| theta(detect)', d['fusion_theta'], '| export theta', a.theta, '| min_danmaku', min_dm)

spark = SparkSession.builder.appName('l2_cand_export').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
d5 = spark.read.parquet(a.wh_root + '/dws_density_5s')
fp = spark.read.parquet(a.wh_root + '/dws_video_rhythm_features').select('bvid','batch_date','n_danmaku')
g = (d5.select('bvid','batch_date', F.struct(F.col('bucket5s').alias('b'), F.col('wcnt').alias('v')).alias('pt'))
       .groupBy('bvid','batch_date').agg(F.sort_array(F.collect_list('pt')).alias('pts'))
       .join(fp, ['bvid','batch_date'], 'left'))
print('groups', g.count(), '| 门槛内', g.filter(F.coalesce(F.col('n_danmaku'), F.lit(0)) >= min_dm).count())

rows = []
n = 0
for r in g.toLocalIterator():
    if r['pts'] is None or r['n_danmaku'] is None or r['n_danmaku'] < min_dm: continue
    mx = max(pt['b'] for pt in r['pts'])
    series = np.zeros(int(mx) + 1, dtype=float)
    for pt in r['pts']: series[int(pt['b'])] = float(pt['v'] or 0.0)
    by = {}
    for m, fn in zip(M.METHODS, (M.detect_d1, M.detect_d2, M.detect_d3, M.detect_d4)):
        segs = M.norm_minmax(fn(series, d))     # 异常不吞，让它报
        by[m] = segs
        for s in segs:
            rows.append((r['bvid'], r['batch_date'], m.upper(),
                         int(s['start'])*bucket_sec, (int(s['end'])+1)*bucket_sec, float(s['score'])))
    for e in M.fuse_video(by, w, a.theta):
        rows.append((r['bvid'], r['batch_date'], 'fused',
                     int(e['start'])*bucket_sec, (int(e['end'])+1)*bucket_sec, float(e['score_fused'])))
    n += 1
    if n % 200 == 0: print('  ...', n, 'videos', len(rows), 'candidates')
with open(a.out, 'w', newline='', encoding='utf-8') as f:
    wr = csv.writer(f); wr.writerow(['bvid','batch_date','source','start_sec','end_sec','score']); wr.writerows(rows)
from collections import Counter
print('TOTAL', len(rows), 'rows |', n, 'video-batches')
print('by source:', dict(Counter(x[2] for x in rows)))
spark.stop()