# -*- coding: utf-8 -*-
'''tools/cross_batch_stability.py — 跨批次检测稳定性（23 号）'''
import csv, collections
rows = [r for r in csv.DictReader(open('data-out/l2_candidates.csv', encoding='utf-8')) if r['source'] == 'fused']
by = collections.defaultdict(dict)
for r in rows:
    try: v = float(r['score'])
    except ValueError: continue
    if v != v: continue
    by[r['bvid']].setdefault(r['batch_date'], []).append((int(r['start_sec']), int(r['end_sec']), v))
def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1]); i = max(0.0, e - s)
    u = (a[1]-a[0]) + (b[1]-b[0]) - i
    return i/u if u > 0 else 0.0
multi = {b: d for b, d in by.items() if len(d) >= 2}
print('bvid 出现在 >=2 批次:', len(multi))
print('%7s %8s %14s %14s' % ('theta', 'pairs', 'event-match', 'top1-match'))
for TH in [0.0, 0.3, 0.5, 0.7]:
    hit = tot = t1 = t1ok = nvid = 0
    for b, d in multi.items():
        bs = sorted(d)
        for i in range(len(bs)-1):
            A = [x for x in d[bs[i]] if x[2] >= TH]; B = [x for x in d[bs[i+1]] if x[2] >= TH]
            if not A or not B: continue
            nvid += 1
            for x in A:
                tot += 1
                if any(iou((x[0], x[1]), (y[0], y[1])) >= 0.3 for y in B): hit += 1
            ta = max(A, key=lambda x: x[2]); tb = max(B, key=lambda x: x[2])
            t1 += 1
            if iou((ta[0], ta[1]), (tb[0], tb[1])) >= 0.3: t1ok += 1
    print('%7.1f %8d %6d/%-7d %6d/%-7d' % (TH, nvid, hit, tot, t1ok, t1))