# -*- coding: utf-8 -*-
'''theta 敏感性扫描（21 号 3/5）：从 theta=0 全候选表算任意阈值的保留量'''
import csv, collections
rows = [r for r in csv.DictReader(open('data-out/l2_candidates.csv', encoding='utf-8'))]
by = collections.defaultdict(list)
for r in rows:
    try: v = float(r['score'])
    except ValueError: continue
    if v != v: continue
    by[r['source']].append((v, r['bvid'], r['batch_date']))
n_group = len({(r['bvid'], r['batch_date']) for r in rows})
print('video-batch 总数:', n_group)
print()
TH = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print('%-6s' % 'theta' + ''.join('%9s' % ('%.1f' % t) for t in TH))
for s in ['fused', 'D1', 'D2', 'D3', 'D4']:
    v = by[s]
    line = '%-6s' % s
    for t in TH:
        k = [x for x in v if x[0] >= t]
        line += '%9d' % len(k)
    print(line)
print()
print('=== fused 在各 theta 下覆盖的 video-batch 数与密度 ===')
print('%-6s %10s %10s %10s' % ('theta', 'events', 'vb_cover', 'ev/vb'))
fv = by['fused']
for t in TH:
    k = [x for x in fv if x[0] >= t]
    cov = len({(x[1], x[2]) for x in k})
    print('%-6.1f %10d %10d %10.2f' % (t, len(k), cov, len(k)/max(cov,1)))
print()
print('=== 关键比值 ===')
n0 = len([x for x in fv if x[0] >= 0.0]); n5 = len([x for x in fv if x[0] >= 0.5])
print('  theta=0 -> %.0f 个；theta=0.5 -> %d 个；保留率 %.2f%%' % (n0, n5, 100*n5/n0))
import statistics as st
qs = sorted(x[0] for x in fv)
for p in [50, 75, 90, 95, 96, 99]:
    print('  fused 分数 P%-3d = %.4f' % (p, qs[int(len(qs)*p/100)-1]))