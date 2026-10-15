# -*- coding: utf-8 -*-
'''任务2b：T3 仅形状；任务3：偏离检测（分区内稳健 z）；任务4：分区 KW'''
import csv, glob, math, datetime
import numpy as np
def F(x):
    try:
        v = float(x); return v if v == v and abs(v) != float('inf') else None
    except: return None
E = list(csv.DictReader(open('/home/molly/projects/danmaku-v2/data-out/exp_data.csv', encoding='utf-8')))
SHAPE = ['opening_5pct','ending_5pct','mid_ratio','rhythm_entropy','burst_cv','early_lag_ratio']
INTEN = ['dm_per_min','wdm_per_min','avg_text_len']

def t3(feat, label):
    rows = [r for r in E if all(F(r.get(c)) is not None for c in feat) and r.get('bvid')]
    g = {}
    for r in rows: g.setdefault(r['bvid'], []).append(np.array([F(r[c]) for c in feat]))
    multi = {k: v for k, v in g.items() if len(v) >= 2}
    singles = {k: v[0] for k, v in g.items() if len(v) == 1}
    M = np.array([x for v in g.values() for x in v])
    mu, sd = M.mean(0), M.std(0); sd[sd == 0] = 1
    z = lambda v: (v - mu) / sd
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a) > 0 and np.linalg.norm(b) > 0 else float('nan')
    w = [cos(z(v[i]), z(v[j])) for v in multi.values() for i in range(len(v)) for j in range(i+1, len(v))]
    keys = list(singles); rng = np.random.default_rng(42)
    b = [cos(z(singles[keys[a]]), z(singles[keys[c]])) for a, c in (rng.choice(len(keys), 2, replace=False) for _ in range(5000))]
    w = np.array([x for x in w if x == x]); b = np.array([x for x in b if x == x])
    print('  %-14s within 中位 %.3f (P10 %.3f) | between 中位 %.3f (P90 %.3f) | within中位>betweenP90 ? %s'
          % (label, np.median(w), np.percentile(w,10), np.median(b), np.percentile(b,90), np.median(w) > np.percentile(b,90)))
print('=== 任务2b：T3 跨批稳定性（分特征集）===')
print('  bvid >=2 批: %d' % len({r['bvid'] for r in E if sum(1 for x in E if x['bvid'] == r['bvid']) >= 2}))
t3(SHAPE, '仅形状(6维)')
t3(INTEN, '仅强度(3维)')
t3(SHAPE + INTEN, '形状+强度')

print()
print('=== 任务3：偏离检测（分区内稳健 z，MAD）===')
dim_t = {}
for r in csv.DictReader(open('/home/molly/projects/danmaku-v2/data-out/outcome_vars.csv', encoding='utf-8')): dim_t[r['bvid']] = r.get('tname')
rows = []
for r in E:
    t = dim_t.get(r['bvid'])
    v = [F(r.get(c)) for c in SHAPE]
    if t and all(x is not None for x in v): rows.append((r['bvid'], r['batch_date'], t, np.array(v)))
byt = {}
for bv, bd, t, v in rows: byt.setdefault(t, []).append((bv, bd, v))
scored = []
for t, lst in byt.items():
    if len(lst) < 20: continue
    M = np.array([x[2] for x in lst])
    med = np.median(M, 0); mad = np.median(np.abs(M - med), 0); mad[mad == 0] = 1e-9
    z = np.abs((M - med) / (1.4826 * mad))
    for i, (bv, bd, _) in enumerate(lst): scored.append((float(z[i].max()), t, bv, bd, int(z[i].argmax())))
scored.sort(reverse=True)
print('  参与分区 %d | 视频 %d' % (len([t for t in byt if len(byt[t]) >= 20]), len(scored)))
print('  偏离榜 TOP10（max robust-z 跨 6 个形状维）：')
for s, t, bv, bd, k in scored[:10]: print('    z=%5.2f  %-10s %s %s  维=%s' % (s, t, bv, bd, SHAPE[k]))
print()
print('  分区偏离率（z>3.5 的比例）：')
from collections import Counter
tot = Counter(); ab = Counter()
for s, t, bv, bd, k in scored:
    tot[t] += 1
    if s > 3.5: ab[t] += 1
for t, n in tot.most_common(8): print('    %-12s %3d/%3d = %5.1f%%' % (t, ab[t], n, 100*ab[t]/n))

print()
print('=== 任务4：分区间节奏差异（Kruskal-Wallis）===')
def kw(groups):
    allv = np.concatenate(groups); n = len(allv)
    r = np.empty(n); o = allv.argsort(); r[o] = np.arange(1, n+1)
    vals, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        avg = np.zeros(len(vals)); np.add.at(avg, inv, r); avg /= cnt; r = avg[inv]
    idx = 0; H = 0.0
    for g in groups:
        gg = r[idx:idx+len(g)]; idx += len(g)
        H += len(g) * (gg.mean() - (n+1)/2.0)**2
    H = 12.0 / (n*(n+1)) * H
    k = len(groups)
    return H, k - 1, H / (n - 1)   # H, df, eta^2 近似
for j, c in enumerate(SHAPE):
    gs = []
    for t, lst in byt.items():
        if len(lst) < 20: continue
        v = [x[2][j] for x in lst]
        if len(v) >= 20: gs.append(np.array(v))
    if len(gs) < 3: continue
    H, df, eta = kw(gs)
    print('  %-16s H=%8.1f  df=%2d  eta2=%.4f  %s' % (c, H, df, eta, '***' if H > 30 else ('*' if H > df*2 else '')))