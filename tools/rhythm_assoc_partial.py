# -*- coding: utf-8 -*-
'''任务1：控年龄偏相关矩阵 + BH-FDR（27 号 3C 正式结果）
任务2：T3 跨批稳定性（签名级 within vs between）
只用 numpy，避开 scipy/pandas-datetime（本 venv 会段错误）
'''
import csv, glob, math, datetime
import numpy as np

def F(x):
    try:
        v = float(x); return v if v == v and abs(v) != float('inf') else None
    except: return None

pub = {}
for fn in glob.glob('/tmp/pubd_local/*.csv'):
    for r in csv.DictReader(open(fn, encoding='utf-8')): pub[r['bvid']] = F(r['pubdate'])
ov = {}
for r in csv.DictReader(open('/home/molly/projects/danmaku-v2/data-out/outcome_vars.csv', encoding='utf-8')): ov[r['bvid']] = r
E = list(csv.DictReader(open('/home/molly/projects/danmaku-v2/data-out/exp_data.csv', encoding='utf-8')))
SHAPE = ['opening_5pct','ending_5pct','mid_ratio','rhythm_entropy','burst_cv']
INTEN = ['dm_per_min','wdm_per_min','avg_text_len']
OUTS  = ['play','half_decay_days','hk_halflife','span_days','coverage','danmaku_got','danmaku_reported']

recs = []
for r in E:
    p = pub.get(r['bvid'])
    if p is None: continue
    try: bd = datetime.datetime.strptime(r['batch_date'], '%Y%m%d')
    except: continue
    age = (bd - (datetime.datetime(1970,1,1) + datetime.timedelta(seconds=p))).total_seconds()/86400.0
    d = {'bvid': r['bvid'], 'age': age}
    for c in SHAPE + INTEN: d[c] = F(r.get(c))
    o = ov.get(r['bvid'], {})
    for c in OUTS: d[c] = F(o.get(c))
    recs.append(d)
print('样本行 %d | unique bvid %d' % (len(recs), len({r['bvid'] for r in recs})))

def rk(a):
    a = np.asarray(a, float); o = a.argsort(); r = np.empty(len(a)); r[o] = np.arange(1, len(a)+1)
    # 并列取平均秩
    vals, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(vals)); s[inv] = r
        avg = np.zeros(len(vals)); np.add.at(avg, inv, r); avg /= cnt
        r = avg[inv]
    return r
def spear(x, y):
    rx, ry = rk(x) - rk(x).mean(), rk(y) - rk(y).mean()
    den = math.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/den) if den > 0 else float('nan')
def pcorr(x, y, ctrl):
    A = np.column_stack([np.ones(len(ctrl))] + [c for c in ctrl.T])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return spear(rx, ry)
def tstat(r, n, k=1):
    df = n - k - 2
    if df <= 0 or abs(r) >= 1: return float('nan')
    return r * math.sqrt(df / (1 - r*r))
from math import erf, sqrt
def pval(t, df):
    # 正态近似（n≈800 足够）
    return 2 * (1 - 0.5*(1 + erf(abs(t)/sqrt(2)))) if t == t else float('nan')

print()
print('=== 任务1：控 age 的偏相关（Spearman）+ BH-FDR ===')
allr = []
for o in OUTS:
    for f in SHAPE + INTEN:
        pr = [(r[f], r[o], r['age']) for r in recs if r.get(f) is not None and r.get(o) is not None]
        if len(pr) < 100: continue
        x = np.array([p[0] for p in pr]); y = np.array([p[1] for p in pr]); z = np.array([[p[2]] for p in pr])
        r0 = spear(x, y); rp = pcorr(x, y, z)
        n = len(pr); t = tstat(rp, n); p = pval(t, n)
        allr.append((o, f, r0, rp, n, p))
allr.sort(key=lambda x: x[5])
m = len(allr)
print('检验数 m = %d' % m)
print('%-18s %-16s %8s %10s %6s %12s %8s' % ('结果变量','特征','rho原始','rho控龄','n','p','BH阈值'))
sig = 0
for i, (o, f, r0, rp, n, p) in enumerate(allr):
    bh = 0.05 * (i + 1) / m
    mark = ' *' if p <= bh else ''
    if p <= bh: sig += 1
    if i < 18 or mark:
        print('%-18s %-16s %+8.3f %+10.3f %6d %12.2e %8.4f%s' % (o, f, r0, rp, n, p, bh, mark))
print()
print('BH-FDR 0.05 下显著: %d / %d' % (sig, m))

print()
print('=== 任务2：T3 跨批稳定性（签名级，within vs between）===')
FEAT = SHAPE + INTEN
rows = [r for r in recs if all(r.get(c) is not None for c in FEAT)]
g = {}
for r in rows: g.setdefault(r['bvid'], []).append(np.array([r[c] for c in FEAT], float))
multi = {k: v for k, v in g.items() if len(v) >= 2}
singles = {k: v[0] for k, v in g.items() if len(v) == 1}
print('有 >=2 批的 bvid: %d | 单批: %d' % (len(multi), len(singles)))
mu = np.mean(np.array([r[c] for r in rows for c in FEAT]).reshape(-1, len(FEAT)), axis=0)
sd = np.std(np.array([r[c] for r in rows for c in FEAT]).reshape(-1, len(FEAT)), axis=0)
sd[sd == 0] = 1.0
def z(v): return (v - mu) / sd
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na*nb)) if na > 0 and nb > 0 else float('nan')
withins = []
for k, v in multi.items():
    zv = [z(x) for x in v]
    for i in range(len(zv)):
        for j in range(i+1, len(zv)): withins.append(cos(zv[i], zv[j]))
keys = list(singles.keys())
rng = np.random.default_rng(42)
betweens = []
for _ in range(5000):
    a, b = rng.choice(len(keys), 2, replace=False)
    betweens.append(cos(z(singles[keys[a]]), z(singles[keys[b]])))
w = np.array([x for x in withins if x == x]); b = np.array([x for x in betweens if x == x])
print('within  n=%d  中位 %.3f  P10 %.3f  P90 %.3f' % (len(w), np.median(w), np.percentile(w,10), np.percentile(w,90)))
print('between n=%d  中位 %.3f  P10 %.3f  P90 %.3f' % (len(b), np.median(b), np.percentile(b,10), np.percentile(b,90)))
print('** within 中位(%.3f) > between P90(%.3f) ? %s' % (np.median(w), np.percentile(b,90), np.median(w) > np.percentile(b,90)))