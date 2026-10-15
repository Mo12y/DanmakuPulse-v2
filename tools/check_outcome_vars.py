# -*- coding: utf-8 -*-
'''结果变量体检（纯 Python，避开本 venv 的 numpy 段错误）
三条：① rho(var, age_days) ② rho(var, n_obs_batch) ③ 跨批次稳定性（vs duration 阳性对照）
'''
import csv, glob, math, datetime, statistics
def F(x):
    try:
        v = float(x)
        return v if v == v and abs(v) != float('inf') else None
    except: return None
pub = {}
for fn in glob.glob('/tmp/pubd_local/*.csv'):
    for r in csv.DictReader(open(fn, encoding='utf-8')): pub[r['bvid']] = F(r['pubdate'])
ov = {}
for r in csv.DictReader(open('/home/molly/projects/danmaku-v2/data-out/outcome_vars.csv', encoding='utf-8')): ov[r['bvid']] = r
E = list(csv.DictReader(open('/home/molly/projects/danmaku-v2/data-out/exp_data.csv', encoding='utf-8')))
VARS_E = ['duration','n_danmaku','dm_per_min','wdm_per_min','opening_5pct','ending_5pct','mid_ratio','rhythm_entropy','burst_cv','early_lag_ratio','avg_text_len','n_days_obs']
VARS_O = ['play','half_decay_days','hk_halflife','metric_halflife_d','span_days','coverage','pool_saturation','danmaku_reported','danmaku_got']
rows = []
for r in E:
    p = pub.get(r['bvid'])
    if p is None: continue
    try: bd = datetime.datetime.strptime(r['batch_date'], '%Y%m%d')
    except: continue
    age = (bd - datetime.datetime(1970,1,1) - datetime.timedelta(seconds=p)).total_seconds()/86400.0
    d = {'bvid': r['bvid'], 'age': age}
    for c in VARS_E: d[c] = F(r.get(c))
    for c in VARS_O:
        v = ov.get(r['bvid'], {}).get(c)
        d[c] = F(v)
    rows.append(d)
nb = {}
for r in rows: nb[r['bvid']] = nb.get(r['bvid'], 0) + 1
print('行 %d | unique bvid %d' % (len(rows), len(nb)))
def rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
    while i < len(idx):
        j = i
        while j+1 < len(idx) and v[idx[j+1]] == v[idx[i]]: j += 1
        avg = (i+j)/2.0 + 1
        for k in range(i, j+1): r[idx[k]] = avg
        i = j + 1
    return r
def pear(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return float('nan') if sx == 0 or sy == 0 else sum((x[i]-mx)*(y[i]-my) for i in range(n))/(sx*sy)
def spear(a, b): return pear(rank(a), rank(b))
def rho(col, against):
    pr = [(r[col], r[against]) for r in rows if r.get(col) is not None and r.get(against) is not None]
    if len(pr) < 30: return float('nan'), len(pr)
    return spear([p[0] for p in pr], [p[1] for p in pr]), len(pr)
print()
print('%-20s %14s %14s %12s' % ('变量','rho(vs age)','rho(vs n批次)','跨批CV中位'))
print('-'*66)
for c in ['duration'] + VARS_E[1:] + VARS_O:
    a, na = rho(c, 'age')
    b, _ = rho(c, 'n_obs_batch') if False else (float('nan'), 0)
    pr = [(r[c], nb[r['bvid']]) for r in rows if r.get(c) is not None]
    b = spear([p[0] for p in pr], [p[1] for p in pr]) if len(pr) >= 30 else float('nan')
    g = {}
    for r in rows:
        if r.get(c) is not None: g.setdefault(r['bvid'], []).append(r[c])
    cvs = []
    for k, v in g.items():
        if len(v) >= 2 and statistics.mean(v) != 0:
            cvs.append(statistics.pstdev(v)/abs(statistics.mean(v)))
    cv = statistics.median(cvs) if cvs else float('nan')
    flag = ''
    if abs(a) > 0.7: flag += ' AGE!'
    if abs(b) > 0.7: flag += ' SAMPLE!'
    print('%-20s %+14.3f %+14.3f %12.3f%s' % (c, a, b, cv, flag))