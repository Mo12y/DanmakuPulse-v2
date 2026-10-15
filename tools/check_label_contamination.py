# -*- coding: utf-8 -*-
'''纯 Python：early_lag_ratio vs n_days_obs 的秩相关 + 控制年龄的偏相关'''
import csv, glob, math, datetime
def f(x):
    try: return float(x)
    except: return None
pub = {}
for fn in glob.glob('/tmp/pubd_local/*.csv'):
    for r in csv.DictReader(open(fn, encoding='utf-8')):
        pub[r['bvid']] = f(r['pubdate'])
rows = []
for r in csv.DictReader(open('/home/molly/projects/danmaku-v2/data-out/exp_data.csv', encoding='utf-8')):
    nd = f(r.get('n_days_obs')); el = f(r.get('early_lag_ratio')); du = f(r.get('duration')); p = pub.get(r['bvid'])
    if None in (nd, el, du, p): continue
    try: bd = datetime.datetime.strptime(r['batch_date'], '%Y%m%d')
    except: continue
    age = (bd - datetime.datetime.utcfromtimestamp(p)).total_seconds()/86400.0
    if age < 0: continue
    rows.append({'nd': nd, 'el': el, 'age': age, 'du': du, 'ndm': f(r.get('n_danmaku')) or 0.0})
print('n =', len(rows))
ages = sorted(x['age'] for x in rows)
print('age_days  P10=%.1f  P50=%.1f  P90=%.1f' % (ages[len(ages)//10], ages[len(ages)//2], ages[9*len(ages)//10]))
def rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0]*len(v); i = 0
    while i < len(idx):
        j = i
        while j+1 < len(idx) and v[idx[j+1]] == v[idx[i]]: j += 1
        avg = (i+j)/2.0 + 1
        for k in range(i, j+1): r[idx[k]] = avg
        i = j+1
    return r
def pear(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    if sx == 0 or sy == 0: return float('nan')
    return sum((x[i]-mx)*(y[i]-my) for i in range(n))/(sx*sy)
def spear(a, b): return pear(rank(a), rank(b))
def resid(y, Z):
    # 最小二乘残差（Z 为列列表，自动加截距）
    import itertools
    n = len(y); X = [[1.0] + [c[i] for c in Z] for i in range(n)]
    m = len(X[0])
    # 正规方程 + 高斯消元
    A = [[sum(X[i][a]*X[i][b] for i in range(n)) for b in range(m)] + [sum(X[i][a]*y[i] for i in range(n))] for a in range(m)]
    for c in range(m):
        p = max(range(c, m), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        if abs(A[c][c]) < 1e-12: continue
        for r in range(m):
            if r != c and A[r][c] != 0:
                fct = A[r][c]/A[c][c]
                for k in range(c, m+1): A[r][k] -= fct*A[c][k]
    beta = [A[i][m]/A[i][i] if abs(A[i][i]) > 1e-12 else 0.0 for i in range(m)]
    return [y[i] - sum(beta[a]*X[i][a] for a in range(m)) for i in range(n)]
nd = [x['nd'] for x in rows]; el = [x['el'] for x in rows]
age = [x['age'] for x in rows]; du = [x['du'] for x in rows]; ndm = [x['ndm'] for x in rows]
print()
print('  rho(early_lag, n_days_obs) = %+.3f' % spear(el, nd))
print('  rho(age_days , n_days_obs) = %+.3f' % spear(age, nd))
print('  rho(early_lag, age_days )  = %+.3f' % spear(el, age))
print()
print('  ** 控 age            : rho = %+.3f' % pear(resid(el, [age]), resid(nd, [age])))
print('  ** 控 age + duration : rho = %+.3f' % pear(resid(el, [age, du]), resid(nd, [age, du])))
print('  ** 控 age + dur + n_danmaku : rho = %+.3f' % pear(resid(el, [age, du, ndm]), resid(nd, [age, du, ndm])))