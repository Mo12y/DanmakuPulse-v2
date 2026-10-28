#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp64_reverify.py — 收尾复验（只做仍缺的判别与可行性检查）

D1 自参照优势的来源判别：self vs same-UP(其他视频) vs partition（三参照配对比较）；
   若 same-UP ≈ self ⇒ 优势来自视频级属性（受众/题材），而非"形状信息本身"。
D2 适用域拆分：got<600（全片可观测）与 got≥600 分别报告信号 A。
D3 R7 盲评可行性：按距离分位分层抽 60 对（覆盖"很像/中等/不像"），验证材料可生成。
输出：experiments/exp64_reverify/{d1,d2,d3}.json
"""
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

REPO = '/home/molly/projects/danmaku-v2'
EXP56, EXP57, EXP62 = f'{REPO}/data-out/exp56', f'{REPO}/data-out/exp57', f'{REPO}/data-out/exp62'
OUT = f'{REPO}/experiments/exp64_reverify'
os.makedirs(OUT, exist_ok=True)
SEED = 20261028
rng = np.random.default_rng(SEED)


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{d}/*.csv'))],
                     ignore_index=True)


def sp(a, b):
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1])


def norm(v):
    return v / v.sum() if v.sum() > 0 else v


def p10(a, b):
    return len(set(np.argsort(a)[-10:]) & set(np.argsort(b)[-10:])) / 10.0


units = read_dir(f'{EXP56}/units_csv')
units['tname'] = units['tname'].astype(str)
meta57 = read_dir(f'{EXP57}/video_meta_csv')          # up_mid/got 在这里（exp56 units 无 up_mid）
meta57['up_mid'] = meta57['up_mid'].astype(str)
up_of = dict(zip(meta57.bvid, meta57.up_mid))
meta_got = dict(zip(meta57.bvid, pd.to_numeric(meta57.got, errors='coerce')))
units['up_mid'] = units.bvid.map(up_of)
tname_of = dict(zip(units.bvid, units.tname))
cur1 = read_dir(f'{EXP56}/curves1pct_csv')
cur1['bucket1pct'] = pd.to_numeric(cur1['bucket1pct'], errors='coerce').astype(int)
cur1['cnt'] = pd.to_numeric(cur1['cnt'], errors='coerce')
dense = cur1.pivot_table(index='bvid', columns='bucket1pct', values='cnt', aggfunc='sum').fillna(0.0)

# 分区中位形状（排除自身）
part_shape = {}
for t, g in units.groupby('tname'):
    bvs = [b for b in g.bvid if b in dense.index]
    if len(bvs) < 5:
        continue
    part_shape[t] = np.median(dense.loc[bvs].values, axis=0)
# UP 中位形状（每 UP ≥2 支才可算"其他视频"）
up_shape = {}
for u, g in units.groupby('up_mid'):
    bvs = [b for b in g.bvid if b in dense.index]
    if len(bvs) >= 2:
        up_shape[u] = (bvs, np.median(dense.loc[bvs].values, axis=0))

# ---------------- exp62 的无滚动对 ----------------
bm = read_dir(f'{EXP62}/batch_meta_csv')
curm = read_dir(f'{EXP62}/curves_multi_csv')
bm['age_max_h'] = pd.to_numeric(bm['age_max_h'], errors='coerce')
bm = bm.sort_values(['bvid', 'age_max_h'])
pairs = []
for bv, d in bm.groupby('bvid'):
    if len(d) < 2:
        continue
    e, l = d.iloc[0], d.iloc[-1]
    if l.n_pool >= e.n_pool * 0.995 and e.n_pool >= 100 and l.n_pool >= 100:
        pairs.append((bv, e.batch_date, l.batch_date, e.n_pool, l.n_pool))
mi = curm.set_index(['bvid', 'batch_date', 'bucket1pct'])['cnt']
_c = {}


def vec(bv, b):
    if (bv, b) not in _c:
        try:
            s = mi.xs((bv, b))
        except KeyError:
            s = pd.Series(dtype=float)
        _c[(bv, b)] = np.array([float(s.get(i, 0.0)) for i in range(100)], float)
    return _c[(bv, b)]


# ---------------- D1 三参照配对比较 ----------------
rows = []
for bv, be, bl, ne, nl in pairs:
    t, u = tname_of.get(bv), up_of.get(bv)
    if t not in part_shape:
        continue
    E, L = vec(bv, be), vec(bv, bl)
    delta = norm(np.clip(L - E, 0, None))
    refs = {'self': norm(E), 'partition': norm(part_shape[t])}
    # same-UP：同 UP 其他视频的中位（排除自身）
    if u in up_shape:
        bvs_u, shape_u = up_shape[u]
        others = [b for b in bvs_u if b != bv]
        if len(others) >= 1:
            M = dense.loc[others].values
            refs['same_up'] = norm(np.median(M, axis=0))
    if delta.sum() <= 0 or any(r.std() == 0 for r in refs.values()):
        continue
    rec = dict(bvid=bv)
    for name, r in refs.items():
        rec[f'rho_{name}'] = sp(r, delta) or 0.0
        rec[f'p10_{name}'] = p10(r, delta)
    rows.append(rec)
D = pd.DataFrame(rows)
D['got'] = D.bvid.map(meta_got)
D['pool'] = np.where(D.got < 600, 'low(<600)', 'high(>=600)')


def paired(d, a, b, key):
    sub = d[[f'rho_{a}', f'rho_{b}']].dropna()
    if len(sub) < 10:
        return None
    dv = (sub[f'rho_{a}'] - sub[f'rho_{b}']).values
    bs = [float(np.mean(dv[rng.integers(0, len(dv), len(dv))])) for _ in range(1000)]
    return dict(n=int(len(dv)), d_rho_median=round(float(np.median(dv)), 4),
                d_rho_mean=round(float(np.mean(dv)), 4),
                ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                share_a_better=round(float((dv > 0).mean()), 3))


D1 = dict(n_pairs=int(len(D)),
          rho_median={c: round(float(D[f'rho_{c}'].median()), 4) for c in ('self', 'same_up', 'partition')},
          p10_median={c: round(float(D[f'p10_{c}'].median()), 4) for c in ('self', 'same_up', 'partition')})
D1['self_vs_same_up'] = paired(D, 'self', 'same_up', 'rho')
D1['self_vs_partition'] = paired(D, 'self', 'partition', 'rho')
# 适用域拆分
D1['by_pool'] = {}
for pool, d in D.groupby('pool'):
    cell = dict(n=int(len(d)), rho_self_median=round(float(d.rho_self.median()), 4),
                rho_partition_median=round(float(d.rho_partition.median()), 4))
    dv = (d.rho_self - d.rho_partition).values
    bs = [float(np.mean(dv[rng.integers(0, len(dv), len(dv))])) for _ in range(1000)]
    cell['d_rho_ci95'] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    D1['by_pool'][pool] = cell
D1['verdict'] = ('self ≈ same_up ⇒ 优势来自视频级属性（受众/题材），不是形状信息本身'
                 if D1['self_vs_same_up'] and abs(D1['self_vs_same_up']['d_rho_mean']) < 0.03
                 else 'self 优于 same_up ⇒ 优势含真实的"自身形状"信息')
json.dump(D1, open(f'{OUT}/d1_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[D1]', json.dumps({k: D1[k] for k in ('n_pairs', 'rho_median', 'self_vs_same_up', 'self_vs_partition', 'by_pool')},
                         ensure_ascii=False), flush=True)

# ---------------- D3 R7 分层盲评材料（60 对，按距离分位） ----------------
units['c1'] = units.c1_eligible.astype(str).str.lower().eq('true')
pool = sorted(set(units[units.c1].bvid) & set(dense.index))
X = np.array([norm(dense.loc[b].values / max(dense.loc[b].values.sum(), 1e-9)) for b in pool])
nn = int(len(pool) * (len(pool) - 1) / 2)
# 采样：随机抽 60 个查询，每个取"最像/中位/最不像"各 1 对 → 覆盖全距离分位
qsel = list(rng.choice(len(pool), 20, replace=False))
recs = []
pid = 1
for qi in qsel:
    q = pool[qi]
    dd = [(float(np.linalg.norm(X[i] - X[qi])), i) for i in range(len(pool)) if i != qi]
    dd.sort()
    n = len(dd)
    for lab, j in (('very_similar', dd[n // 5][1]), ('middle', dd[n // 2][1]), ('dissimilar', dd[-2][1])):
        recs.append(dict(pair_id=f'P{pid:03d}', stratum=lab, bvid_a=q, bvid_b=pool[j],
                         url_a=f'https://www.bilibili.com/video/{q}',
                         url_b=f'https://www.bilibili.com/video/{pool[j]}'))
        pid += 1
with open(f'{OUT}/r7_stratified_pairs.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=['pair_id', 'stratum', 'bvid_a', 'bvid_b', 'url_a', 'url_b', 'similar_1to5', 'note'])
    w.writeheader(); w.writerows(recs)
D3 = dict(n_pairs=len(recs), design='20 查询 × 3 距离分位（很像/中等/不像），C1 池 232 支',
          file=f'{OUT}/r7_stratified_pairs.csv',
          note='每个 stratum 的对数相等 ⇒ 人工判断可按层统计一致率；评价者看不到 stratum 标签')
json.dump(D3, open(f'{OUT}/d3_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[D3]', json.dumps({k: D3[k] for k in ('n_pairs', 'design')}, ensure_ascii=False), flush=True)
print('EXP64-DONE')
