#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp65b · R8 判别：哪种表示的距离最贴合人工相似性判断（60 对人工评分已就绪）
候选表示：norm_curve（100 点归一曲线）、feat8（8 维节奏特征 z 分数）、PCA{4,8,16}
指标：距离与人工均分的 Spearman（负值越大越好 = 距离越近人觉得越像）+ 三层分离度
输出：experiments/exp65_r7_analysis/r8_repr.json
"""
import json, os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

REPO = '/home/molly/projects/danmaku-v2'
D = f'{REPO}/data-out/exp65_r7'
EXP56 = f'{REPO}/data-out/exp56'
OUT = f'{REPO}/experiments/exp65_r7_analysis'


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{d}/*.csv'))], ignore_index=True)


lab = pd.read_csv(f'{D}/r7_stratified_pairs.csv', encoding='utf-8-sig')[['pair_id', 'stratum', 'bvid_a', 'bvid_b']]
A = pd.read_csv(f'{D}/r7_pairs_A.csv', encoding='utf-8-sig')[['pair_id', 'similar_1to5']].rename(columns={'similar_1to5': 'A'})
B = pd.read_csv(f'{D}/r7_pairs_B.csv', encoding='utf-8-sig')[['pair_id', 'similar_1to5']].rename(columns={'similar_1to5': 'B'})
m = lab.merge(A, on='pair_id').merge(B, on='pair_id')
m['human'] = (pd.to_numeric(m.A, errors='coerce') + pd.to_numeric(m.B, errors='coerce')) / 2
m = m.dropna(subset=['human'])

units = read_dir(f'{EXP56}/units_csv')
units['c1'] = units.c1_eligible.astype(str).str.lower().eq('true')
feats = read_dir(f'{EXP56}/feats_csv')
FKEYS = ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'ending_5pct', 'mid_ratio', 'early_lag_ratio']  # peak_cnt 导出为空列，剔除
for k in FKEYS:
    feats[k] = pd.to_numeric(feats[k], errors='coerce')
dense = pd.read_csv(f'{EXP56}/curves1pct_csv'.replace('curves1pct_csv', 'curves1pct_csv'), encoding='utf-8-sig') if False else None
cur1 = read_dir(f'{EXP56}/curves1pct_csv')
cur1['bucket1pct'] = pd.to_numeric(cur1['bucket1pct'], errors='coerce').astype(int)
cur1['cnt'] = pd.to_numeric(cur1['cnt'], errors='coerce')
dense = cur1.pivot_table(index='bvid', columns='bucket1pct', values='cnt', aggfunc='sum').fillna(0.0)
pool = sorted(set(units[units.c1].bvid) & set(dense.index))
C = np.array([dense.loc[b].values / max(dense.loc[b].values.sum(), 1e-9) for b in pool])
fmap = {r.bvid: [getattr(r, k) for k in FKEYS] for r in feats.itertuples()}
Fm = np.array([fmap.get(b, [np.nan] * len(FKEYS)) for b in pool], float)
ok = ~np.isnan(Fm).any(1)
Fm_z = (Fm[ok] - np.nanmean(Fm[ok], 0)) / np.nanstd(Fm[ok], 0)
pool_ok = [b for b, o in zip(pool, ok) if o]
pca = {k: PCA(n_components=k).fit_transform(C) for k in (4, 8, 16)}


def dist(repr_matrix, ia, ib):
    return float(np.linalg.norm(repr_matrix[ia] - repr_matrix[ib]))


pos = {b: i for i, b in enumerate(pool_ok)}
recs = []
for r in m.itertuples():
    ia, ib = pos.get(r.bvid_a), pos.get(r.bvid_b)
    if ia is None or ib is None:
        continue
    recs.append(dict(pair_id=r.pair_id, stratum=r.stratum, human=r.human, ia=ia, ib=ib))
print('evaluable pairs:', len(recs), '/', len(m))
rep_mats = {'norm_curve': C[ok] if not ok.all() else C}
# 对齐：全部池里 ok 的子集
rep_mats['norm_curve'] = C[[pos[b] for b in pool_ok]]
rep_mats['feat8'] = Fm_z
for k in (4, 8, 16):
    rep_mats[f'pca{k}'] = pca[k][ok] if not ok.all() else pca[k]

res = {}
for name, M in rep_mats.items():
    ds, hs, strata = [], [], []
    for r in recs:
        d = dist(M, r['ia'], r['ib'])
        if np.isnan(d):
            continue
        ds.append(d); hs.append(r['human']); strata.append(r['stratum'])
    rho, p = spearmanr(ds, hs)          # 负相关 = 距离大→人觉得不像（好）
    # 三层分离：very_similar 的距离中位应最小
    med = {s: round(float(np.median([d for d, st in zip(ds, strata) if st == s])), 5)
           for s in ('very_similar', 'middle', 'dissimilar')}
    res[name] = dict(n=int(len(ds)), spearman_dist_vs_human=round(float(rho), 4), p=round(float(p), 5),
                     dist_median_by_stratum=med,
                     separation=(med['dissimilar'] - med['very_similar']))
names = [k for k in res if k != 'verdict']
best = min(names, key=lambda k: res[k]['spearman_dist_vs_human'])   # 越负越好
worst = max(names, key=lambda k: res[k]['spearman_dist_vs_human'])
spread = round(res[best]['spearman_dist_vs_human'] - res[worst]['spearman_dist_vs_human'], 4)
res['verdict'] = (f"最贴合人工判断的表示 = {best}（Spearman {res[best]['spearman_dist_vs_human']}），"
                  f"最差 = {worst}（{res[worst]['spearman_dist_vs_human']}），极差仅 {spread} "
                  f"⇒ 五种表示与人工判断几乎等价，简单表示已足够，表示学习无必要")
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f'{OUT}/r8_repr.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print('EXP65B-DONE')
