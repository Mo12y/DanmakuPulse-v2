#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp65c · R7 聚类修正检验（P0-3）：60 对按查询聚类 → 每查询 very_similar 与 dissimilar 的配对比较
预注册：符号检验（双侧）+ Wilcoxon；n=20 个查询。
输出：experiments/exp65_r7_analysis/clustered_results.json
"""
import json, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

REPO = '/home/molly/projects/danmaku-v2'
D = f'{REPO}/data-out/exp65_r7'
OUT = f'{REPO}/experiments/exp65_r7_analysis'

lab = pd.read_csv(f'{D}/r7_stratified_pairs.csv', encoding='utf-8-sig')[['pair_id', 'stratum', 'bvid_a']]
A = pd.read_csv(f'{D}/r7_pairs_A.csv', encoding='utf-8-sig')[['pair_id', 'similar_1to5']].rename(columns={'similar_1to5': 'A'})
B = pd.read_csv(f'{D}/r7_pairs_B.csv', encoding='utf-8-sig')[['pair_id', 'similar_1to5']].rename(columns={'similar_1to5': 'B'})
m = lab.merge(A, on='pair_id').merge(B, on='pair_id')
for c in ('A', 'B'):
    m[c] = pd.to_numeric(m[c], errors='coerce')
m['mean'] = (m.A + m.B) / 2

diffs = []
for (q), g in m.groupby('bvid_a'):
    v = float(g.loc[g.stratum == 'very_similar', 'mean'].mean())
    d = float(g.loc[g.stratum == 'dissimilar', 'mean'].mean())
    diffs.append(dict(query=q, very=v, dissim=d, diff=v - d))
C = pd.DataFrame(diffs)
k_pos = int((C['diff'] > 0).sum())
sign_p = float(binomtest(k_pos, len(C), 0.5, alternative='two-sided').pvalue)
try:
    wil_p = float(wilcoxon(C.very, C.dissim, alternative='two-sided').pvalue)
except Exception:
    wil_p = None
per = {'A': [], 'B': []}
for w in ('A', 'B'):
    pw = pd.read_csv(f'{D}/r7_pairs_{w}.csv', encoding='utf-8-sig')[['pair_id', 'similar_1to5']]
    mm = lab.merge(pw, on='pair_id')
    mm['s'] = pd.to_numeric(mm['similar_1to5'], errors='coerce')
    for q, g in mm.groupby('bvid_a'):
        v = float(g.loc[g.stratum == 'very_similar', 's'].mean())
        d = float(g.loc[g.stratum == 'dissimilar', 's'].mean())
        per[w].append(v > d)
res = dict(
    n_queries=int(len(C)),
    queries_very_gt_dissim=k_pos,
    sign_test_p=round(sign_p, 5),
    wilcoxon_p=(round(wil_p, 5) if wil_p is not None else None),
    mean_diff=round(float(C['diff'].mean()), 4),
    per_annotator_pos={w: f'{sum(per[w])}/{len(per[w])}' for w in ('A', 'B')},
    note='按查询聚类后 n=20（非 60）；60 对含同一查询 3 次的非独立样本，原 MW p 不成立',
    conclusion=('聚类修正后仍显著（p<0.05）⇒ 检索排序与人工判断一致的方向成立'
                if sign_p < 0.05 else '聚类修正后不显著 ⇒ R7 证据不足'))
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f'{OUT}/clustered_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print('R7-CLUSTERED:', res['conclusion'])
