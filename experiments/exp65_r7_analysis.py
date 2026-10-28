#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp65_r7_analysis.py — R7 相似性盲评一致性分析

输入：data-out/exp65_r7/{r7_pairs_A.csv, r7_pairs_B.csv, r7_stratified_pairs.csv}
判定标准（先于结果，预注册）：
  通过 = ① very_similar 层均分显著高于 dissimilar 层（Mann-Whitney p<0.05，两层 n=20）
        ② 三层均分单调（very ≥ middle ≥ dissimilar）
        ③ 评价者间评分相关 ≥0.3（两人独立性的最低信度）
输出：experiments/exp65_r7_analysis/results.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

REPO = '/home/molly/projects/danmaku-v2'
D = f'{REPO}/data-out/exp65_r7'
OUT = f'{REPO}/experiments/exp65_r7_analysis'
os.makedirs(OUT, exist_ok=True)

lab = pd.read_csv(f'{D}/r7_stratified_pairs.csv', encoding='utf-8-sig')[
    ['pair_id', 'stratum']]
A = pd.read_csv(f'{D}/r7_pairs_A.csv', encoding='utf-8-sig')[['pair_id', 'similar_1to5']].rename(
    columns={'similar_1to5': 'A'})
B = pd.read_csv(f'{D}/r7_pairs_B.csv', encoding='utf-8-sig')[['pair_id', 'similar_1to5']].rename(
    columns={'similar_1to5': 'B'})
m = lab.merge(A, on='pair_id').merge(B, on='pair_id')
for c in ('A', 'B'):
    m[c] = pd.to_numeric(m[c], errors='coerce')
m = m.dropna(subset=['A', 'B'])
invalid = m[(m.A < 1) | (m.A > 5) | (m.B < 1) | (m.B > 5)]
m = m[(m.A >= 1) & (m.A <= 5) & (m.B >= 1) & (m.B <= 5)]
m['mean'] = (m.A + m.B) / 2

by = m.groupby('stratum')['mean'].agg(['count', 'mean', 'median']).round(4)
r_ab, p_ab = spearmanr(m.A, m.B)

vs = m.loc[m.stratum == 'very_similar', 'mean'].values
ds = m.loc[m.stratum == 'dissimilar', 'mean'].values
mw = mannwhitneyu(vs, ds, alternative='greater')
p_value = float(mw.pvalue)
mono = (by.loc['very_similar', 'mean'] >= by.loc['middle', 'mean'] >= by.loc['dissimilar', 'mean'])

res = dict(
    n_pairs=int(len(m)), n_invalid_dropped=int(len(invalid)),
    stratum_means={s: round(float(m[m.stratum == s]['mean'].mean()), 4)
                   for s in ('very_similar', 'middle', 'dissimilar')},
    per_annotator={w: {s: round(float(m[m.stratum == s][w].mean()), 4)
                       for s in ('very_similar', 'middle', 'dissimilar')} for w in ('A', 'B')},
    monotonic=bool(mono), mannwhitney_p=round(p_value, 5),
    annotator_spearman=round(float(r_ab), 4), annotator_spearman_p=round(float(p_ab), 6),
    gate=('PASS' if (p_value < 0.05 and mono and r_ab >= 0.3) else 'FAIL'),
    pre_registered='①MW p<0.05（very>dissim）②三层单调③评分者相关>=0.3')
json.dump(res, open(f'{OUT}/results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print(f"R7-GATE-{res['gate']}")
