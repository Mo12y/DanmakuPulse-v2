#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp61_e3_d4_operating_point.py — D4"无增益"是否只是阈值口径问题

背景：exp55 用 thr=0.5 比较模型 F1(0.276) 与"爆发持续规则"(0.478)，据此判"无稳定增益"。
      但 0.5 是概率校准后的默认阈值，未必是模型的合理工作点。
做法：同一 cohort（1012–1026，bvid 不在 D-train）、同一冻结模型，画 PR 曲线：
      ① 模型在**规则召回 0.625** 处的精度（同召回比精度，公平口径）
      ② 模型的最大 F1（在 cohort 上选阈值 = 乐观上界，单独标注）
输出：experiments/exp61_audit/e3.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

REPO = '/home/molly/projects/danmaku-v2'
sys.path.insert(0, f'{REPO}/experiments/exp_ml_scan_20261027')
import run_scan as rs  # noqa: E402
import joblib          # noqa: E402

d = rs.build_daily()
P_dt = rs.build_points()
dt_bvids = set(P_dt.bvid.unique())
cw = []
for (bv, bt), g in d.groupby(['bvid', 'batch']):
    if bv in dt_bvids:
        continue
    dur = float(g['duration'].iloc[0]); Kf = int(dur // 5)
    if Kf < 18 or float(g['part_count'].iloc[0] if pd.notna(g['part_count'].iloc[0]) else 0) >= 2:
        continue
    off = g['time_offset'].values.astype(float)
    edge = np.isclose(off, dur); off = off[~edge]
    b = (off // 5).astype(int); b = b[b < Kf]
    c = np.bincount(b, minlength=Kf)
    for t in range(12, Kf - 5, 6):
        lam = float(c[t - 12:t].mean()); thr = max(3.0, lam + 2.0 * np.sqrt(max(lam, 0.0)))
        sinc, cens = 12, 1
        for dd in range(1, 13):
            kk = t - dd
            if kk < 12:
                break
            lk = float(c[kk - 12:kk].mean()); tk = max(3.0, lk + 2.0 * np.sqrt(max(lk, 0.0)))
            if c[kk] >= tk:
                sinc, cens = dd, 0
                break
        cw.append((bv, bt, t, lam, int(c[t:t + 6].max() >= thr), float(c[t:t + 6].sum()), sinc, cens,
                   float(np.log1p(lam)), float(np.log1p(c[t - 6:t].mean()) - np.log1p(c[t - 12:t - 6].mean())),
                   t / Kf, float(np.log1p(dur))))
Pc = pd.DataFrame(cw, columns=P_dt.columns)
FEATS = ['log1p_lam', 'slope', 'since', 'censored', 'prog', 'log1p_duration']
art = joblib.load(f'{REPO}/experiments/exp38_integrate/models/d4_lr6_platt.joblib')
s = art['pipe'].decision_function(Pc[FEATS])
p = art['platt'].predict_proba(s.reshape(-1, 1))[:, 1]
y = Pc.y.values.astype(int)
rule = (Pc.censored.values == 0)
tp_rule = int((rule & (y == 1)).sum())
rule_p = tp_rule / max(int(rule.sum()), 1)
rule_r = tp_rule / max(int(y.sum()), 1)
rule_f1 = 2 * rule_p * rule_r / (rule_p + rule_r)

prec, rec, thr = precision_recall_curve(y, p)
order = np.argsort(rec)
rec_s, prec_s = rec[order], prec[order]
p_at_rule_r = float(np.interp(rule_r, rec_s, prec_s))
# 模型最大 F1（乐观：阈值在 cohort 上选）
f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
best_i = int(np.nanargmax(f1s[:-1]))
out = dict(n_points=int(len(Pc)), pos_rate=round(float(y.mean()), 4),
           auc=round(float(roc_auc_score(y, p)), 4),
           rule=dict(precision=round(rule_p, 4), recall=round(rule_r, 4), f1=round(rule_f1, 4)),
           model_at_rule_recall=dict(recall_target=round(rule_r, 4), precision=round(p_at_rule_r, 4),
                                     implied_f1=round(2 * p_at_rule_r * rule_r / (p_at_rule_r + rule_r), 4)),
           model_best_f1_optimistic=dict(f1=round(float(f1s[best_i]), 4),
                                         precision=round(float(prec[best_i]), 4),
                                         recall=round(float(rec[best_i]), 4),
                                         thr=round(float(thr[best_i]), 4),
                                         note='阈值在 cohort 上选 ⇒ 乐观上界，非部署口径'),
           pr_auc=round(float(average_precision_score(y, p)), 4))
out['verdict'] = ('同召回下模型精度高于规则 ⇒ 之前"无增益"是阈值口径问题'
                  if out['model_at_rule_recall']['precision'] > rule_p + 0.02
                  else '同召回下模型不优于规则 ⇒ "无增益"结论成立')
json.dump(out, open(f'{REPO}/experiments/exp61_audit/e3.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
