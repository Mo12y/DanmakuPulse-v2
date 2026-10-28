#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp55 · 三路线并行验证（A=未来强度 D1 / B=短期变化 D4 / C=未来曲线形态）

原则：复用已有数据管线与冻结模型，只补证据缺口，不重训生产模型、不改冻结协议。
  A：复现 exp38 D1 时间切（train first_seen≤20261020 / test ≥20261021，bvid 隔离），
     补 MAE/RMSE（log1p 与原始尺度）+ 持久性基线(4×c0) + 历史均值基线。
  B：复现 exp38 D4 独立 cohort（1012-1026，bvid 不在 D-train），用冻结 d4_lr6_platt.joblib
     （不重训）补 PR-AUC / F1 / precision / recall（thr=0.5）+ 多数类基线 + 爆发持续规则基线。
  C：复用 exp48.1 的 346 序列（视频级 80/20 切分），滚动起点多步预测（未来 10 个分钟桶）：
     持久性 / 近端均值 / 历史均值 vs OLS(lag 特征)。同切分同测试点。
复现校验：A 的 R² 必须逐位对上 0.4532/0.5607/0.5391；B 的 AUC 必须对上 0.6001/0.4538。
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, \
    average_precision_score, precision_score, recall_score, f1_score

REPO = '/home/molly/projects/danmaku-v2'
sys.path.insert(0, f'{REPO}/experiments/exp_ml_scan_20261027')
sys.path.insert(0, f'{REPO}/experiments')
import run_scan as rs                                   # noqa: E402
import exp48_1_scoring_aligned as e481                  # noqa: E402

SEED = 20261027
OUT = f'{REPO}/experiments/exp55_route_abc_verify'
os.makedirs(OUT, exist_ok=True)
R = {'meta': dict(seed=SEED, engine='exp55-route-abc-verify')}

print('[1/4] build_daily()（读 1012-1026 日批 CSV）…', flush=True)
d = rs.build_daily()

# ================================================================ Part A：路线 A（D1）
print('[2/4] 路线 A：D1 复现 + 补 MAE/RMSE/持久性基线', flush=True)
rows = []
for (bv, bt), g in d.groupby(['bvid', 'batch']):
    pub = float(g['pubdate'].iloc[0]); dur = float(g['duration'].iloc[0])
    rel = (g['send_time'].values - pub) / 3600.0
    if rel.max() < 31.0:
        continue
    n01 = ((rel >= 0) & (rel < 1)).sum(); n13 = ((rel >= 1) & (rel < 3)).sum()
    n36 = ((rel >= 3) & (rel < 6)).sum(); c0 = n01 + n13 + n36
    if c0 < 1:
        continue
    rows.append((bv, g['first_seen'].iloc[0], n01, n13, n36, c0, ((rel > 6) & (rel <= 30)).sum(), dur))
u = pd.DataFrame(rows, columns=['bvid', 'fs', 'n01', 'n13', 'n36', 'c0', 'y', 'dur'])
tr, te = u[u.fs <= 20261020], u[u.fs >= 20261021]
assert len(set(tr.bvid) & set(te.bvid)) == 0
ytr, yte = np.log1p(tr.y.values), np.log1p(te.y.values)


def mats(df, feats):
    cols = {'c0': np.log1p(df.c0), 'n01': np.log1p(df.n01), 'n13': np.log1p(df.n13),
            'n36': np.log1p(df.n36), 'slope': np.log1p(df.n36) - np.log1p(df.n13), 'dur': np.log1p(df.dur)}
    return np.column_stack([cols[f] for f in feats])


F1F, F6F = ['c0'], ['c0', 'n36', 'n13', 'n01', 'slope', 'dur']
b1 = LinearRegression().fit(mats(tr, F1F), ytr)
m6 = LinearRegression().fit(mats(tr, F6F), ytr)
hgb = HistGradientBoostingRegressor(max_depth=3, random_state=SEED).fit(mats(tr, F6F), ytr)


def reg_report(name, pred_log):
    """log1p 尺度 R²/MAE/RMSE + 原始尺度 MAE/RMSE（expm1，clip≥0）"""
    r2 = r2_score(yte, pred_log)
    mae_l = mean_absolute_error(yte, pred_log)
    rmse_l = float(np.sqrt(mean_squared_error(yte, pred_log)))
    po, to = np.clip(np.expm1(pred_log), 0, None), np.expm1(yte)
    return dict(model=name, R2_log1p=round(float(r2), 4), MAE_log1p=round(float(mae_l), 4),
                RMSE_log1p=round(rmse_l, 4), MAE_orig=round(float(mean_absolute_error(to, po)), 1),
                RMSE_orig=round(float(np.sqrt(mean_squared_error(to, po))), 1), n_test=len(te))


pa = dict(n_train=len(tr), n_test=len(te), bvid_overlap=0,
          B1_log1p_c0=reg_report('B1 OLS(log1p c0)', b1.predict(mats(te, F1F))),
          OLS6=reg_report('OLS-6(生产特征)', m6.predict(mats(te, F6F))),
          HGB=reg_report('HGB depth3', hgb.predict(mats(te, F6F))),
          persistence_4x_c0=reg_report('持久性 ŷ=4×c0', np.log1p(4.0 * te.c0.values)),
          hist_mean=reg_report('历史均值(训练集均值)', np.full(len(te), ytr.mean())),
          verification=dict(B1_expect=0.4532, OLS6_expect=0.5607, HGB_expect=0.5391,
                           B1_ok=abs(r2_score(yte, b1.predict(mats(te, F1F))) - 0.4532) < 5e-4,
                           OLS6_ok=abs(r2_score(yte, m6.predict(mats(te, F6F))) - 0.5607) < 5e-4,
                           HGB_ok=abs(r2_score(yte, hgb.predict(mats(te, F6F))) - 0.5391) < 5e-4))
R['route_A'] = pa
print(json.dumps(pa['verification'], ensure_ascii=False), '| persistence R2=',
      pa['persistence_4x_c0']['R2_log1p'], flush=True)

# ================================================================ Part B：路线 B（D4 cohort）
print('[3/4] 路线 B：D4 cohort 复现 + 补 PR-AUC/F1/类别比例基线', flush=True)
import joblib                                          # noqa: E402
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
s_c = art['pipe'].decision_function(Pc[FEATS])
p_c = art['platt'].predict_proba(s_c.reshape(-1, 1))[:, 1]
yc = Pc.y.values.astype(int)
s_neglam = -Pc.log1p_lam.values

# 简单规则基线：爆发状态持续（上一窗已爆发 ⇒ 预测下一窗爆发）
rule_persist = ((Pc.log1p_lam.values) > 0)  # 占位；下面用 lam 阈值规则重算
lam_arr = Pc.lam.values
thr_rule = np.array([max(3.0, l + 2.0 * np.sqrt(max(l, 0.0))) for l in lam_arr])
# 持续规则：上一窗（log1p_lam 对应的上一窗计数）已超阈值 —— 用 since/censored 语义重建：
# sinc<12 ⇒ 前窗内出现过突起。Pc.sinc 是距上次突起的窗口数，censored=1 表示 12 窗内无突起
rule_persist = (Pc.censored.values == 0)

pb = dict(n_points=len(Pc), n_videos=int(Pc.bvid.nunique()), pos_rate=round(float(yc.mean()), 4),
          AUC_LR6=round(float(roc_auc_score(yc, p_c)), 4),
          PR_AUC_LR6=round(float(average_precision_score(yc, p_c)), 4),
          AUC_NegLam=round(float(roc_auc_score(yc, s_neglam)), 4),
          AUC_majority=0.5,
          at_thr_0p5=dict(precision=round(float(precision_score(yc, p_c >= 0.5, zero_division=0)), 4),
                          recall=round(float(recall_score(yc, p_c >= 0.5)), 4),
                          f1=round(float(f1_score(yc, p_c >= 0.5, zero_division=0)), 4),
                          n_pred_pos=int((p_c >= 0.5).sum())),
          rule_persistence_burst=dict(precision=round(float(precision_score(yc, rule_persist, zero_division=0)), 4),
                                      recall=round(float(recall_score(yc, rule_persist)), 4),
                                      f1=round(float(f1_score(yc, rule_persist, zero_division=0)), 4),
                                      n_pred_pos=int(rule_persist.sum())),
          majority=dict(always_negative=True, accuracy=round(float((yc == 0).mean()), 4), F1_pos=0.0),
          verification=dict(AUC_LR6_expect=0.6001, AUC_NegLam_expect=0.4538,
                            AUC_LR6_ok=abs(float(roc_auc_score(yc, p_c)) - 0.6001) < 5e-4,
                            AUC_NegLam_ok=abs(float(roc_auc_score(yc, s_neglam)) - 0.4538) < 5e-4))
R['route_B'] = pb
print(json.dumps(pb['verification'], ensure_ascii=False), '| PR-AUC=', pb['PR_AUC_LR6'], flush=True)

# ================================================================ Part C：路线 C（多步曲线）
print('[4/4] 路线 C：多步曲线预测（持久性/近端均值/历史均值 vs OLS-lag）', flush=True)
seqs = e481.load()
TR, TE = e481.split(seqs)
H, STEP, MINHIST = 10, 5, 15


def rolling_samples(seq, train=True):
    Xs, Ys, metas = [], [], []
    c = seq['c']; n = len(c)
    for t in range(MINHIST, n - H + 1, STEP):
        feats = [np.log1p(c[t - 1]), np.log1p(np.mean(c[t - 5:t])), np.log1p(np.max(c[t - 5:t])),
                 np.log1p(np.std(c[t - 5:t]) + 1e-9), np.log1p(seq['got']), t / n]
        Xs.append(feats)
        Ys.append(c[t:t + H])
        metas.append((seq['bvid'], t))
    return Xs, Ys, metas


trX, trY = [], []
for s in TR:
    a, b, _ = rolling_samples(s)
    trX.extend(a); trY.extend(b)
ols = LinearRegression().fit(np.array(trX), np.vstack(trY))

per_h = {h: dict(persist=[], recmean=[], histmean=[], ols=[]) for h in range(H)}
curve_mae = dict(persist=[], recmean=[], histmean=[], ols=[])
for s in TE:
    c = s['c']; n = len(c)
    if n < MINHIST + H:
        continue
    hist_mean = float(np.mean([np.mean(x['c']) for x in TR]))
    for t in range(MINHIST, n - H + 1, STEP):
        truth = c[t:t + H]
        preds = {
            'persist': np.full(H, c[t - 1]),
            'recmean': np.full(H, np.mean(c[t - 5:t])),
            'histmean': np.full(H, hist_mean),
            'ols': np.clip(ols.predict(np.array([[
                np.log1p(c[t - 1]), np.log1p(np.mean(c[t - 5:t])), np.log1p(np.max(c[t - 5:t])),
                np.log1p(np.std(c[t - 5:t]) + 1e-9), np.log1p(s['got']), t / n]]))[0], 0, None),
        }
        for name, pv in preds.items():
            for k in range(H):
                per_h[k][name].append(abs(truth[k] - pv[k]))
            curve_mae[name].append(float(np.mean(np.abs(truth - pv))))
overall = {name: dict(MAE=round(float(np.mean([m for h in per_h for m in per_h[h][name]])), 2),
                      curve_MAE=round(float(np.mean(curve_mae[name])), 2),
                      skill_vs_persist=round(1 - float(np.mean(curve_mae[name])) /
                                             float(np.mean(curve_mae['persist'])), 4))
           for name in ('persist', 'recmean', 'histmean', 'ols')}
per_horizon = {h: {name: round(float(np.mean(v)), 2) for name, v in per_h[h].items()} for h in per_h}
pc = dict(n_test_videos=len(TE), horizon=H, step=STEP, min_hist=MINHIST,
          split='视频级 80/20（exp48.1 seed=20261027 随机切分，无视频重叠）',
          overall=overall, per_horizon_MAE=per_horizon,
          note='计数为重尾分布，MAE 原始尺度受高活跃视频主导；同时看 skill_vs_persist（曲线 MAE 相对持久性基线的下降比例）')
R['route_C'] = pc
print(json.dumps(pc, ensure_ascii=False, indent=1), flush=True)

with open(f'{OUT}/results.json', 'w', encoding='utf-8') as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print('EXP55-DONE →', f'{OUT}/results.json')
