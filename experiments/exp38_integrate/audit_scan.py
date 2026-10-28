# -*- coding: utf-8 -*-
"""exp38_integrate/audit_scan.py · 任务38 审计+补充验证+生产工件（一次性）
复用 exp_ml_scan_20261027.run_scan 的构建函数；不修改其产物。
产出 audit_results.json + models/{d1_ols.json,d2_hgb.joblib,d4_lr6_platt.joblib,meta.json}
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
SCAN = f"{V2}/experiments/exp_ml_scan_20261027"
OUT = f"{V2}/experiments/exp38_integrate"
MDIR = f"{OUT}/models"
os.makedirs(MDIR, exist_ok=True)
sys.path.insert(0, SCAN)
import run_scan as rs  # noqa: E402  (build_daily/build_points/r2; main 有 __guard__ 不会执行)

from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score  # noqa: E402
import joblib  # noqa: E402

SEED = 20261027
R = {}


def log(*a):
    print(*a, flush=True)


# ---------- 1) 复算 D1（含消融） ----------
d = rs.build_daily()
rows = []
for (bv, bt), g in d.groupby(["bvid", "batch"]):
    pub = float(g["pubdate"].iloc[0]); dur = float(g["duration"].iloc[0])
    rel = (g["send_time"].values - pub) / 3600.0
    if rel.max() < 31.0:
        continue
    n01 = ((rel >= 0) & (rel < 1)).sum(); n13 = ((rel >= 1) & (rel < 3)).sum()
    n36 = ((rel >= 3) & (rel < 6)).sum(); c0 = n01 + n13 + n36
    if c0 < 1:
        continue
    rows.append((bv, g["first_seen"].iloc[0], n01, n13, n36, c0, ((rel > 6) & (rel <= 30)).sum(), dur))
u = pd.DataFrame(rows, columns=["bvid", "fs", "n01", "n13", "n36", "c0", "y", "dur"])
tr, te = u[u.fs <= 20261020], u[u.fs >= 20261021]
assert len(set(tr.bvid) & set(te.bvid)) == 0
ytr, yte = np.log1p(tr.y.values), np.log1p(te.y.values)


def mats(df, feats):
    cols = {"c0": np.log1p(df.c0), "n01": np.log1p(df.n01), "n13": np.log1p(df.n13),
            "n36": np.log1p(df.n36), "slope": np.log1p(df.n36) - np.log1p(df.n13), "dur": np.log1p(df.dur)}
    return np.column_stack([cols[f] for f in feats])


vcodes, vidx = np.unique(te.bvid.values, return_inverse=True)
rng = np.random.default_rng(SEED)
boot_idx = [np.concatenate([np.where(vidx == w)[0] for w in rng.choice(len(vcodes), len(vcodes), True)])
            for _ in range(2000)]

STEPS = [("c0",), ("c0", "n36"), ("c0", "n36", "n13"), ("c0", "n36", "n13", "n01"),
         ("c0", "n36", "n13", "n01", "slope"), ("c0", "n36", "n13", "n01", "slope", "dur")]
abl, preds = [], {}
for i, feats in enumerate(STEPS):
    m = LinearRegression().fit(mats(tr, feats), ytr)
    p = m.predict(mats(te, feats))
    preds[i] = p
    cis = []
    for idx in boot_idx:
        a = rs.r2(yte[idx], p[idx])
        if np.isfinite(a):
            cis.append(a - rs.r2(yte[idx], preds[0][idx]))
    abl.append(dict(model=f"OLS_{len(feats)}f", feats=list(feats), R2_test=round(rs.r2(yte, p), 4),
                    dR2_vs_baseline=round(rs.r2(yte, p) - rs.r2(yte, preds[0]), 4),
                    ci=[round(float(np.percentile(cis, 2.5)), 4), round(float(np.percentile(cis, 97.5)), 4)]))
gb = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=20, random_state=SEED).fit(mats(tr, STEPS[-1]), ytr)
pg = gb.predict(mats(te, STEPS[-1]))
cis = []
for idx in boot_idx:
    a = rs.r2(yte[idx], pg[idx])
    if np.isfinite(a):
        cis.append(a - rs.r2(yte[idx], preds[0][idx]))
abl.append(dict(model="HGB", feats=STEPS[-1], R2_test=round(rs.r2(yte, pg), 4),
                dR2_vs_baseline=round(rs.r2(yte, pg) - rs.r2(yte, preds[0]), 4),
                ci=[round(float(np.percentile(cis, 2.5)), 4), round(float(np.percentile(cis, 97.5)), 4)]))
R["D1_recheck"] = dict(n_train=len(tr), n_test=len(te), bvid_overlap=0, ablation=abl,
                       replicate_exp37=dict(OLS1=abl[0]["R2_test"], OLS6=abl[5]["R2_test"], HGB=abl[6]["R2_test"]))
log("[D1-recheck]", json.dumps(R["D1_recheck"]["replicate_exp37"]))
log("[D1-ablation]", json.dumps(abl))

# D1 数据卫生：off>dur 占比（日批）
bad, tot = 0, 0
for (bv, bt), g in d.groupby(["bvid", "batch"]):
    off = g["time_offset"].values.astype(float); dur = float(g["duration"].iloc[0])
    tot += len(off); bad += int((off > dur).sum())
R["data_hygiene_offset_gt_duration"] = dict(rows=tot, gt_duration=bad, share=round(bad / max(tot, 1), 6))

# ---------- 2) 复算 D2 + 补指标 ----------
sys.path.insert(0, "/home/molly/projects/danmaku/analysis")
from danmaku_metrics import density_curve, extract_8_indicators  # noqa: E402

DIMS = ["cold_pct", "cv", "main_peak_pct"]
groups = list(d.groupby(["bvid", "batch"]))
fullm, meta = {}, {}
for (bv, bt), g in groups:
    pub = float(g["pubdate"].iloc[0]); dur = float(g["duration"].iloc[0])
    if ((g["send_time"].values - pub) / 3600.0).max() < 1.5:
        continue
    fullm[(bv, bt)] = extract_8_indicators(density_curve(g["time_offset"].values, dur), dur)
    meta[(bv, bt)] = (pub, dur)
rws = []
for (bv, bt), g in groups:
    if (bv, bt) not in fullm:
        continue
    pub, dur = meta[(bv, bt)]
    st = g["send_time"].values; rel = (st - pub) / 3600.0
    off = g["time_offset"].values; fs = g["first_seen"].iloc[0]
    for t in [1, 3, 6, 12, 24]:
        if rel.max() < t + 0.5:
            continue
        m = rel <= t; k = int(m.sum())
        if k < 10:
            continue
        ind = extract_8_indicators(density_curve(off[m], dur), dur)
        rws.append(dict(bvid=bv, batch=bt, fs=fs, t=t, k=k,
                        cover=len(set((off[m] / max(dur, 1) * 20).astype(int))) / 20,  # 与 exp37 完全同口径（不 clip）
                        last1=float((rel >= t - 1).sum()) / k, dur=dur,
                        **{f"m_{x}": ind[x] for x in DIMS}, **{f"f_{x}": fullm[(bv, bt)][x] for x in DIMS}))
u2 = pd.DataFrame(rws)
tr2, te2 = u2[u2.fs <= 20261020], u2[u2.fs >= 20261021]
assert len(set(tr2.bvid) & set(te2.bvid)) == 0
q = {x: max(float(tr2[f"f_{x}"].quantile(.75) - tr2[f"f_{x}"].quantile(.25)), 1e-9) for x in DIMS}


def lab(df):
    err = np.max(np.column_stack([np.abs(df[f"m_{x}"] - df[f"f_{x}"]) / q[x] for x in DIMS]), axis=1)
    return (err <= 0.25).astype(int)


ytr2, yte2 = lab(tr2), lab(te2)
Xtr2 = np.column_stack([np.log1p(tr2.k), tr2.t, tr2.cover, tr2.last1, np.log1p(tr2.dur)])
Xte2 = np.column_stack([np.log1p(te2.k), te2.t, te2.cover, te2.last1, np.log1p(te2.dur)])
base2 = np.log1p(te2.k.values)
h2 = HistGradientBoostingClassifier(max_depth=2, min_samples_leaf=20, random_state=SEED).fit(Xtr2, ytr2)
pg2 = h2.predict_proba(Xte2)[:, 1]
l2 = LogisticRegression(max_iter=1000).fit(Xtr2, ytr2); pl2 = l2.predict_proba(Xte2)[:, 1]
vc2, vi2 = np.unique(te2.bvid.values, return_inverse=True)
rng2 = np.random.default_rng(SEED)
da = []
for _ in range(2000):
    idx = np.concatenate([np.where(vi2 == w)[0] for w in rng2.choice(len(vc2), len(vc2), True)])
    if len(np.unique(yte2[idx])) < 2:
        continue
    da.append(roc_auc_score(yte2[idx], pg2[idx]) - roc_auc_score(yte2[idx], base2[idx]))
k30 = int(len(yte2) * 0.3)
order = np.argsort(-pg2)
R["D2_recheck"] = dict(n_train=len(tr2), n_test=len(te2), bvid_overlap=0, s_d_frozen={k: round(v, 3) for k, v in q.items()},
                       AUC_base=round(roc_auc_score(yte2, base2), 4), AUC_HGB=round(roc_auc_score(yte2, pg2), 4),
                       dAUC=round(roc_auc_score(yte2, pg2) - roc_auc_score(yte2, base2), 4),
                       ci=[round(float(np.percentile(da, 2.5)), 4), round(float(np.percentile(da, 97.5)), 4)],
                       PR_AUC_base=round(average_precision_score(yte2, base2), 4),
                       PR_AUC_HGB=round(average_precision_score(yte2, pg2), 4),
                       precision_at_30pct_base=round(float(yte2[np.argsort(-base2)[:k30]].mean()), 4),
                       precision_at_30pct_HGB=round(float(yte2[order[:k30]].mean()), 4),
                       Brier_baseL0=round(float(yte2.mean()), 4), Brier_HGB=round(float(((pg2 - yte2) ** 2).mean()), 4),
                       Brier_LR=round(float(((pl2 - yte2) ** 2).mean()), 4),
                       replicate_exp37=dict(AUC_base=0.7914, AUC_HGB=0.8696, dAUC=0.0783))
log("[D2-recheck]", json.dumps({k: R["D2_recheck"][k] for k in ["AUC_base", "AUC_HGB", "dAUC", "ci", "PR_AUC_HGB", "precision_at_30pct_HGB", "Brier_HGB"]}))

# ---------- 3) D4 独立 cohort 验证（日批 1012-1026，exp35 口径） ----------
P_dt = rs.build_points()  # D-train 点表（已验证与冻结一致）
dt_bvids = set(P_dt.bvid.unique())
# 日批 → 点表（同口径）
cw = []
for (bv, bt), g in d.groupby(["bvid", "batch"]):
    if bv in dt_bvids:
        continue
    dur = float(g["duration"].iloc[0]); Kf = int(dur // 5)
    if Kf < 18 or float(g["part_count"].iloc[0] if pd.notna(g["part_count"].iloc[0]) else 0) >= 2:
        continue
    off = g["time_offset"].values.astype(float)
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
FEATS = ["log1p_lam", "slope", "since", "censored", "prog", "log1p_duration"]
# 生产版 D4：D-train 全量拟合 LR6 + Platt（单层，不再嵌套）
s_dt = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(P_dt[FEATS], P_dt.y).decision_function(P_dt[FEATS])
platt = LogisticRegression(C=1e6, max_iter=1000).fit(s_dt.reshape(-1, 1), P_dt.y)
joblib.dump({"pipe": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(P_dt[FEATS], P_dt.y),
             "platt": platt, "features": FEATS, "trained_on": "D-train 0902/0907 (4,862 pts / 182 vids)"},
            f"{MDIR}/d4_lr6_platt.joblib")
# cohort 评估（独立视频，时间外推 9 月→10 月）
s_c = joblib.load(f"{MDIR}/d4_lr6_platt.joblib")["pipe"].decision_function(Pc[FEATS])
p_c = platt.predict_proba(s_c.reshape(-1, 1))[:, 1]
yc, pc = Pc.y.values, np.clip(p_c, 1e-6, 1 - 1e-6)
lg = np.log(pc / (1 - pc))
sl = LogisticRegression(C=1e6, max_iter=1000).fit(lg.reshape(-1, 1), yc)
auc_lr6 = roc_auc_score(yc, s_c); auc_neg = roc_auc_score(yc, -Pc.log1p_lam.values)
R["D4_independent_cohort"] = dict(
    cohort="daily 1012-1026, bvid not in D-train (time-extrapolated)",
    n_points=len(Pc), n_videos=int(Pc.bvid.nunique()), pos_rate=round(float(yc.mean()), 4),
    AUC_LR6=round(float(auc_lr6), 4), AUC_NegLam=round(float(auc_neg), 4),
    Brier_cal=round(float(((pc - yc) ** 2).mean()), 4), Brier_L0=round(float(yc.mean()), 4),
    brier_skill=round(float(1 - ((pc - yc) ** 2).mean() / yc.mean()), 4),
    cal_slope=round(float(sl.coef_[0][0]), 3), cal_intercept=round(float(sl.intercept_[0]), 3),
    note="LR6+Platt 在 D-train 全量拟合后直接用于时间外推 cohort（生产口径）；不属 exp35 冻结协议")
log("[D4-cohort]", json.dumps(R["D4_independent_cohort"]))

# ---------- 4) 生产工件 D1/D2 ----------
ols_prod = LinearRegression().fit(mats(u, STEPS[-1]), np.log1p(u.y.values))
res_log = np.log1p(u.y.values) - ols_prod.predict(mats(u, STEPS[-1]))
d1_art = dict(model="OLS-6 (6 features: c0,n36,n13,n01,slope,dur; production full-window refit)", features=STEPS[-1],
              coef=[float(x) for x in ols_prod.coef_], intercept=float(ols_prod.intercept_),
              resid_log_q=dict(p10=round(float(np.quantile(res_log, .1)), 4), p50=round(float(np.median(res_log)), 4),
                               p90=round(float(np.quantile(res_log, .9)), 4)),
              trained_on="daily 1012-1026 full window (1,313 units / videos age>=31h)", eval_ref="exp38 audit (time-split CI)")
json.dump(d1_art, open(f"{MDIR}/d1_ols.json", "w"), ensure_ascii=False, indent=2)
joblib.dump({"model": h2, "features": ["log1p_k", "t", "cover", "last1", "log1p_dur"],
             "s_d_frozen": q, "trained_on": f"daily 1012-1026 full window ({len(u2)} units)"}, f"{MDIR}/d2_hgb.joblib")
json.dump(R, open(f"{OUT}/audit_results.json", "w"), ensure_ascii=False, indent=2, default=str)
log("AUDIT-DONE →", f"{OUT}/audit_results.json")
