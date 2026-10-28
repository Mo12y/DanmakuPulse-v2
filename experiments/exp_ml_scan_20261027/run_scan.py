# -*- coding: utf-8 -*-
"""exp_ml_scan_20261027 · 剩余 ML 机会一次性验证（只读数据，新目录，不碰 D-dev/D-conf/冻结产物）

方向（预注册成功判据，先于运行写死）：
  D1 未来24h弹幕量回归：时间切(bvid不相交) 测试集 ΔR²(HGB−OLS单特征基线) ≥0.05 且 bootstrap 95%CI下界>0
  D2 结论稳定度(RCP)：时间切(bvid不相交) ΔAUC(HGB−log1p(k)单调基线) ≥0.03 且 CI下界>0
  D3 30s计数回归(expD_count协议移植)：视频内log1p R² Δ(OLS6−6λ基线) 95%CI下界>0（视频级GroupKFold隔离）
  D4 爆发风险概率校准：嵌套OOF Platt 后 Brier skill>0 且 校准斜率∈[0.8,1.25]
数据边界：日批 20261012–20261026（不含 0914/0917=D-dev，不含 1027+=D-conf/待裁定）；
         旧仓 warehouse ODS 仅取 dt∈{20260902,20260907}=D-train（前轮已授权口径）。
"""
import glob
import json
import os
import re
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 20261027
B_BOOT = 2000
V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp_ml_scan_20261027"
OLD_WH = "/home/molly/projects/danmaku/data/warehouse/ods"
os.makedirs(OUT, exist_ok=True)
RES = {}


def log(*a):
    print(*a, flush=True)


def boot_ci(diff, B=B_BOOT, seed=SEED, unit=None):
    """bootstrap 均值 CI；unit=分组重采样索引数组"""
    rng = np.random.default_rng(seed)
    n = len(diff)
    if unit is None:
        idx = rng.integers(0, n, size=(B, n))
        means = diff[idx].mean(axis=1)
    else:
        ug = np.unique(unit)
        m = means = np.empty(B)
        for b in range(B):
            pick = rng.choice(ug, size=len(ug), replace=True)
            sel = np.concatenate([np.where(unit == u)[0] for u in pick])
            means[b] = diff[sel].mean()
    return float(np.nanmean(diff)), float(np.nanpercentile(means, 2.5)), float(np.nanpercentile(means, 97.5))


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    sst = ((y - y.mean()) ** 2).sum()
    return float("nan") if sst <= 0 else float(1 - ((y - p) ** 2).sum() / sst)


# ================= 数据构建 A：日批 1012–1026 =================
def build_daily():
    cols = ["bvid", "duration", "pubdate", "time_offset", "send_time", "danmaku_got", "status", "part_count"]
    parts = []
    for f in sorted(glob.glob(f"{V2}/data/cli_collected/hotlist_top200_202610*.csv")):
        day = re.search(r"(2026\d{4})", os.path.basename(f)).group(1)
        if not ("20261012" <= day <= "20261026"):
            continue
        df = pd.read_csv(f, usecols=cols, encoding="utf-8-sig", low_memory=False)
        df["batch"] = int(day)
        parts.append(df)
    d = pd.concat(parts, ignore_index=True)
    for c in ["duration", "pubdate", "time_offset", "send_time", "danmaku_got", "part_count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["status"] == "ok") & d[["pubdate", "duration", "send_time"]].notna().all(axis=1)]
    d = d[d["duration"] > 0]
    d = d[~(d["part_count"].fillna(0) >= 2)]  # 排除已知多P
    first_seen = d.groupby("bvid")["batch"].min()
    d["first_seen"] = d["bvid"].map(first_seen)
    log(f"[build] rows={len(d)} videos={d.bvid.nunique()} batches={d.batch.nunique()} "
        f"train_vids(<=1020)={(first_seen<=20261020).sum()} test_vids(>=1021)={(first_seen>=20261021).sum()}")
    return d


# ================= D1 未来24h弹幕量 =================
def d1_volume(d):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    rows = []
    for (bv, bt), g in d.groupby(["bvid", "batch"]):
        pub = float(g["pubdate"].iloc[0]); dur = float(g["duration"].iloc[0])
        rel = (g["send_time"].values - pub) / 3600.0  # 小时
        age = rel.max()
        if age < 31.0:
            continue
        n01 = ((rel >= 0) & (rel < 1)).sum(); n13 = ((rel >= 1) & (rel < 3)).sum()
        n36 = ((rel >= 3) & (rel < 6)).sum(); c0 = n01 + n13 + n36
        if c0 < 1:
            continue
        yw = ((rel > 6) & (rel <= 30)).sum()
        rows.append((bv, bt, g["first_seen"].iloc[0], n01, n13, n36, c0, yw, dur))
    u = pd.DataFrame(rows, columns=["bvid", "batch", "fs", "n01", "n13", "n36", "c0", "y", "dur"])
    tr = u[u.fs <= 20261020]; te = u[u.fs >= 20261021]
    ytr, yte = np.log1p(tr.y.values), np.log1p(te.y.values)

    def X(df):
        return np.column_stack([np.log1p(df.n01), np.log1p(df.n13), np.log1p(df.n36),
                                np.log1p(df.n36) - np.log1p(df.n13), np.log1p(df.dur)])
    Xtr, Xte = X(tr), X(te)
    ols1 = LinearRegression().fit(np.log1p(tr.c0.values).reshape(-1, 1), ytr)
    p1 = ols1.predict(np.log1p(te.c0.values).reshape(-1, 1))
    ols6 = LinearRegression().fit(Xtr, ytr); p6 = ols6.predict(Xte)
    gb = HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=20, random_state=SEED).fit(Xtr, ytr)
    pg = gb.predict(Xte)
    r1, r6, rg = r2(yte, p1), r2(yte, p6), r2(yte, pg)
    # 视频级聚类 bootstrap（同视频多快照相关）→ 直接在重采样集合上重算两个 R²
    vcodes, vidx = np.unique(te.bvid.values, return_inverse=True)
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(vcodes), size=len(vcodes), replace=True)
        idx = np.concatenate([np.where(vidx == u)[0] for u in pick])
        a_, b_ = r2(yte[idx], pg[idx]), r2(yte[idx], p1[idx])
        if np.isfinite(a_) and np.isfinite(b_):
            diffs.append(a_ - b_)
    lo, hi = np.percentile(diffs, [2.5, 97.5]); m = float(np.mean(diffs))
    diffs2 = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(vcodes), size=len(vcodes), replace=True)
        idx = np.concatenate([np.where(vidx == u)[0] for u in pick])
        a_, b_ = r2(yte[idx], p6[idx]), r2(yte[idx], p1[idx])
        if np.isfinite(a_) and np.isfinite(b_):
            diffs2.append(a_ - b_)
    lo2, hi2 = np.percentile(diffs2, [2.5, 97.5]); m2 = float(np.mean(diffs2))
    out = dict(n_train=len(tr), n_test=len(te), test_videos=te.bvid.nunique(),
               bvid_overlap=int(len(set(tr.bvid) & set(te.bvid))),
               R2_baseline_OLS_c0=round(r1, 4), R2_OLS6=round(r6, 4), R2_HGB=round(rg, 4),
               dR2_HGB_vs_OLS1=round(m, 4), ci=[round(lo, 4), round(hi, 4)],
               dR2_OLS6_vs_OLS1_secondary=round(m2, 4), ci_secondary=[round(lo2, 4), round(hi2, 4)],
               verdict=("POSITIVE" if (m >= 0.05 and lo > 0) else "NEGATIVE" if hi <= 0 else "INCONCLUSIVE"))
    log("[D1]", json.dumps(out, ensure_ascii=False))
    return out


# ================= D2 结论稳定度 RCP（复用 audit_conv 口径） =================
def d2_rcp(d):
    sys.path.insert(0, "/home/molly/projects/danmaku/analysis")
    from danmaku_metrics import density_curve, extract_8_indicators
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    DIMS = ["cold_pct", "cv", "main_peak_pct"]
    rows, fullm = [], {}
    groups = list(d.groupby(["bvid", "batch"]))
    for (bv, bt), g in groups:
        pub = float(g["pubdate"].iloc[0]); dur = float(g["duration"].iloc[0])
        st = g["send_time"].values; rel = (st - pub) / 3600.0
        if rel.max() < 1.5:
            continue
        fullm[(bv, bt)] = extract_8_indicators(density_curve(g["time_offset"].values, dur), dur)
    for (bv, bt), g in groups:
        if (bv, bt) not in fullm:
            continue
        pub = float(g["pubdate"].iloc[0]); dur = float(g["duration"].iloc[0])
        st = g["send_time"].values; rel = (st - pub) / 3600.0
        off = g["time_offset"].values; fs = g["first_seen"].iloc[0]
        for t in [1, 3, 6, 12, 24]:
            if rel.max() < t + 0.5:
                continue
            m = rel <= t
            k = int(m.sum())
            if k < 10:
                continue
            ind = extract_8_indicators(density_curve(off[m], dur), dur)
            cover = len(set((off[m] / max(dur, 1) * 20).astype(int))) / 20
            last1 = float((rel >= t - 1).sum()) / k
            rows.append(dict(bvid=bv, batch=bt, fs=fs, t=t, k=k, cover=cover, last1=last1, dur=dur,
                             **{f"m_{x}": ind[x] for x in DIMS}, **{f"f_{x}": fullm[(bv, bt)][x] for x in DIMS}))
    u = pd.DataFrame(rows)
    tr = u[u.fs <= 20261020]; te = u[u.fs >= 20261021]
    q = {}
    for x in DIMS:
        iqr = tr[f"f_{x}"].quantile(.75) - tr[f"f_{x}"].quantile(.25)
        q[x] = max(float(iqr), 1e-9)  # 训练期冻结
    def lab(df):
        err = np.max(np.column_stack([np.abs(df[f"m_{x}"] - df[f"f_{x}"]) / q[x] for x in DIMS]), axis=1)
        return (err <= 0.25).astype(int)
    ytr, yte = lab(tr), lab(te)
    Xtr = np.column_stack([np.log1p(tr.k), tr.t, tr.cover, tr.last1, np.log1p(tr.dur)])
    Xte = np.column_stack([np.log1p(te.k), te.t, te.cover, te.last1, np.log1p(te.dur)])
    base = np.log1p(te.k.values)  # 单调零参基线
    hgb = HistGradientBoostingClassifier(max_depth=2, min_samples_leaf=20, random_state=SEED).fit(Xtr, ytr)
    pg = hgb.predict_proba(Xte)[:, 1]
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr); pl = lr.predict_proba(Xte)[:, 1]
    auc_b, auc_g, auc_l = roc_auc_score(yte, base), roc_auc_score(yte, pg), roc_auc_score(yte, pl)
    # 视频级聚类 bootstrap ΔAUC
    rng = np.random.default_rng(SEED)
    vcodes, vidx = np.unique(te.bvid.values, return_inverse=True)
    diffs = []
    for _ in range(B_BOOT):
        pick = rng.choice(len(vcodes), size=len(vcodes), replace=True)
        idx = np.concatenate([np.where(vidx == u)[0] for u in pick])
        if len(np.unique(yte[idx])) < 2:
            continue
        diffs.append(roc_auc_score(yte[idx], pg[idx]) - roc_auc_score(yte[idx], base[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # GroupKFold OOF 次要
    gkf = GroupKFold(5); oof = np.zeros(len(u))
    Xa = np.column_stack([np.log1p(u.k), u.t, u.cover, u.last1, np.log1p(u.dur)])
    ya = lab(u)
    for trn, tst in gkf.split(Xa, ya, groups=u.bvid.values):
        m2 = HistGradientBoostingClassifier(max_depth=2, min_samples_leaf=20, random_state=SEED).fit(Xa[trn], ya[trn])
        oof[tst] = m2.predict_proba(Xa[tst])[:, 1]
    out = dict(n_train=len(tr), n_test=len(te), test_videos=te.bvid.nunique(),
               pos_rate_test=round(float(yte.mean()), 3), s_d_frozen={k: round(v, 3) for k, v in q.items()},
               bvid_overlap=int(len(set(tr.bvid) & set(te.bvid))),
               AUC_baseline_logk=round(auc_b, 4), AUC_HGB=round(auc_g, 4), AUC_LR=round(auc_l, 4),
               AUC_HGB_gkf_oof=round(roc_auc_score(ya, oof), 4),
               dAUC_HGB_vs_base=round(auc_g - auc_b, 4), ci=[round(float(lo), 4), round(float(hi), 4)],
               verdict=("POSITIVE" if (auc_g - auc_b >= 0.03 and lo > 0) else "NEGATIVE" if hi <= 0 else "INCONCLUSIVE"))
    log("[D2]", json.dumps(out, ensure_ascii=False))
    return out


# ================= D3/D4 构建：旧仓 ODS 0902/0907 → 5s 桶点表 =================
def build_points():
    dt_dirs = [p for p in sorted(glob.glob(f"{OLD_WH}/dt=*")) if re.search(r"2026090[27]", p)]
    log("[pts] dt dirs:", [os.path.basename(p) for p in dt_dirs])
    fr = []
    for p in dt_dirs:
        fs = [f for f in glob.glob(f"{p}/**/*.parquet", recursive=True) if not os.path.basename(f).startswith(".")]
        x = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        x["batch"] = int(re.search(r"(2026\d{4})", os.path.basename(p)).group(1))
        fr.append(x)
    a = pd.concat(fr, ignore_index=True)
    for c in ["duration", "pubdate", "time_offset"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.dropna(subset=["duration", "time_offset"])
    a = a[(a["duration"] > 0) & (a["time_offset"] >= 0)]
    pts_ref = pd.read_parquet(f"{V2}/data-out/m2_l6/ods/points_train.parquet")
    pts_ref["batch"] = pd.to_numeric(pts_ref["batch"]).astype(int)
    pc = pts_ref.groupby(["bvid", "batch"])["part_count"].first()
    rows = []
    for (bv, bt), g in a.groupby(["bvid", "batch"]):
        pcv = pc.get((bv, bt), 0)
        if pd.notna(pcv) and pcv >= 2:
            continue
        dur = float(g["duration"].iloc[0])
        Kf = int(dur // 5)
        off = g["time_offset"].values.astype(float)
        edge = np.isclose(off, dur)
        off = off[~edge]
        b = (off // 5).astype(int)
        b = b[b < Kf]
        if Kf < 18:
            continue
        c = np.bincount(b, minlength=Kf)
        e = np.zeros(Kf, bool)
        for t in range(12, Kf - 5, 6):
            lam = float(c[t - 12:t].mean())
            thr = max(3.0, lam + 2.0 * np.sqrt(max(lam, 0.0)))
            yb = int(c[t:t + 6].max() >= thr) if t + 6 <= Kf else None
            if t + 6 > Kf:
                continue
            sinc, cens = 12, 1
            for dd in range(1, 13):
                k = t - dd
                if k < 12:
                    break
                lk = float(c[k - 12:k].mean())
                tk = max(3.0, lk + 2.0 * np.sqrt(max(lk, 0.0)))
                if c[k] >= tk:
                    sinc, cens = dd, 0
                    break
            rows.append((bv, bt, t, lam, yb, float(c[t:t + 6].sum()), sinc, cens,
                         float(np.log1p(lam)), float(np.log1p(c[t - 6:t].mean()) - np.log1p(c[t - 12:t - 6].mean())),
                         t / Kf, float(np.log1p(dur))))
    P = pd.DataFrame(rows, columns=["bvid", "batch", "t", "lam", "y", "y_count", "since", "censored",
                                    "log1p_lam", "slope", "prog", "log1p_duration"])
    # 与冻结构建交叉校验（有效性检查）
    m = P.merge(pts_ref[["bvid", "batch", "t", "lam", "y"]], on=["bvid", "batch", "t"], suffixes=("", "_ref"))
    lam_ok = float((np.abs(m.lam - m.lam_ref) < 1e-6).mean()); y_ok = float((m.y == m.y_ref).mean())
    log(f"[pts] rebuilt={len(P)} pts / {P.bvid.nunique()} videos; cross-check vs frozen build: "
        f"n_matched={len(m)} lam_match={lam_ok:.4f} y_match={y_ok:.4f} (frozen total={len(pts_ref)})")
    return P


# ================= D3 30s计数回归 =================
def d3_count(P):
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold
    Pv = P.groupby("bvid").filter(lambda g: len(g) >= 6).reset_index(drop=True)
    ylog = np.log1p(Pv.y_count.values)
    base = np.log1p(6 * Pv.lam.values)
    X = Pv[["log1p_lam", "slope", "since", "censored", "prog", "log1p_duration"]].values
    try:
        gkf = GroupKFold(5, shuffle=True, random_state=SEED)
    except TypeError:
        gkf = GroupKFold(5)
    oof = np.zeros(len(Pv))
    for trn, tst in gkf.split(X, ylog, groups=Pv.bvid.values):
        ols = LinearRegression().fit(X[trn], ylog[trn])
        oof[tst] = ols.predict(X[tst])
    vids, d_r2 = [], []
    for bv, g in Pv.groupby("bvid"):
        i = g.index.to_numpy()
        yy, po, pb = ylog[i], oof[i], base[i]
        sst = ((yy - yy.mean()) ** 2).sum()
        if sst <= 0 or len(yy) < 2:
            continue
        r_m = 1 - ((yy - po) ** 2).sum() / sst
        r_b = 1 - ((yy - pb) ** 2).sum() / sst
        vids.append(bv); d_r2.append(r_m - r_b)
    d_r2 = np.array(d_r2)
    m, lo, hi = boot_ci(d_r2, unit=np.arange(len(d_r2)))
    out = dict(n_points=len(Pv), n_videos_elig=len(d_r2),
               R2_base_6lam=round(float(d_r2.mean() + m - d_r2.mean()), 4),  # 占位，下一行覆盖
               dR2=round(m, 4), ci=[round(lo, 4), round(hi, 4)],
               verdict=("POSITIVE" if lo > 0 else "NEGATIVE" if hi <= 0 else "INCONCLUSIVE"))
    # 直接算两侧宏平均
    r2s_m, r2s_b = [], []
    for bv, g in Pv.groupby("bvid"):
        i = g.index.to_numpy(); yy = ylog[i]
        sst = ((yy - yy.mean()) ** 2).sum()
        if sst <= 0 or len(yy) < 2:
            continue
        r2s_m.append(1 - ((yy - oof[i]) ** 2).sum() / sst)
        r2s_b.append(1 - ((yy - base[i]) ** 2).sum() / sst)
    out["R2_model_macro"] = round(float(np.mean(r2s_m)), 4)
    out["R2_base_macro"] = round(float(np.mean(r2s_b)), 4)
    out.pop("R2_base_6lam")
    # 敏感性：λ>0 子集（阈值不退化的位置）
    Pz = Pv[Pv.lam > 0].reset_index(drop=True)
    if len(Pz) >= 200:
        ylz = np.log1p(Pz.y_count.values); Xz = Pz[["log1p_lam", "slope", "since", "censored", "prog", "log1p_duration"]].values
        baz = np.log1p(6 * Pz.lam.values)
        oofz = np.zeros(len(Pz))
        for trn, tst in gkf.split(Xz, ylz, groups=Pz.bvid.values):
            ols = LinearRegression().fit(Xz[trn], ylz[trn]); oofz[tst] = ols.predict(Xz[tst])
        rm, rb = [], []
        for bv, g in Pz.groupby("bvid"):
            i = g.index.to_numpy(); yy = ylz[i]
            sst = ((yy - yy.mean()) ** 2).sum()
            if sst <= 0 or len(yy) < 2:
                continue
            rm.append(1 - ((yy - oofz[i]) ** 2).sum() / sst); rb.append(1 - ((yy - baz[i]) ** 2).sum() / sst)
        out["sens_lam_gt0"] = dict(n_points=len(Pz), n_videos=len(rm),
                                   R2_model_macro=round(float(np.mean(rm)), 4), R2_base_macro=round(float(np.mean(rb)), 4))
    log("[D3]", json.dumps(out, ensure_ascii=False))
    return out


# ================= D4 概率校准 =================
def d4_calib(P):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    y = P.y.values
    s1 = -P.log1p_lam.values  # NegLam 零参基线
    X = P[["log1p_lam", "slope", "since", "censored", "prog", "log1p_duration"]].values
    try:
        gkf = GroupKFold(5, shuffle=True, random_state=SEED)
    except TypeError:
        gkf = GroupKFold(5)
    folds = list(gkf.split(X, y, groups=P.bvid.values))
    s2 = np.zeros(len(P))  # LR6 OOF 分数
    for trn, tst in folds:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
        pipe.fit(X[trn], y[trn]); s2[tst] = pipe.decision_function(X[tst])
    def platt_oof(s):
        p = np.zeros(len(P))
        for trn, tst in folds:
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(s[trn].reshape(-1, 1), y[trn])
            p[tst] = pl.predict_proba(s[tst].reshape(-1, 1))[:, 1]
        return p
    def eval_p(p, name):
        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        brier = float(((pc - y) ** 2).mean()); b0 = float(y.mean())
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(pc / (1 - pc)).reshape(-1, 1), y)
        slope, inter = float(lr.coef_[0][0]), float(lr.intercept_[0])
        dec = np.quantile(pc, np.linspace(0, 1, 11))
        rel = [float(y[(pc >= dec[i]) & (pc <= dec[i + 1])].mean() - pc[(pc >= dec[i]) & (pc <= dec[i + 1])].mean())
               for i in range(10)]
        return dict(name=name, brier=round(brier, 4), brier_L0=round(b0, 4),
                    brier_skill=round(float(1 - brier / b0), 4), cal_slope=round(slope, 3), cal_intercept=round(inter, 3),
                    max_decile_gap=round(float(np.max(np.abs(rel))), 3))
    outs = [eval_p(platt_oof(s1), "NegLam+Platt"), eval_p(platt_oof(s2), "LR6_OOF+Platt")]
    ok = [o for o in outs if o["brier_skill"] > 0 and 0.8 <= o["cal_slope"] <= 1.25]
    out = dict(n_points=len(P), n_videos=int(P.bvid.nunique()), pos_rate=round(float(y.mean()), 3),
               models=outs, verdict=("POSITIVE" if ok else "NEGATIVE"))
    log("[D4]", json.dumps(out, ensure_ascii=False))
    return out


# ================= 主流程 =================
def main():
    t0 = time.time()
    RES["env"] = dict(seed=SEED, B=B_BOOT, python=sys.version.split()[0],
                      pandas=pd.__version__, scope="daily 1012-1026 + old-ods 0902/0907 (D-train only)")
    try:
        d = build_daily()
    except Exception:
        log(traceback.format_exc()); RES["build_daily"] = "FAILED"; d = None
    if d is not None:
        for name, fn in [("D1_volume", d1_volume), ("D2_rcp_stability", d2_rcp)]:
            try:
                RES[name] = fn(d)
            except Exception:
                RES[name] = dict(error=traceback.format_exc()[-800:]); log(f"[{name}] FAILED")
    try:
        P = build_points()
    except Exception:
        log(traceback.format_exc()); RES["build_points"] = "FAILED"; P = None
    if P is not None:
        for name, fn in [("D3_count30s", d3_count), ("D4_calibration", d4_calib)]:
            try:
                RES[name] = fn(P)
            except Exception:
                RES[name] = dict(error=traceback.format_exc()[-800:]); log(f"[{name}] FAILED")
    RES["elapsed_sec"] = round(time.time() - t0, 1)
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(RES, f, ensure_ascii=False, indent=2, default=str)
    log("DONE", RES["elapsed_sec"], "s →", f"{OUT}/results.json")


if __name__ == "__main__":
    import time
    main()
