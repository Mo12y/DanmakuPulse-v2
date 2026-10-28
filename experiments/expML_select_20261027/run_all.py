#!/usr/bin/env python3
"""expML_select_20261027: one-shot D-train-only comparison.

Reuses the frozen D-train sample (5,157 points / 134 videos, bvid 5-fold,
fold seed 20261026) and existing OOF probabilities. No D-dev/D-conf.
No grid search. No modification of exp35 assets.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "jobs"))
import m2_l6_burst_risk as M  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

OUT = HERE
OOFDIR = os.path.join(OUT, "oof")
EXIST = os.path.join(REPO, "data-out", "exp35_models")
SEED = 42
BOOT_B = 2000
BOOT_SEED = 20261026

warnings.filterwarnings("ignore")


def scaled(clf):
    return Pipeline([("sc", StandardScaler()), ("clf", clf)])


def eval_oof(name, oof, *, prob=True, task="direction2", runtime=None, note="", extra=None):
    oof = np.asarray(oof, dtype=float)
    if oof.size != y.size or not np.isfinite(oof).all():
        return {"model": name, "task": task, "status": "failed",
                "error": f"oof size={oof.size} finite={np.isfinite(oof).all()}"}
    m = M.macro_within_video_auc(y, oof, vid)
    per = np.asarray(list(m["per_video"].values()), dtype=float)
    folds = []
    for f in range(M.CV_FOLDS):
        te = fold_id == f
        try:
            folds.append(float(M.macro_within_video_auc(y[te], oof[te], vid[te])["macro"]))
        except Exception:
            folds.append(float("nan"))
    finite = [v for v in folds if np.isfinite(v)]
    rec = {
        "task": task, "model": name, "status": "ok",
        "n_points": int(oof.size), "n_videos": int(m["n_eligible"]),
        "macro_auc": float(m["macro"]),
        "auc_p10": float(np.percentile(per, 10)),
        "auc_p50": float(np.percentile(per, 50)),
        "auc_p90": float(np.percentile(per, 90)),
        "auc_sd": float(per.std(ddof=1)) if per.size > 1 else None,
        "share_auc_gt_half": float((per > 0.5).mean()),
        "pooled_auc": float(roc_auc_score(y, oof)),
        "pr_auc": float(average_precision_score(y, oof)),
        "brier": float(brier_score_loss(y, oof)) if prob else None,
        "fold_aucs": folds,
        "n_folds_gt_half": int(sum(v > 0.5 for v in finite)),
        "fold_min": float(min(finite)) if finite else None,
        "fold_max": float(max(finite)) if finite else None,
        "fold_mean": float(np.mean(finite)) if finite else None,
        "runtime_s": runtime,
        "note": note,
    }
    if name != "NegLam":
        try:
            b = M.paired_bootstrap_delta(y, oof, neg, vid, B=BOOT_B, seed=BOOT_SEED)
            rec.update({"delta_vs_neg": float(b["delta"]), "neg_lo": float(b["ci_lo"]),
                        "neg_hi": float(b["ci_hi"]), "neg_reps": int(b["n_valid_reps"]),
                        "neg_discarded": int(b["n_discarded"])})
        except Exception as exc:
            rec["neg_error"] = f"{type(exc).__name__}: {exc}"[:160]
    if extra:
        rec.update(extra)
    return rec


def oof_fit(X, factory):
    oof = np.full(len(y), np.nan, dtype=float)
    for f in range(M.CV_FOLDS):
        te = fold_id == f
        tr = ~te
        est = factory()
        est.fit(X[tr], y[tr])
        oof[te] = est.predict_proba(X[te])[:, 1]
    return oof


# ── data: exact frozen D-train sample ────────────────────────────────
seqs, _ = M.load_sequences(M.DEFAULT_MANIFEST, M.DEFAULT_DATA_ROOT,
                           batches=M.D_TRAIN_BATCHES, layer="ods")
rows_all = M.filter_multip(M.build_rows(seqs), "keep")
elig = set(M.eligible_video_ids(rows_all))
rows = [r for r in rows_all if r["bvid"] in elig]
y = np.array([r["y"] for r in rows], dtype=int)
vid = np.array([r["bvid"] for r in rows], dtype=str)
fold_of = M.group_folds(vid, folds=M.CV_FOLDS, seed=M.CV_SEED)
fold_id = np.array([fold_of[v] for v in vid], dtype=int)
X1 = M.select_features(rows, M.FEATURES_M1)
X6 = M.select_features(rows, M.FEATURES_M2)
lam = np.array([float(r["log1p_lam"]) for r in rows], dtype=float)
neg = -lam
assert len(y) == 5157 and len(set(vid)) == 134, (len(y), len(set(vid)))

# ── direction 1/2: load existing OOFs ────────────────────────────────
EXIST_MODELS = {
    "M1_LR_1feat": "oof_round1_M1_LR_1feat.npy",
    "LR_6feat": "oof_round1_LR_6feat.npy",
    "RF": "oof_round1_RF.npy",
    "ExtraTrees": "oof_round1_ExtraTrees.npy",
    "HistGB": "oof_round1_HistGB.npy",
    "SVM_RBF": "oof_round1_SVM_RBF.npy",
    "MLP": "oof_round1_MLP.npy",
    "XGBoost": "oof_round1_XGBoost.npy",
    "LightGBM": "oof_round1_LightGBM.npy",
    "CatBoost": "oof_round1_CatBoost.npy",
}
oofs = {}
for name, fn in EXIST_MODELS.items():
    a = np.load(os.path.join(EXIST, fn))
    assert a.size == y.size, (name, a.size)
    oofs[name] = a

results = []
base = eval_oof("NegLam", neg, prob=False, task="direction2", runtime=0.0,
                note="fixed -log1p_lam; no fitting")
results.append(base)

for name, a in oofs.items():
    results.append(eval_oof(name, a, prob=True, task="direction2"))

# new regularized linear (L1)
t0 = time.time()
l1 = oof_fit(X6, lambda: scaled(LogisticRegression(
    penalty="l1", C=1.0, solver="liblinear", max_iter=1000, random_state=SEED)))
rt = time.time() - t0
np.save(os.path.join(OOFDIR, "LR_L1_6feat.npy"), l1)
results.append(eval_oof("LR_L1_6feat", l1, prob=True, task="direction2", runtime=round(rt, 2)))

dir2 = [r for r in results if r["task"] == "direction2"]

# ── direction 1 audit ────────────────────────────────────────────────
by = {r["model"]: r for r in dir2}
m1 = oofs["M1_LR_1feat"]
m2 = oofs["LR_6feat"]
corr_folds = []
for f in range(M.CV_FOLDS):
    te = fold_id == f
    corr_folds.append(float(np.corrcoef(m1[te], lam[te])[0, 1]))
m2_m1 = M.paired_bootstrap_delta(y, m2, m1, vid, B=BOOT_B, seed=BOOT_SEED)
m2_neg = M.paired_bootstrap_delta(y, m2, neg, vid, B=BOOT_B, seed=BOOT_SEED)
m1_neg = M.paired_bootstrap_delta(y, m1, neg, vid, B=BOOT_B, seed=BOOT_SEED)
direction1 = {
    "sample": {"n_points": int(y.size), "n_videos": int(len(set(vid))),
               "pos": int(y.sum()), "rate": float(y.mean())},
    "m1": {"macro_auc": by["M1_LR_1feat"]["macro_auc"],
           "fold_aucs": by["M1_LR_1feat"]["fold_aucs"],
           "fold0_auc": by["M1_LR_1feat"]["fold_aucs"][0],
           "score_log1p_lam_corr_by_fold": corr_folds,
           "p10": by["M1_LR_1feat"]["auc_p10"]},
    "neg_lam": {"macro_auc": by["NegLam"]["macro_auc"],
                "p10": by["NegLam"]["auc_p10"]},
    "m2": {"macro_auc": by["LR_6feat"]["macro_auc"]},
    "deltas": {
        "m2_minus_m1": {"delta": float(m2_m1["delta"]), "lo": float(m2_m1["ci_lo"]),
                        "hi": float(m2_m1["ci_hi"])},
        "m2_minus_neg": {"delta": float(m2_neg["delta"]), "lo": float(m2_neg["ci_lo"]),
                         "hi": float(m2_neg["ci_hi"])},
        "m1_minus_neg": {"delta": float(m1_neg["delta"]), "lo": float(m1_neg["ci_lo"]),
                         "hi": float(m1_neg["ci_hi"])},
    },
    "reading": "M1 折0 AUC<0.5 且 score~log1p_lam 相关为正 => 该折系数符号翻转；M1 宏AUC不可作为稳定基线，M2-M1 的 +0.004 不能解释为时序特征增量。",
}

# ── direction 4 ablation ─────────────────────────────────────────────
SETS = {
    "Abl_M1": ("log1p_lam",),
    "Abl_M2": M.FEATURES_M2,
    "Abl_M2_no_slope": ("log1p_lam", "since", "censored", "prog", "log1p_duration"),
    "Abl_M2_no_since": ("log1p_lam", "slope", "prog", "log1p_duration"),
    "Abl_M1_plus_slope": ("log1p_lam", "slope"),
}
ablation = []
for name, cols in SETS.items():
    if name == "Abl_M1":
        a = m1
        rt = 0.0
    elif name == "Abl_M2":
        a = m2
        rt = 0.0
    else:
        t0 = time.time()
        X = M.select_features(rows, cols)
        a = oof_fit(X, lambda: scaled(LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)))
        rt = round(time.time() - t0, 2)
        np.save(os.path.join(OOFDIR, f"{name}.npy"), a)
    rec = eval_oof(name, a, prob=True, task="ablation", runtime=rt,
                   note="features=" + ",".join(cols))
    for ref_name, ref in (("M1", m1), ("M2", m2)):
        b = M.paired_bootstrap_delta(y, a, ref, vid, B=BOOT_B, seed=BOOT_SEED)
        rec[f"delta_vs_{ref_name}"] = float(b["delta"])
        rec[f"{ref_name}_lo"] = float(b["ci_lo"])
        rec[f"{ref_name}_hi"] = float(b["ci_hi"])
    ablation.append(rec)

# ── direction 3: window counts and gated core runs ───────────────────
dur_of = {s.bvid: s.duration for s in seqs}


def window_stats(h):
    pts = []
    for s in seqs:
        pts.extend(M.label_points(s, h=h))
    n_points = len(pts)
    n_vids = len({p.bvid for p in pts})
    pos = int(sum(p.y for p in pts))
    agg = {}
    for p in pts:
        a = agg.setdefault(p.bvid, [0, 0])
        a[0] += 1
        a[1] += int(p.y)
    elig_v = [v for v, (n, q) in agg.items() if n >= M.MIN_POINTS and q >= M.MIN_POS and (n - q) >= M.MIN_NEG]
    elig_pts = int(sum(agg[v][0] for v in elig_v))
    med_pos = float(np.median([agg[v][1] for v in elig_v])) if elig_v else None
    partial_v = sum(1 for s in seqs if s.duration % M.B != 0)
    last_partial = sum(1 for p in pts if p.win_end == p.K and dur_of[p.bvid] % M.B != 0)
    gate = bool(len(elig_v) >= 100 and elig_pts >= 500 and 0.15 <= pos / n_points <= 0.60
                and (med_pos is not None and med_pos >= 3))
    return {"h_buckets": h, "h_seconds": h * M.B, "n_points": n_points,
            "n_videos": n_vids, "n_pos": pos, "pos_rate": pos / n_points,
            "eligible_videos": len(elig_v), "eligible_points": elig_pts,
            "median_pos_per_eligible_video": med_pos,
            "videos_duration_mod5_nonzero": partial_v,
            "last_window_partial_points": last_partial,
            "gate_pass": gate}


def build_rows_h(h):
    out = []
    for s in seqs:
        for pt in M.label_points(s, h=h):
            feats = M.feature_row(s, pt.t)
            row = pt.as_row()
            row.update(feats)
            row["part_count"] = s.part_count
            out.append(row)
    ok = set(M.eligible_video_ids(out))
    return [r for r in out if r["bvid"] in ok]


def core_for_h(h):
    rr = build_rows_h(h)
    yy = np.array([r["y"] for r in rr], dtype=int)
    vv = np.array([r["bvid"] for r in rr], dtype=str)
    ff = M.group_folds(vv, folds=M.CV_FOLDS, seed=M.CV_SEED)
    fid = np.array([ff[v] for v in vv], dtype=int)
    x1 = M.select_features(rr, M.FEATURES_M1)
    x6 = M.select_features(rr, M.FEATURES_M2)
    ll = np.array([float(r["log1p_lam"]) for r in rr], dtype=float)
    out = []
    for name, X, prob, score in (
        ("NegLam", None, False, -ll),
        ("M1_LR_1feat", x1, True, None),
        ("LR_6feat", x6, True, None),
    ):
        if score is not None:
            oof = score
            rt = 0.0
        else:
            t0 = time.time()
            oof = np.full(len(yy), np.nan)
            for f in range(M.CV_FOLDS):
                te = fid == f
                tr = ~te
                est = scaled(LogisticRegression(C=1.0, max_iter=1000,
                                                solver="lbfgs", random_state=SEED))
                est.fit(X[tr], yy[tr])
                oof[te] = est.predict_proba(X[te])[:, 1]
            rt = round(time.time() - t0, 2)
        mm = M.macro_within_video_auc(yy, oof, vv)
        per = np.asarray(list(mm["per_video"].values()), dtype=float)
        rec = {"task": f"window_H{h}", "model": name, "status": "ok",
               "n_points": int(len(yy)), "n_videos": int(mm["n_eligible"]),
               "macro_auc": float(mm["macro"]),
               "auc_p10": float(np.percentile(per, 10)),
               "auc_p50": float(np.percentile(per, 50)),
               "auc_p90": float(np.percentile(per, 90)),
               "pr_auc": float(average_precision_score(yy, oof)),
               "brier": float(brier_score_loss(yy, oof)) if prob else None,
               "runtime_s": rt}
        if name != "NegLam":
            b = M.paired_bootstrap_delta(yy, oof, -ll, vv, B=BOOT_B, seed=BOOT_SEED)
            rec.update({"delta_vs_neg": float(b["delta"]), "neg_lo": float(b["ci_lo"]),
                        "neg_hi": float(b["ci_hi"])})
        out.append(rec)
    return out


windows = [window_stats(h) for h in (2, 3, 6)]
window_runs = []
for w in windows:
    if w["h_buckets"] in (2, 3) and w["gate_pass"]:
        window_runs.extend(core_for_h(w["h_buckets"]))

# ── direction 4: expD_count status ───────────────────────────────────
expd_path = os.path.join(REPO, "experiments", "expD_count", "p0_evidence.json")
expd = {"status": "blocked_no_fit"}
if os.path.exists(expd_path):
    d = json.load(open(expd_path, encoding="utf-8"))
    expd.update({"p0_passed": d.get("passed"), "decision": d.get("decision"),
                 "path": expd_path,
                 "reason": "P0 windows empty in both D-train meta files"})

out = {
    "meta": {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "D-train only (20260902/20260907); no D-dev/D-conf",
        "sample": {"n_points": int(y.size), "n_videos": int(len(set(vid))),
                   "pos": int(y.sum()), "rate": float(y.mean())},
        "folds": {"scheme": "bvid GroupKFold", "k": M.CV_FOLDS,
                  "seed": M.CV_SEED},
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "bvid"},
        "existing_oof": EXIST,
        "protocol": os.path.join(OUT, "protocol.md"),
    },
    "direction1": direction1,
    "direction2": dir2,
    "direction3": {"windows": windows, "gated_core_runs": window_runs},
    "direction4": {"ablation": ablation, "expD_count": expd},
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=float)

# csv
def write_csv(path, rows):
    if not rows:
        open(path, "w", encoding="utf-8").write("")
        return
    keys = ["task", "model", "n_points", "n_videos", "macro_auc", "pr_auc", "brier",
            "delta_vs_neg", "neg_lo", "neg_hi", "delta_vs_M1", "M1_lo", "M1_hi",
            "delta_vs_M2", "M2_lo", "M2_hi", "n_folds_gt_half", "fold_min", "fold_max",
            "auc_p10", "auc_p50", "auc_p90", "runtime_s", "note"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

write_csv(os.path.join(OUT, "metrics.csv"), dir2 + ablation + window_runs)
write_csv(os.path.join(OUT, "time_windows.csv"), windows)

print(json.dumps({
    "sample": out["meta"]["sample"],
    "direction1": {"m1_fold0": direction1["m1"]["fold0_auc"],
                   "m1_corr_folds": corr_folds,
                   "m1_macro": direction1["m1"]["macro_auc"],
                   "neg_macro": direction1["neg_lam"]["macro_auc"],
                   "m2_macro": direction1["m2"]["macro_auc"],
                   "m2_minus_m1": direction1["deltas"]["m2_minus_m1"],
                   "m2_minus_neg": direction1["deltas"]["m2_minus_neg"]},
    "direction2": [{k: r.get(k) for k in ("model", "macro_auc", "delta_vs_neg",
                                         "neg_lo", "neg_hi", "n_folds_gt_half")}
                   for r in dir2],
    "windows": windows,
    "ablation": [{k: r.get(k) for k in ("model", "macro_auc", "delta_vs_M1",
                                       "M1_lo", "M1_hi", "delta_vs_M2")}
                 for r in ablation],
    "expD": expd,
}, ensure_ascii=False, indent=2, default=float))
