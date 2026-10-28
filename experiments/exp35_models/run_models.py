#!/usr/bin/env python3
"""exp35 多模型探索（隔离实验，只读复用冻结流水线）。

复用 `jobs/m2_l6_burst_risk.py` 的：数据读取、标签、特征、资格、按 bvid 分组折、
视频内宏平均 AUC、配对视频级 bootstrap。**不改动该文件**；本目录只做探索。

三个梯度提升库安装在 `~/exp-libs`（--no-deps --target），**不修改冻结 venv**：
  PYTHONPATH=$HOME/exp-libs ~/danmaku-venv/bin/python experiments/exp35_models/run_models.py --round 1

输出：data-out/exp35_models/
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "jobs"))
import m2_l6_burst_risk as M  # noqa: E402

from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,  # noqa: E402
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,  # noqa: E402
                             roc_auc_score)
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

MANIFEST = os.path.join(REPO, "docs", "exp35_508", "file_manifest.csv")
DATA_ROOT = "/home/molly/projects/danmaku"
OUT = os.path.join(REPO, "data-out", "exp35_models")
SEED = 42

# ── 可选梯度提升库（装在 ~/exp-libs，失败不影响其它模型）─────────────
LIBS = {}


def _try(mod):
    try:
        m = __import__(mod)
        return m
    except Exception:
        return None


XGB = _try("xgboost")
LGBM = _try("lightgbm")
CAT = _try("catboost")
for name, mod in (("xgboost", XGB), ("lightgbm", LGBM), ("catboost", CAT)):
    LIBS[name] = getattr(mod, "__version__", None) if mod else None


def load_dtrain():
    seqs, _ = M.load_sequences(MANIFEST, DATA_ROOT, batches=M.D_TRAIN_BATCHES, layer="ods")
    rows = M.filter_multip(M.build_rows(seqs), "keep")
    elig = set(M.eligible_video_ids(rows))
    rows = [r for r in rows if r["bvid"] in elig]
    y = np.array([r["y"] for r in rows], dtype=int)
    vid = np.array([r["bvid"] for r in rows], dtype=str)
    X6 = M.select_features(rows, M.FEATURES_M2)
    X1 = M.select_features(rows, M.model_features("M1"))
    fold_of = M.group_folds(vid, folds=M.CV_FOLDS, seed=M.CV_SEED)
    fold_id = np.array([fold_of[v] for v in vid], dtype=int)
    return dict(X1=X1, X6=X6, y=y, vid=vid, fold_id=fold_id,
                n=int(y.size), n_vid=int(np.unique(vid).size),
                pos=int(y.sum()), rate=float(y.mean()))


def _scaled(est):
    return Pipeline([("sc", StandardScaler()), ("clf", est)])


def _xgb(**kw):
    p = dict(n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
             colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist",
             n_jobs=-1, random_state=SEED, eval_metric="logloss")
    p.update(kw)
    return XGB.XGBClassifier(**p)


def _lgbm(**kw):
    p = dict(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.9,
             colsample_bytree=0.9, n_jobs=-1, random_state=SEED, verbose=-1)
    p.update(kw)
    return LGBM.LGBMClassifier(**p)


def _cat(**kw):
    p = dict(iterations=300, depth=6, learning_rate=0.08, l2_leaf_reg=3.0,
             random_seed=SEED, verbose=0, allow_writing_files=False, thread_count=-1)
    p.update(kw)
    return CAT.CatBoostClassifier(**p)


def models_round1():
    specs = [
        ("M1_LR_1feat", "log1p_lam", lambda: _scaled(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))),
        ("LR_6feat", "6feat", lambda: _scaled(
            LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))),
        ("RF", "6feat", lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        ("ExtraTrees", "6feat", lambda: ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        ("HistGB", "6feat", lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            early_stopping=False, random_state=SEED)),
        ("SVM_RBF", "6feat", lambda: _scaled(
            SVC(C=1.0, gamma="scale", probability=True, random_state=SEED))),
        ("MLP", "6feat", lambda: _scaled(
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                          early_stopping=True, validation_fraction=0.1, random_state=SEED))),
    ]
    if XGB is not None:
        specs.append(("XGBoost", "6feat", lambda: _xgb()))
    if LGBM is not None:
        specs.append(("LightGBM", "6feat", lambda: _lgbm()))
    if CAT is not None:
        specs.append(("CatBoost", "6feat", lambda: _cat()))
    return specs


def models_round2(top):
    out = [("M1_LR_1feat", "log1p_lam", lambda: _scaled(
        LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")))]
    if "LR_6feat" in top:
        for C in (0.1, 10.0):
            out.append((f"LR_C{C}", "6feat", lambda C=C: _scaled(
                LogisticRegression(C=C, max_iter=1000, solver="lbfgs"))))
    if "RF" in top:
        for leaf in (2, 10):
            out.append((f"RF_leaf{leaf}", "6feat", lambda leaf=leaf: RandomForestClassifier(
                n_estimators=300, min_samples_leaf=leaf, n_jobs=-1, random_state=SEED)))
    if "ExtraTrees" in top:
        for leaf in (2, 10):
            out.append((f"ET_leaf{leaf}", "6feat", lambda leaf=leaf: ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=leaf, n_jobs=-1, random_state=SEED)))
    if "HistGB" in top:
        for lr_, leaves in ((0.05, 31), (0.15, 15)):
            out.append((f"HGB_lr{lr_}_leaf{leaves}", "6feat",
                        lambda lr_=lr_, leaves=leaves: HistGradientBoostingClassifier(
                            max_iter=300, learning_rate=lr_, max_leaf_nodes=leaves,
                            early_stopping=False, random_state=SEED)))
    if "SVM_RBF" in top:
        for C in (0.3, 3.0):
            out.append((f"SVM_C{C}", "6feat", lambda C=C: _scaled(
                SVC(C=C, gamma="scale", probability=True, random_state=SEED))))
    if "MLP" in top:
        out.append(("MLP_128", "6feat", lambda: _scaled(
            MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                          early_stopping=True, validation_fraction=0.1, random_state=SEED))))
    if "XGBoost" in top and XGB is not None:
        for depth in (3, 6):
            out.append((f"XGB_d{depth}", "6feat", lambda depth=depth: _xgb(max_depth=depth)))
        out.append(("XGB_lr0.03", "6feat", lambda: _xgb(learning_rate=0.03, n_estimators=600)))
    if "LightGBM" in top and LGBM is not None:
        for leaves in (15, 63):
            out.append((f"LGBM_leaves{leaves}", "6feat",
                        lambda leaves=leaves: _lgbm(num_leaves=leaves)))
        out.append(("LGBM_lr0.03", "6feat", lambda: _lgbm(learning_rate=0.03, n_estimators=600)))
    if "CatBoost" in top and CAT is not None:
        for depth in (4, 8):
            out.append((f"CAT_d{depth}", "6feat", lambda depth=depth: _cat(depth=depth)))
    return out


def run_one(name, kind, factory, d):
    X = d["X1"] if kind == "log1p_lam" else d["X6"]
    y, vid, fold_id = d["y"], d["vid"], d["fold_id"]
    oof = np.full(y.size, np.nan)
    fit_s = pred_s = 0.0
    warns = []
    t0 = time.time()
    try:
        for f in range(M.CV_FOLDS):
            te = fold_id == f
            tr = ~te
            est = factory()
            a = time.time()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                est.fit(X[tr], y[tr])
            fit_s += time.time() - a
            for x in w[:1]:
                warns.append(str(x.message)[:70])
            a = time.time()
            oof[te] = est.predict_proba(X[te])[:, 1]
            pred_s += time.time() - a
    except Exception as exc:
        return dict(model=name, features=kind, status="failed",
                    error=f"{type(exc).__name__}: {exc}"[:200]), None
    if not np.isfinite(oof).all():
        return dict(model=name, features=kind, status="failed", error="OOF 不完整"), None

    m = M.macro_within_video_auc(y, oof, vid)
    per = np.array(list(m["per_video"].values()), dtype=float)
    rec = dict(model=name, features=kind, status="ok",
               macro_auc=float(m["macro"]), n_eligible=int(m["n_eligible"]),
               auc_p10=float(np.percentile(per, 10)),
               auc_p50=float(np.percentile(per, 50)),
               auc_p90=float(np.percentile(per, 90)),
               auc_sd=float(per.std(ddof=1)),
               share_auc_gt_half=float((per > 0.5).mean()),
               pooled_auc=float(roc_auc_score(y, oof)),
               pr_auc=float(average_precision_score(y, oof)),
               brier=float(brier_score_loss(y, oof)),
               logloss=float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
               fit_s=round(fit_s, 2), pred_s=round(pred_s, 2),
               total_s=round(time.time() - t0, 2), n_warn=len(warns),
               warn=sorted(set(warns))[:2])
    return rec, oof


def add_delta(rec, oof, ref_oof, y, vid):
    if ref_oof is None or rec["model"] == "M1_LR_1feat":
        return rec
    try:
        b = M.paired_bootstrap_delta(y, oof, ref_oof, vid, B=M.BOOT_B, seed=M.BOOT_SEED)
        rec.update(delta_vs_M1=b["delta"], delta_lo=b["ci_lo"], delta_hi=b["ci_hi"],
                   delta_reps=b["n_valid_reps"], delta_discarded=b["n_discarded"])
    except Exception as exc:
        rec.update(delta_error=f"{type(exc).__name__}: {exc}"[:120])
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--top", default="LR_6feat,RF,ExtraTrees,HistGB,SVM_RBF,MLP,XGBoost,LightGBM,CatBoost")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    d = load_dtrain()
    print(f"[data] points={d['n']} videos={d['n_vid']} pos={d['pos']} rate={d['rate']:.4f} "
          f"folds={M.CV_FOLDS} seed={M.CV_SEED}")
    print(f"[libs] {LIBS}")

    if args.round == 1:
        specs, tag = models_round1(), "round1"
    else:
        specs, tag = models_round2({s.strip() for s in args.top.split(",") if s.strip()}), "round2"

    ref_oof = None
    results = []
    ordered = sorted(specs, key=lambda s: 0 if s[0] == "M1_LR_1feat" else 1)
    for name, kind, fac in ordered:
        rec, oof = run_one(name, kind, fac, d)
        if name == "M1_LR_1feat" and oof is not None:
            ref_oof = oof
        rec = add_delta(rec, oof, ref_oof, d["y"], d["vid"])
        if oof is not None:
            np.save(os.path.join(OUT, f"oof_{tag}_{name}.npy"), oof)
        results.append(rec)
        if rec["status"] == "ok":
            print(f"  {name:20s} macro={rec['macro_auc']:.4f} pooled={rec['pooled_auc']:.4f} "
                  f"pr={rec['pr_auc']:.4f} brier={rec['brier']:.4f} "
                  f"d={rec.get('delta_vs_M1', float('nan')):+.4f} {rec['total_s']}s")
        else:
            print(f"  {name:20s} FAILED: {rec['error']}")

    payload = dict(protocol_tag="exp35-model-exploration", round=tag, libs=LIBS,
                   data=dict(points=d["n"], videos=d["n_vid"], pos=d["pos"],
                             rate=d["rate"], batches=list(M.D_TRAIN_BATCHES),
                             split="D-train only (no dev/confirm used)",
                             folds=M.CV_FOLDS, fold_seed=M.CV_SEED,
                             bootstrap_B=M.BOOT_B, bootstrap_seed=M.BOOT_SEED),
                   results=results)
    with open(os.path.join(OUT, f"results_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    cols = ["model", "features", "status", "macro_auc", "auc_p10", "auc_p50", "auc_p90",
            "auc_sd", "share_auc_gt_half", "pooled_auc", "pr_auc", "brier", "logloss",
            "delta_vs_M1", "delta_lo", "delta_hi", "total_s", "n_warn"]
    with open(os.path.join(OUT, f"results_{tag}.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: -r.get("macro_auc", -9)))
    print("saved:", os.path.join(OUT, f"results_{tag}.json"))


if __name__ == "__main__":
    main()
