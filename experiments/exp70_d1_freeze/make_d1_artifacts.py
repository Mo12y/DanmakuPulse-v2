# -*- coding: utf-8 -*-
"""exp70 make_d1_artifacts.py — save B1/OLS6 coefficients and 395-row predictions from the existing exp55 flow.

No new model, no hyperparameter change, no split change. This is the same B1/OLS6 fit that exp55
already performs; we only persist its coefficients and row-level outputs with (bvid,batch) keys.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp70_d1_freeze"
EXP67 = f"{V2}/experiments/exp67_evidence_audit"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{V2}/experiments/exp_ml_scan_20261027")
import run_scan as rs  # noqa


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def met(pred_log, y_log):
    pred_log = np.clip(np.asarray(pred_log, float), 0, 20)
    true = np.expm1(np.asarray(y_log, float))
    pred = np.expm1(pred_log)
    return dict(R2=round(float(r2_score(y_log, pred_log)), 4),
                MAE_orig=round(float(mean_absolute_error(true, pred)), 2),
                RMSE_orig=round(float(np.sqrt(mean_squared_error(true, pred))), 2), n=int(len(y_log)))


# ---- rebuild exp55 loop WITH batch key (same grouping/filters) ----
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
    rows.append((bv, int(bt), g["first_seen"].iloc[0], n01, n13, n36, c0,
                 ((rel > 6) & (rel <= 30)).sum(), dur))
u = pd.DataFrame(rows, columns=["bvid", "batch", "fs", "n01", "n13", "n36", "c0", "y", "dur"])
tr, te = u[u.fs <= 20261020].copy(), u[u.fs >= 20261021].copy()
assert len(u) == 1313 and len(tr) == 918 and len(te) == 395, (len(u), len(tr), len(te))
assert len(set(tr.bvid) & set(te.bvid)) == 0
assert not te.duplicated(["bvid", "batch"]).any()

# compare keys to exp67 canonical table
u67 = pd.read_csv(f"{EXP67}/d1_units.csv")
te67 = u67[u67.fs >= 20261021]
keys = set(zip(te.bvid, te.batch)); keys67 = set(zip(te67.bvid, te67.batch))
assert keys == keys67, (len(keys), len(keys67))
assert set(te.bvid) == set(te67.bvid)
assert te.bvid.duplicated().sum() == te67.bvid.duplicated().sum() == 41

# ---- same B1/OLS6 fits as exp55 ----
def x1(df):
    return np.log1p(df.c0.values).reshape(-1, 1)


def x6(df):
    return np.column_stack([np.log1p(df.c0), np.log1p(df.n36), np.log1p(df.n13), np.log1p(df.n01),
                            np.log1p(df.n36) - np.log1p(df.n13), np.log1p(df.dur)])


ytr, yte = np.log1p(tr.y.values), np.log1p(te.y.values)
b1 = LinearRegression().fit(x1(tr), ytr)
m6 = LinearRegression().fit(x6(tr), ytr)
p1 = b1.predict(x1(te)); p6 = m6.predict(x6(te))
pers = 4.0 * te.c0.values
metrics = dict(persistence_4xc0=met(np.log1p(pers), yte), B1=met(p1, yte), OLS6=met(p6, yte))
assert abs(metrics["persistence_4xc0"]["MAE_orig"] - 814.24) < 0.02
assert abs(metrics["B1"]["MAE_orig"] - 277.69) < 0.02
assert abs(metrics["OLS6"]["MAE_orig"] - 238.39) < 0.02

# ---- save coefficients ----
b1_coef = dict(model="B1 (OLS, single feature log1p(c0))", feature=["c0"],
               coef=[float(b1.coef_[0])], intercept=float(b1.intercept_),
               trained_on="exp55 train: first_seen<=20261020, 918 rows / 649 bvid",
               target="log1p(count(6h<rel<=30h))", n_train=int(len(tr)),
               note="saved from the existing exp55 B1 fit; no new model/hyperparameter")
json.dump(b1_coef, open(f"{OUT}/b1_coef.json", "w"), ensure_ascii=False, indent=1)

ols6_coef = dict(model="OLS-6 (evaluation model; train-only)", features=["c0", "n36", "n13", "n01", "slope", "dur"],
                 coef=[float(x) for x in m6.coef_], intercept=float(m6.intercept_),
                 trained_on="exp55 train: first_seen<=20261020, 918 rows / 649 bvid",
                 target="log1p(count(6h<rel<=30h))", n_train=int(len(tr)),
                 note="evaluation model; differs from production d1_ols.json full-window refit")
json.dump(ols6_coef, open(f"{OUT}/ols6_eval_coef.json", "w"), ensure_ascii=False, indent=1)

# ---- save row-level outputs ----
base = te[["bvid", "batch", "fs", "c0", "y", "dur"]].copy()
base["y_log"] = yte
out = base.copy()
out["b1_pred_log"] = p1; out["b1_pred_orig"] = np.expm1(p1); out["b1_resid_log"] = yte - p1
out["ols6_pred_log"] = p6; out["ols6_pred_orig"] = np.expm1(p6); out["ols6_resid_log"] = yte - p6
out["persistence_4xc0"] = pers; out["persistence_resid_log"] = yte - np.log1p(pers)
out.to_csv(f"{OUT}/d1_test_predictions_395.csv", index=False, encoding="utf-8-sig")
te[["bvid", "batch", "fs", "n01", "n13", "n36", "c0", "y", "dur"]].to_csv(
    f"{OUT}/exp55_test_rowkeys_395.csv", index=False, encoding="utf-8-sig")

manifest = dict(
    created_at="2026-10-28",
    source_flow="experiments/exp55_route_abc_verify.py Part A (same loop, batch key retained)",
    repo_head="500c7c7",
    n_total=int(len(u)), n_train=int(len(tr)), n_test=int(len(te)),
    n_train_bvid=int(tr.bvid.nunique()), n_test_bvid=int(te.bvid.nunique()),
    test_dup_bvid_rows=int(te.bvid.duplicated().sum()),
    bvid_overlap=int(len(set(tr.bvid) & set(te.bvid))),
    keys_equal_exp67=bool(keys == keys67),
    metrics=metrics, metrics_expected=dict(persistence_MAE=814.2, B1_MAE=277.7, OLS6_MAE=238.4),
    b1_coef=b1_coef, ols6_eval_coef=ols6_coef,
    files={n: dict(path=f"{OUT}/{n}", sha256=sha(f"{OUT}/{n}"))
           for n in ["b1_coef.json", "ols6_eval_coef.json", "d1_test_predictions_395.csv",
                     "exp55_test_rowkeys_395.csv"]},
    note="No model/split/hyperparameter change. B1/OLS6 are the same fits exp55 already performs; only persisted.",
)
json.dump(manifest, open(f"{OUT}/artifact_manifest.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(dict(metrics=metrics, keys_equal_exp67=keys == keys67,
                      b1_coef=b1_coef["coef"][0], files=manifest["files"]), ensure_ascii=False, indent=1))
print("EXP70-ARTIFACTS-DONE")
