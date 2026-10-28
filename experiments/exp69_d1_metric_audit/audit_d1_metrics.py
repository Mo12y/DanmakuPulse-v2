# -*- coding: utf-8 -*-
"""exp69 audit_d1_metrics.py — read-only D1 metric ledger checks.

No model is saved or replaced. B1/OLS-6 are re-fitted only as verification refits
(same train rows, same features, same target) to check the published MAE values.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp69_d1_metric_audit"
os.makedirs(OUT, exist_ok=True)
EXP67 = f"{V2}/experiments/exp67_evidence_audit"
EXP68 = f"{V2}/experiments/exp68_d1_residual_bridge"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def met(pred_log, y_log):
    pred_log = np.clip(np.asarray(pred_log, float), 0, 20)
    true = np.expm1(np.asarray(y_log, float))
    pred = np.expm1(pred_log)
    return dict(R2=round(float(r2_score(y_log, pred_log)), 4),
                MAE_orig=round(float(mean_absolute_error(true, pred)), 2),
                RMSE_orig=round(float(np.sqrt(mean_squared_error(true, pred))), 2),
                n=int(len(y_log)))


u = pd.read_csv(f"{EXP67}/d1_units.csv")
tr, te = u[u.fs <= 20261020].copy(), u[u.fs >= 20261021].copy()
ytr, yte = np.log1p(tr.y.values), np.log1p(te.y.values)


def x6(df):
    return np.column_stack([np.log1p(df.c0), np.log1p(df.n36), np.log1p(df.n13), np.log1p(df.n01),
                            np.log1p(df.n36) - np.log1p(df.n13), np.log1p(df.dur)])


m6 = LinearRegression().fit(x6(tr), ytr)
b1 = LinearRegression().fit(np.log1p(tr.c0.values).reshape(-1, 1), ytr)
p6, p1 = m6.predict(x6(te)), b1.predict(np.log1p(te.c0.values).reshape(-1, 1))
pp = 4.0 * te.c0.values

res = dict(
    files={p: sha(p) for p in [
        f"{EXP67}/d1_units.csv", f"{V2}/experiments/exp55_route_abc_verify/results.json",
        f"{V2}/experiments/exp60_finalize/r6_final.json", f"{EXP67}/d1_audit.json",
        f"{EXP68}/d1_repro.json", f"{EXP68}/results.json",
        f"{V2}/experiments/exp38_integrate/models/d1_ols.json"]},
    sample=dict(rows_total=int(len(u)), videos_total=int(u.bvid.nunique()),
                train_rows=int(len(tr)), train_videos=int(tr.bvid.nunique()),
                test_rows=int(len(te)), test_videos=int(te.bvid.nunique()),
                bvid_overlap=int(len(set(tr.bvid) & set(te.bvid))),
                test_dup_bvid_rows=int(te.bvid.duplicated().sum()),
                test_dup_key_rows=int(te.duplicated(["bvid", "batch"]).sum()),
                missing=({c: int(u[c].isna().sum()) for c in ["bvid", "batch", "fs", "n01", "n13", "n36", "c0", "y", "dur", "age"]}),
                age_test=dict(min=float(te.age.min()), q05=float(te.age.quantile(.05)),
                              median=float(te.age.median()), q95=float(te.age.quantile(.95)), max=float(te.age.max()))),
    metrics=dict(persistence_4xc0=met(np.log1p(pp), yte), B1=met(p1, yte), OLS6=met(p6, yte)),
    b1_coef=dict(coef=float(b1.coef_[0]), intercept=float(b1.intercept_)),
    ols6_coef=dict(coef=[float(x) for x in m6.coef_], intercept=float(m6.intercept_)),
)

# production artifact vs train-only evaluation model
prod = json.load(open(f"{V2}/experiments/exp38_integrate/models/d1_ols.json"))
pred_prod_log = x6(te) @ np.array(prod["coef"]) + prod["intercept"]
res["production_vs_eval"] = dict(
    max_abs_coef_diff=float(np.max(np.abs(np.array(prod["coef"]) - m6.coef_))),
    max_abs_logpred_diff=float(np.max(np.abs(pred_prod_log - p6))),
    production_on_test=met(pred_prod_log, yte),
    eval_on_test=met(p6, yte),
    production_features=prod["features"], production_name=prod["model"],
)

# ratio-calibrated persistence
r = float(np.median(tr.y / tr.c0))
res["ratio_baseline"] = dict(train_median_y_over_c0=round(r, 4), **met(np.log1p(r * te.c0.values), yte))

# row-level vs video-level sensitivity (latest snapshot per bvid)
te_d = te.sort_values("age").drop_duplicates("bvid", keep="last")
res["dedup_test"] = dict(
    n=int(len(te_d)), n_videos=int(te_d.bvid.nunique()),
    persistence=met(np.log1p(4.0 * te_d.c0.values), np.log1p(te_d.y.values)),
    B1=met(b1.predict(np.log1p(te_d.c0.values).reshape(-1, 1)), np.log1p(te_d.y.values)),
    OLS6=met(m6.predict(x6(te_d)), np.log1p(te_d.y.values)),
)

# c0 drift across repeated snapshots of the same bvid
drift = []
for bv, g in u.groupby("bvid"):
    if len(g) < 2:
        continue
    g = g.sort_values("age")
    drift.append((float(g.iloc[-1].c0) - float(g.iloc[0].c0)) / max(float(g.iloc[0].c0), 1))
res["c0_drift"] = dict(n_bvids=len(drift), median=round(float(np.median(drift)), 4),
                       p05=round(float(np.percentile(drift, 5)), 4),
                       p95=round(float(np.percentile(drift, 95)), 4),
                       share_abs_gt_1pct=round(float(np.mean(np.abs(drift) > 0.01)), 3))

# exp68 row matching and prediction version
d = np.load(f"{EXP68}/bridge_data.npz", allow_pickle=True)
meta = pd.read_csv(f"{EXP68}/bridge_test_meta.csv", encoding="utf-8-sig")
feat = pd.read_csv(f"{EXP68}/features.csv", encoding="utf-8-sig")
fe = feat[feat.split == "test"].copy()
key_te = set(zip(te.bvid, te.batch))
key_meta = set(zip(meta.bvid, meta.batch))
key_fe = set(zip(fe.bvid, fe.batch))
idx_te = te.set_index(["bvid", "batch"])
idx_meta = meta.set_index(["bvid", "batch"])
common = list(idx_meta.index)
res["exp68_match"] = dict(
    n_pred=int(len(d["yte_log"])), n_meta=int(len(meta)), n_features_test=int(len(fe)),
    key_sets_equal=bool(key_te == key_meta == key_fe),
    key_meta_minus_te=len(key_meta - key_te), key_te_minus_meta=len(key_te - key_meta),
    bvid_overlap_train_test=int(len(set(d["bvid_tr"]) & set(d["bvid_te"]))),
    max_abs_y_diff=float(np.max(np.abs(d["yte_log"] - np.log1p(meta.y.values)))),
    max_abs_pred_vs_eval=0.0,
    pred_te_r2=round(float(r2_score(d["yte_log"], d["pred_te"])), 4),
    oof_zeros=int((d["pred_oof"] == 0).sum()),
    oof_r2=round(float(r2_score(d["ytr_log"], d["pred_oof"])), 4),
    full_r2=round(float(r2_score(ytr, m6.predict(x6(tr)))), 4),
    max_abs_oof_vs_full=float(np.max(np.abs(d["pred_oof"] - m6.predict(x6(tr))))),
    n_train_pred=int(len(d["ytr_log"])),
    Xte0_vs_pred=float(np.max(np.abs(d["Xte"][:, 0] - d["pred_te"]))),
    Xtr0_vs_oof=float(np.max(np.abs(d["Xtr"][:, 0] - d["pred_oof"]))),
)
ug = tr.bvid.drop_duplicates().values.copy()
rng = np.random.default_rng(20261028); rng.shuffle(ug)
fold_of = {b: i % 5 for i, b in enumerate(ug)}
fold = tr.bvid.map(fold_of).values
res["exp68_match"]["oof_fold_counts"] = {str(k): int((fold == k).sum()) for k in range(5)}
res["exp68_match"]["oof_nan_folds"] = int(pd.isna(fold).sum())
res["exp68_match"]["oof_unique_bvids"] = int(len(set(ug)))
# align pred_te to te order and compare with verification p6
map_te = idx_te.loc[common]
p6_series = pd.Series(p6, index=pd.MultiIndex.from_arrays([te.bvid.values, te.batch.values]))
p6_aligned = p6_series.loc[common].values
res["exp68_match"]["max_abs_pred_vs_eval"] = float(np.max(np.abs(d["pred_te"] - p6_aligned)))
res["exp68_match"]["pred_vs_prod_maxdiff"] = float(np.max(np.abs(d["pred_te"] - pred_prod_log)))

# exp68 result headline
r68 = json.load(open(f"{EXP68}/results.json"))
res["exp68_result"] = dict(primary=r68.get("primary"), controls=r68.get("controls"),
                           ablation=r68.get("ablation", {}).get("structure_beyond_static"),
                           prereg_verdict=r68.get("prereg_verdict"),
                           metrics=r68.get("metrics"), delta=r68.get("delta"))

json.dump(res, open(f"{OUT}/d1_metric_audit.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:4000])
print("EXP69-AUDIT-DONE")
