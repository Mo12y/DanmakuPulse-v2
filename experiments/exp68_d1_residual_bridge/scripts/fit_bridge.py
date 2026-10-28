# -*- coding: utf-8 -*-
"""exp68 fit_bridge.py — reproduce D1, build OOF residuals, fit pre-registered ridge bridge.

Outputs: d1_repro.json, results.json, bridge_data.npz, bridge_train.csv, bridge_test.csv.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp68_d1_residual_bridge"
SEED = 20261028
ALPHA = 1.0
N_FOLDS = 5

G2 = ["entropy_norm", "burst_cv", "peak_pos", "peak_share", "zero_bin_share", "gini_bins"]
G3 = ["mean_text_len", "long_share", "colored_share"]
G4 = ["user_cov", "user_hhi", "user_top1_share", "user_unique_share"]
G5 = ["multi_user_second_share", "repeat_user_row_share", "progress_gini", "second_top1_share"]
STATIC = ["log_dur", "part_count_bin", "pub_hour_sin", "pub_hour_cos", "pub_dow_sin", "pub_dow_cos"]


def d1_design(df):
    return np.column_stack([np.log1p(df.c0), np.log1p(df.n36), np.log1p(df.n13), np.log1p(df.n01),
                            np.log1p(df.n36) - np.log1p(df.n13), np.log1p(df.dur)])


def b1_design(df):
    return np.log1p(df.c0.values).reshape(-1, 1)


def metrics(pred_log, y_log):
    pred_log = np.clip(np.asarray(pred_log, float), 0, 20)
    true = np.expm1(np.asarray(y_log, float))
    pred = np.expm1(pred_log)
    return dict(R2=round(float(r2_score(y_log, pred_log)), 4),
                MAE_orig=round(float(mean_absolute_error(true, pred)), 2),
                RMSE_orig=round(float(np.sqrt(mean_squared_error(true, pred))), 2))


# ---------- data ----------
feat = pd.read_csv(f"{OUT}/features.csv", encoding="utf-8-sig")
units = pd.read_csv(f"{V2}/experiments/exp67_evidence_audit/d1_units.csv")[["bvid", "batch", "n01", "n13", "n36"]]
df = feat.merge(units, on=["bvid", "batch"], validate="one_to_one")
assert len(df) == 1313
tr, te = df[df.split == "train"].copy(), df[df.split == "test"].copy()
assert len(tr) == 918 and len(te) == 395, (len(tr), len(te))
assert len(set(tr.bvid) & set(te.bvid)) == 0

# tname dummies: top7 fit on train only
top = tr.tname.value_counts().head(7).index.tolist()
for i, t in enumerate(top):
    tr[f"tname_{i}"] = (tr.tname == t).astype(int)
    te[f"tname_{i}"] = (te.tname == t).astype(int)
DUMMIES = [f"tname_{i}" for i in range(len(top))]

# zero-variance drop on train
cand = STATIC + DUMMIES + G2 + G3 + G4 + G5
drop = [c for c in cand if float(tr[c].std()) == 0]
keep = [c for c in cand if c not in drop]
G1 = [c for c in STATIC + DUMMIES if c in keep]
groups = {"G1": G1, "G2": [c for c in G2 if c in keep], "G3": [c for c in G3 if c in keep],
          "G4": [c for c in G4 if c in keep], "G5": [c for c in G5 if c in keep]}
feature_names = ["pred_log"] + sum(groups.values(), [])
group_of = ["G0"] + sum([[g] * len(v) for g, v in groups.items()], [])

# ---------- reproduce D1 ----------
Xtr6, Xte6 = d1_design(tr), d1_design(te)
ytr_log, yte_log = np.log1p(tr.y.values), np.log1p(te.y.values)
m6 = LinearRegression().fit(Xtr6, ytr_log)
b1 = LinearRegression().fit(b1_design(tr), ytr_log)
pred_te = m6.predict(Xte6)
pred_te_b1 = b1.predict(b1_design(te))

# OOF D1 predictions (shuffled GroupKFold by bvid)
rng = np.random.default_rng(SEED)
ug = tr.bvid.drop_duplicates().values.copy()
rng.shuffle(ug)
fold_of = {b: i % N_FOLDS for i, b in enumerate(ug)}
fold = tr.bvid.map(fold_of).values
pred_oof = np.zeros(len(tr))
for k in range(N_FOLDS):
    va = fold == k
    mm = LinearRegression().fit(Xtr6[~va], ytr_log[~va])
    pred_oof[va] = mm.predict(Xtr6[va])

# production artifact check
prod = json.load(open(f"{V2}/experiments/exp38_integrate/models/d1_ols.json"))
full = LinearRegression().fit(d1_design(df), np.log1p(df.y.values))
maxdiff = float(np.max(np.abs(np.array(prod["coef"]) - full.coef_)))

snap = json.load(open(f"{V2}/data-out/ml_forecast/ml_snapshot.json"))
ages_ok = [x["age_h"] for x in snap["items"] if x["d1"]["status"] == "ok"]
ages_pending = [x["age_h"] for x in snap["items"] if x["d1"]["status"] == "pending"]
d1_repro = dict(
    n_train=int(len(tr)), n_test=int(len(te)), bvid_overlap=len(set(tr.bvid) & set(te.bvid)),
    OLS6=metrics(pred_te, yte_log), B1=metrics(pred_te_b1, yte_log),
    OOF_R2=round(float(r2_score(ytr_log, pred_oof)), 4),
    expected=dict(R2=0.5607, MAE=238.4, B1_MAE=277.7),
    reproduction_ok=dict(R2=abs(r2_score(yte_log, pred_te) - 0.5607) < 5e-4,
                         MAE=abs(mean_absolute_error(np.expm1(yte_log), np.expm1(np.clip(pred_te, 0, 20))) - 238.4) < 0.2,
                         B1_MAE=abs(mean_absolute_error(np.expm1(yte_log), np.expm1(np.clip(pred_te_b1, 0, 20))) - 277.7) < 0.2),
    production=dict(max_coef_diff_vs_full_refit=maxdiff, features=prod["features"],
                    snapshot_ok=len(ages_ok), snapshot_pending=len(ages_pending),
                    age_ok_median=float(np.median(ages_ok)), age_pending_median=float(np.median(ages_pending))),
    age_train=dict(q05=float(tr.age.quantile(.05)), median=float(tr.age.median()), q95=float(tr.age.quantile(.95))),
    age_test=dict(q05=float(te.age.quantile(.05)), median=float(te.age.median()), q95=float(te.age.quantile(.95))),
    feature_missing=dict(train=int(tr[keep].isna().sum().sum()), test=int(te[keep].isna().sum().sum())),
    n6_lt2_test=int((te.n6 < 2).sum()),
    zero_variance_dropped=drop,
    split_rule="first_seen<=20261020 train / >=20261021 test; bvid disjoint",
)
json.dump(d1_repro, open(f"{OUT}/d1_repro.json", "w"), ensure_ascii=False, indent=1)
print("[D1]", json.dumps(d1_repro, ensure_ascii=False)[:700], flush=True)
assert all(d1_repro["reproduction_ok"].values()), "D1 reproduction failed"

# ---------- design matrices ----------
Xtr = np.column_stack([pred_oof] + [tr[c].values.astype(float) for c in sum(groups.values(), [])])
Xte = np.column_stack([pred_te] + [te[c].values.astype(float) for c in sum(groups.values(), [])])
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
res_tr = ytr_log - pred_oof

# baseline G0-only (calibration)
g0 = Ridge(alpha=ALPHA).fit(Xtr_s[:, :1], res_tr)
pred_g0 = pred_te + g0.predict(Xte_s[:, :1])
# main
main = Ridge(alpha=ALPHA).fit(Xtr_s, res_tr)
pred_main = pred_te + main.predict(Xte_s)

res = dict(
    meta=dict(seed=SEED, alpha=ALPHA, n_folds=N_FOLDS, n_train=int(len(tr)), n_test=int(len(te)),
              n_features=int(Xtr.shape[1]), feature_names=feature_names, groups=group_of,
              zero_variance_dropped=drop, split="first_seen<=20261020 / >=20261021",
              primary_metric="delta R2 (main - G0-only) on log1p(y_test)"),
    metrics=dict(raw_D1=metrics(pred_te, yte_log), G0_only=metrics(pred_g0, yte_log),
                 main=metrics(pred_main, yte_log)),
    delta=dict(R2_vs_G0=round(float(r2_score(yte_log, pred_main) - r2_score(yte_log, pred_g0)), 4),
               R2_vs_raw=round(float(r2_score(yte_log, pred_main) - r2_score(yte_log, pred_te)), 4),
               MAE_vs_G0=round(metrics(pred_main, yte_log)["MAE_orig"] - metrics(pred_g0, yte_log)["MAE_orig"], 2)),
    coef=dict(zip(feature_names, [round(float(x), 4) for x in main.coef_])),
    note="OOF residuals for train; fixed full-train D1 for test; no test tuning.",
)
json.dump(res, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
print("[MAIN]", json.dumps(res["delta"], ensure_ascii=False), flush=True)

np.savez(f"{OUT}/bridge_data.npz", Xtr=Xtr, Xte=Xte, ytr_log=ytr_log, yte_log=yte_log,
         pred_oof=pred_oof, pred_te=pred_te, bvid_tr=tr.bvid.values, bvid_te=te.bvid.values,
         batch_te=te.batch.values, feature_names=np.array(feature_names), group_of=np.array(group_of),
         alpha=np.array([ALPHA]))
tr[["bvid", "batch", "n6", "age", "got_snapshot", "tname", "y"]].to_csv(f"{OUT}/bridge_train_meta.csv", index=False, encoding="utf-8-sig")
te[["bvid", "batch", "n6", "age", "got_snapshot", "tname", "y"]].to_csv(f"{OUT}/bridge_test_meta.csv", index=False, encoding="utf-8-sig")
print("EXP68-FIT-DONE")
