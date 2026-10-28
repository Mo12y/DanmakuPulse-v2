# -*- coding: utf-8 -*-
"""exp68 bootstrap_controls.py — cluster bootstrap, group ablation, negative controls, sensitivity."""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp68_d1_residual_bridge"
SEED = 20261028
ALPHA = 1.0
B_BOOT = 2000
B_SHUFFLE = 1000
B_RANDOM = 500

d = np.load(f"{OUT}/bridge_data.npz", allow_pickle=True)
Xtr, Xte = d["Xtr"], d["Xte"]
ytr, yte = d["ytr_log"], d["yte_log"]
pred_oof, pred_te = d["pred_oof"], d["pred_te"]
bvid_tr, bvid_te = d["bvid_tr"], d["bvid_te"]
names = list(d["feature_names"])
gof = list(d["group_of"])
res_tr = ytr - pred_oof
tr_meta = pd.read_csv(f"{OUT}/bridge_train_meta.csv", encoding="utf-8-sig")
te_meta = pd.read_csv(f"{OUT}/bridge_test_meta.csv", encoding="utf-8-sig")
assert len(tr_meta) == len(Xtr) and len(te_meta) == len(Xte)
assert (tr_meta.bvid.values == bvid_tr).all() and (te_meta.bvid.values == bvid_te).all()

idx = {g: [i for i, x in enumerate(gof) if x == g] for g in ["G0", "G1", "G2", "G3", "G4", "G5"]}
struct_idx = sum([idx[g] for g in ["G2", "G3", "G4", "G5"]], [])


def fit_pred(cols, Xtr_use=None, Xte_use=None):
    A = Xtr if Xtr_use is None else Xtr_use
    B = Xte if Xte_use is None else Xte_use
    sc = StandardScaler().fit(A[:, cols])
    m = Ridge(alpha=ALPHA).fit(sc.transform(A[:, cols]), res_tr)
    return pred_te + m.predict(sc.transform(B[:, cols]))


def delta_r2(p_main, p_base, y):
    return float(r2_score(y, p_main) - r2_score(y, p_base))


p_g0 = fit_pred(idx["G0"])
p_main = fit_pred(list(range(Xtr.shape[1])))
obs = delta_r2(p_main, p_g0, yte)

# ---------- cluster bootstrap ----------
rng = np.random.default_rng(SEED)
vids = np.unique(bvid_te)
pos = {v: np.where(bvid_te == v)[0] for v in vids}
boot = []
for _ in range(B_BOOT):
    pick = rng.choice(vids, len(vids), True)
    ii = np.concatenate([pos[v] for v in pick])
    if np.unique(yte[ii]).size < 2:
        continue
    boot.append(delta_r2(p_main[ii], p_g0[ii], yte[ii]))
boot = np.array(boot)
ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

# MAE delta CI
def mae_o(p, y):
    return mean_absolute_error(np.expm1(y), np.expm1(np.clip(p, 0, 20)))


mae_boot = []
for _ in range(B_BOOT):
    pick = rng.choice(vids, len(vids), True)
    ii = np.concatenate([pos[v] for v in pick])
    mae_boot.append(mae_o(p_main[ii], yte[ii]) - mae_o(p_g0[ii], yte[ii]))
mae_ci = [float(np.percentile(mae_boot, 2.5)), float(np.percentile(mae_boot, 97.5))]

# ---------- group ablation ----------
abl = {}
cols_so_far = list(idx["G0"])
abl["G0"] = dict(n_features=1, R2=round(float(r2_score(yte, p_g0)), 4), dR2_vs_G0=0.0)
for g in ["G1", "G2", "G3", "G4", "G5"]:
    cols_so_far += idx[g]
    pp = fit_pred(cols_so_far)
    abl[f"G0+{g}" if len(abl) == 1 else f"+{g}"] = dict(
        n_features=len(cols_so_far), R2=round(float(r2_score(yte, pp)), 4),
        dR2_vs_G0=round(delta_r2(pp, p_g0, yte), 4))
# each group alone (G0 + one group)
alone = {}
for g in ["G1", "G2", "G3", "G4", "G5"]:
    pp = fit_pred(idx["G0"] + idx[g])
    alone[g] = dict(n_features=1 + len(idx[g]), R2=round(float(r2_score(yte, pp)), 4),
                    dR2_vs_G0=round(delta_r2(pp, p_g0, yte), 4))
# structure beyond static (G0+G1 -> +G2..G5)
p_g1 = fit_pred(idx["G0"] + idx["G1"])
p_g1_struct = fit_pred(idx["G0"] + idx["G1"] + struct_idx)
struct_beyond_static = dict(R2_G0G1=round(float(r2_score(yte, p_g1)), 4),
                            R2_G0G1_struct=round(float(r2_score(yte, p_g1_struct)), 4),
                            dR2=round(delta_r2(p_g1_struct, p_g1, yte), 4),
                            n_extra_features=len(struct_idx))

# ---------- negative controls ----------
def shuffle_within_batch(X, cols, batch):
    Z = X.copy()
    for b in np.unique(batch):
        ii = np.where(batch == b)[0]
        perm = rng.permutation(ii)
        Z[np.ix_(ii, cols)] = X[np.ix_(perm, cols)]
    return Z


null_shuffle = []
for _ in range(B_SHUFFLE):
    Xs = shuffle_within_batch(Xtr, struct_idx, tr_meta.batch.values)
    pp = fit_pred(list(range(Xtr.shape[1])), Xtr_use=Xs, Xte_use=Xte)
    null_shuffle.append(delta_r2(pp, p_g0, yte))
null_shuffle = np.array(null_shuffle)
p_shuffle = float((1 + np.sum(null_shuffle >= obs)) / (1 + len(null_shuffle)))

null_random = []
for _ in range(B_RANDOM):
    Xs = Xtr.copy(); Xt = Xte.copy()
    Xs[:, struct_idx] = rng.normal(size=(len(Xs), len(struct_idx)))
    Xt[:, struct_idx] = rng.normal(size=(len(Xt), len(struct_idx)))
    pp = fit_pred(list(range(Xtr.shape[1])), Xtr_use=Xs, Xte_use=Xt)
    null_random.append(delta_r2(pp, p_g0, yte))
null_random = np.array(null_random)
p_random = float((1 + np.sum(null_random >= obs)) / (1 + len(null_random)))

# ---------- sensitivity (same fitted model, subgroup metrics) ----------
got_med = float(tr_meta.got_snapshot.median())
age_med = float(tr_meta.age.median())
top_t = tr_meta.tname.value_counts().head(7).index.tolist()


def sub(mask, name):
    mask = np.asarray(mask, bool)
    if mask.sum() < 20 or np.unique(yte[mask]).size < 2:
        return dict(n=int(mask.sum()), dR2=None)
    return dict(n=int(mask.sum()), R2_main=round(float(r2_score(yte[mask], p_main[mask])), 4),
                R2_G0=round(float(r2_score(yte[mask], p_g0[mask])), 4),
                dR2=round(delta_r2(p_main[mask], p_g0[mask], yte[mask]), 4))


sens = dict(
    n6_lt10=sub(te_meta.n6 < 10, "n6<10"), n6_ge10=sub(te_meta.n6 >= 10, "n6>=10"),
    got_low=sub(te_meta.got_snapshot <= got_med, "got_low"), got_high=sub(te_meta.got_snapshot > got_med, "got_high"),
    age_low=sub(te_meta.age <= age_med, "age_low"), age_high=sub(te_meta.age > age_med, "age_high"),
    tname_top7=sub(te_meta.tname.isin(top_t), "top7"), tname_other=sub(~te_meta.tname.isin(top_t), "other"),
    by_batch={str(b): sub(te_meta.batch == b, str(b)) for b in sorted(te_meta.batch.unique())},
    cutoffs=dict(got_median_train=got_med, age_median_train=age_med, top7=top_t),
)

# ---------- descriptive: structure vs c0 ----------
corr = {}
for g in ["G2", "G3", "G4", "G5"]:
    corr[g] = {}
    for c in [names[i] for i in idx[g]]:
        j = names.index(c)
# use c0 from features.csv (test rows) for descriptive correlation
feat = pd.read_csv(f"{OUT}/features.csv", encoding="utf-8-sig")
fe = feat[feat.split == "test"].reset_index(drop=True)
assert (fe.bvid.values == bvid_te).all()
for g in ["G2", "G3", "G4", "G5"]:
    corr[g] = {}
    for c in [names[i] for i in idx[g]]:
        corr[g][c] = round(float(spearmanr(fe[c].values, np.log1p(fe.c0.values)).statistic), 3)

# ---------- D1 cluster CI (for d1_repro.json) ----------
d1p = json.load(open(f"{OUT}/d1_repro.json"))
br, bm = [], []
rng_d1 = np.random.default_rng(SEED)
for _ in range(B_BOOT):
    pick = rng_d1.choice(vids, len(vids), True)
    ii = np.concatenate([pos[v] for v in pick])
    if np.unique(yte[ii]).size < 2:
        continue
    br.append(float(r2_score(yte[ii], pred_te[ii])))
    bm.append(float(mean_absolute_error(np.expm1(yte[ii]), np.expm1(np.clip(pred_te[ii], 0, 20)))))
d1p["bootstrap"] = dict(r2_ci95=[round(float(np.percentile(br, 2.5)), 4), round(float(np.percentile(br, 97.5)), 4)],
                        mae_ci95=[round(float(np.percentile(bm, 2.5)), 1), round(float(np.percentile(bm, 97.5)), 1)],
                        B=len(br), seed=SEED, cluster="bvid")
json.dump(d1p, open(f"{OUT}/d1_repro.json", "w"), ensure_ascii=False, indent=1)

out = dict(
    primary=dict(observed_delta_R2=round(obs, 4), boot_ci95=ci, boot_B=len(boot),
                 delta_MAE=round(mae_o(p_main, yte) - mae_o(p_g0, yte), 2), delta_MAE_ci95=mae_ci),
    controls=dict(shuffle_within_batch=dict(mean=round(float(null_shuffle.mean()), 4),
                                            p95=round(float(np.percentile(null_shuffle, 95)), 4),
                                            p_value=p_shuffle, B=len(null_shuffle)),
                  random_structure=dict(mean=round(float(null_random.mean()), 4),
                                        p95=round(float(np.percentile(null_random, 95)), 4),
                                        p_value=p_random, B=len(null_random))),
    ablation=dict(cumulative=abl, alone=alone, structure_beyond_static=struct_beyond_static),
    sensitivity=sens,
    feature_vs_c0_spearman=corr,
    note="p_value = (1+#null>=obs)/(1+B); primary = main(G0+G1..G5) vs G0-only calibration.",
)
old = json.load(open(f"{OUT}/results.json"))
old.update(out)
json.dump(old, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
json.dump(out, open(f"{OUT}/results_controls.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:4000])
print("EXP68-CONTROLS-DONE")
