# -*- coding: utf-8 -*-
"""exp67 D1/D2 audit: rebuild tables, quantify selection/duplication, recompute metrics + cluster bootstrap."""
import json
import os
import sys

import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
sys.path.insert(0, f"{V2}/experiments/exp_ml_scan_20261027")
sys.path.insert(0, f"{V2}/experiments/exp38_integrate")
import run_scan as rs  # noqa

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score

SEED = 20261027
OUT = f"{V2}/experiments/exp67_evidence_audit"
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


d = rs.build_daily()
log("[daily] rows=%d videos=%d batches=%d" % (len(d), d.bvid.nunique(), d.batch.nunique()))

# ---------- attrition, duplicates, batch composition ----------
n_groups = d.groupby(["bvid", "batch"]).ngroups
rows, drop_age, drop_c0 = [], 0, 0
for (bv, bt), g in d.groupby(["bvid", "batch"]):
    pub = float(g["pubdate"].iloc[0]); dur = float(g["duration"].iloc[0])
    rel = (g["send_time"].values - pub) / 3600.0
    if rel.max() < 31.0:
        drop_age += 1
        continue
    n01 = ((rel >= 0) & (rel < 1)).sum(); n13 = ((rel >= 1) & (rel < 3)).sum()
    n36 = ((rel >= 3) & (rel < 6)).sum(); c0 = n01 + n13 + n36
    if c0 < 1:
        drop_c0 += 1
        continue
    rows.append((bv, bt, g["first_seen"].iloc[0], n01, n13, n36, c0,
                 ((rel > 6) & (rel <= 30)).sum(), dur, rel.max()))
u = pd.DataFrame(rows, columns=["bvid", "batch", "fs", "n01", "n13", "n36", "c0", "y", "dur", "age"])
tr, te = u[u.fs <= 20261020], u[u.fs >= 20261021]
log("[attrition] groups=%d dropped_age<31h=%d dropped_c0<1=%d kept_rows=%d videos=%d"
    % (n_groups, drop_age, drop_c0, len(u), u.bvid.nunique()))
log("[dup] rows=%d unique_bvid=%d repeated_rows=%d | train rows=%d vids=%d | test rows=%d vids=%d"
    % (len(u), u.bvid.nunique(), len(u) - u.bvid.nunique(), len(tr), tr.bvid.nunique(), len(te), te.bvid.nunique()))
log("[train batch mix] %s" % tr.batch.value_counts().sort_index().to_dict())
log("[test batch mix] %s" % te.batch.value_counts().sort_index().to_dict())
log("[train rows with batch>=1021] %d / %d (%.1f%%)"
    % ((tr.batch >= 20261021).sum(), len(tr), 100 * (tr.batch >= 20261021).mean()))
log("[y==0] train %d/%d (%.2f%%) test %d/%d (%.2f%%)"
    % ((tr.y == 0).sum(), len(tr), 100 * (tr.y == 0).mean(), (te.y == 0).sum(), len(te), 100 * (te.y == 0).mean()))
log("[y quantiles] train %s test %s" % (tr.y.quantile([.5, .9, .99]).round(1).to_dict(),
                                        te.y.quantile([.5, .9, .99]).round(1).to_dict()))
log("[age at snapshot] train %s test %s" % (tr.age.quantile([.05, .5, .95]).round(1).to_dict(),
                                            te.age.quantile([.05, .5, .95]).round(1).to_dict()))
dup_te = te.bvid.duplicated().sum()
log("[test dup rows] %d (%.1f%%)" % (dup_te, 100 * dup_te / len(te)))

# c0 drift across batches for the same bvid (is c0 really a 6h-time quantity?)
drift = []
for bv, g in u.groupby("bvid"):
    if len(g) < 2:
        continue
    g = g.sort_values("age")
    drift.append((g.iloc[-1].c0 - g.iloc[0].c0) / max(g.iloc[0].c0, 1))
log("[c0 drift first->last snapshot] n=%d median=%.4f p5=%.4f p95=%.4f share_abs>1%%=%.2f"
    % (len(drift), np.median(drift), np.percentile(drift, 5), np.percentile(drift, 95),
       np.mean(np.abs(drift) > 0.01)))

# ---------- D1 metrics ----------
def mats(df, feats):
    cols = {"c0": np.log1p(df.c0), "n01": np.log1p(df.n01), "n13": np.log1p(df.n13),
            "n36": np.log1p(df.n36), "slope": np.log1p(df.n36) - np.log1p(df.n13), "dur": np.log1p(df.dur)}
    return np.column_stack([cols[f] for f in feats])


F1F, F6F = ["c0"], ["c0", "n36", "n13", "n01", "slope", "dur"]
ytr, yte = np.log1p(tr.y.values), np.log1p(te.y.values)
b1 = LinearRegression().fit(mats(tr, F1F), ytr)
m6 = LinearRegression().fit(mats(tr, F6F), ytr)
hgb = HistGradientBoostingRegressor(max_depth=3, random_state=SEED).fit(mats(tr, F6F), ytr)


def rep(pred):
    r2 = r2_score(yte, pred)
    po, to = np.clip(np.expm1(pred), 0, None), np.expm1(yte)
    return dict(R2=round(float(r2), 4), MAE_log=round(float(mean_absolute_error(yte, pred)), 4),
                RMSE_log=round(float(np.sqrt(mean_squared_error(yte, pred))), 4),
                MAE_orig=round(float(mean_absolute_error(to, po)), 1),
                RMSE_orig=round(float(np.sqrt(mean_squared_error(to, po))), 1))


D1 = dict(split=dict(n_train=int(len(tr)), n_test=int(len(te)),
                     bvid_overlap=len(set(tr.bvid) & set(te.bvid))),
          B1=rep(b1.predict(mats(te, F1F))),
          OLS6=rep(m6.predict(mats(te, F6F))),
          HGB=rep(hgb.predict(mats(te, F6F))),
          persistence=rep(np.log1p(4.0 * te.c0.values)),
          histmean=rep(np.full(len(te), ytr.mean())))
log("[D1] %s" % json.dumps(D1, ensure_ascii=False))
u.to_csv(f"{OUT}/d1_units.csv", index=False)

# per-group MAE: snapshot batch vs first_seen cohort (r6_final labels the latter "by_test_batch")
po_all = np.clip(np.expm1(pred6 := m6.predict(mats(te, F6F))), 0, None)
te2 = te.copy(); te2["ae"] = np.abs(np.expm1(te2.y.values.astype(float)) - po_all)
by_batch_audit = {str(b): dict(n=int(len(g)), mae=round(float(g.ae.mean()), 1))
                  for b, g in te2.groupby("batch") if len(g) >= 5}
by_fs_audit = {str(b): dict(n=int(len(g)), mae=round(float(g.ae.mean()), 1))
               for b, g in te2.groupby("fs") if len(g) >= 10}
log("[D1 by snapshot batch] %s" % by_batch_audit)
log("[D1 by first_seen cohort] %s" % by_fs_audit)

# cluster bootstrap over bvid (test)
rng = np.random.default_rng(SEED)
vids = te.bvid.values
uniq = np.unique(vids)
pos = {v: np.where(vids == v)[0] for v in uniq}
pred6 = m6.predict(mats(te, F6F))
bs = []
for _ in range(2000):
    pick = rng.choice(uniq, len(uniq), True)
    idx = np.concatenate([pos[v] for v in pick])
    if len(np.unique(yte[idx])) < 2:
        continue
    bs.append((r2_score(yte[idx], pred6[idx]),
               mean_absolute_error(np.expm1(yte[idx]), np.clip(np.expm1(pred6[idx]), 0, None))))
bs = np.array(bs)
log("[D1 bootstrap bvid] R2 95%% CI [%.4f, %.4f]; MAE_orig 95%% CI [%.1f, %.1f] (B=%d)"
    % (np.percentile(bs[:, 0], 2.5), np.percentile(bs[:, 0], 97.5),
       np.percentile(bs[:, 1], 2.5), np.percentile(bs[:, 1], 97.5), len(bs)))

# dedup sensitivity: one row per bvid (latest snapshot)
td = te.sort_values("age").drop_duplicates("bvid", keep="last")
pred_d = np.clip(m6.predict(mats(td, F6F)), 0, 20)
yv = np.log1p(np.asarray(td.y.values, float)); pv = np.asarray(pred_d, float)
log("[D1 dedup debug] n=%d nonfinite_y=%d nonfinite_p=%d y_max=%.1f"
    % (len(yv), int((~np.isfinite(yv)).sum()), int((~np.isfinite(pv)).sum()), float(np.nanmax(td.y.values))))
ok = np.isfinite(yv) & np.isfinite(pv)
r2d = 1 - ((yv[ok] - pv[ok]) ** 2).sum() / ((yv[ok] - yv[ok].mean()) ** 2).sum()
maed = float(np.mean(np.abs(np.expm1(yv[ok]) - np.expm1(pv[ok]))))
log("[D1 dedup one-row-per-video] n=%d R2=%.4f MAE_orig=%.1f" % (int(ok.sum()), r2d, maed))

# strict temporal: train rows only from batches <=1020
trs = tr[tr.batch <= 20261020]
m6s = LinearRegression().fit(mats(trs, F6F), np.log1p(trs.y.values))
pred_s = np.clip(m6s.predict(mats(te, F6F)), 0, 20)
ays = np.asarray(yte, float); aps = np.asarray(pred_s, float)
oks = np.isfinite(ays) & np.isfinite(aps)
r2s = 1 - ((ays[oks] - aps[oks]) ** 2).sum() / ((ays[oks] - ays[oks].mean()) ** 2).sum()
maes = float(np.mean(np.abs(np.expm1(ays[oks]) - np.expm1(aps[oks]))))
log("[D1 strict-train<=1020] n_train=%d R2=%.4f MAE_orig=%.1f" % (len(trs), r2s, maes))

json.dump(dict(attrition=dict(groups=int(n_groups), drop_age=int(drop_age), drop_c0=int(drop_c0)),
               dup=dict(rows=int(len(u)), videos=int(u.bvid.nunique()), train_rows=int(len(tr)),
                        train_videos=int(tr.bvid.nunique()), test_rows=int(len(te)),
                        test_videos=int(te.bvid.nunique())),
               train_batch_mix={str(k): int(v) for k, v in tr.batch.value_counts().items()},
               d1=D1, bootstrap=dict(r2_ci=[float(np.percentile(bs[:, 0], 2.5)), float(np.percentile(bs[:, 0], 97.5))],
                                     mae_ci=[float(np.percentile(bs[:, 1], 2.5)), float(np.percentile(bs[:, 1], 97.5))]),
               dedup=dict(n=int(ok.sum()), r2=float(r2d), mae_orig=maed),
               by_batch=by_batch_audit, by_first_seen=by_fs_audit,
               strict_train=dict(n_train=int(len(trs)), r2=float(r2s), mae_orig=maes)),
          open(f"{OUT}/d1_audit.json", "w"), ensure_ascii=False, indent=1)
log("D1-DONE")
