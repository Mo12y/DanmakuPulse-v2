# -*- coding: utf-8 -*-
"""exp67 D2 audit: rebuild conclusion-stability table, verify metrics, decompose by decision time t."""
import json
import os
import sys

import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
sys.path.insert(0, f"{V2}/experiments/exp_ml_scan_20261027")
import run_scan as rs  # noqa

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 20261027
OUT = f"{V2}/experiments/exp67_evidence_audit"
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, "/home/molly/projects/danmaku/analysis")
from danmaku_metrics import density_curve, extract_8_indicators  # noqa

DIMS = ["cold_pct", "cv", "main_peak_pct"]
d = rs.build_daily()
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
    rel = (g["send_time"].values - pub) / 3600.0
    off = g["time_offset"].values; fs = g["first_seen"].iloc[0]
    for t in [1, 3, 6, 12, 24]:
        if rel.max() < t + 0.5:
            continue
        m = rel <= t; k = int(m.sum())
        if k < 10:
            continue
        ind = extract_8_indicators(density_curve(off[m], dur), dur)
        rws.append(dict(bvid=bv, batch=bt, fs=fs, t=t, k=k,
                        cover=len(set((off[m] / max(dur, 1) * 20).astype(int))) / 20,
                        last1=float((rel >= t - 1).sum()) / k, dur=dur,
                        **{f"m_{x}": ind[x] for x in DIMS}, **{f"f_{x}": fullm[(bv, bt)][x] for x in DIMS}))
u2 = pd.DataFrame(rws)
tr2, te2 = u2[u2.fs <= 20261020], u2[u2.fs >= 20261021]
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

out = dict(n_train_rows=int(len(tr2)), n_train_videos=int(tr2.bvid.nunique()),
           n_test_rows=int(len(te2)), n_test_videos=int(te2.bvid.nunique()),
           bvid_overlap=len(set(tr2.bvid) & set(te2.bvid)),
           dup_test_rows=int(te2.bvid.duplicated().sum()),
           iqr=sorted((k, round(v, 4)) for k, v in q.items()),
           pos_rate_train=round(float(ytr2.mean()), 4), pos_rate_test=round(float(yte2.mean()), 4),
           AUC_base=round(float(roc_auc_score(yte2, base2)), 4),
           AUC_HGB=round(float(roc_auc_score(yte2, pg2)), 4),
           dAUC=round(float(roc_auc_score(yte2, pg2) - roc_auc_score(yte2, base2)), 4),
           PR_base=round(float(average_precision_score(yte2, base2)), 4),
           PR_HGB=round(float(average_precision_score(yte2, pg2)), 4),
           Brier_zero=round(float(yte2.mean()), 4),
           Brier_const=round(float(((yte2.mean() - yte2) ** 2).mean()), 4),
           Brier_HGB=round(float(((pg2 - yte2) ** 2).mean()), 4),
           Brier_LR=round(float(((pl2 - yte2) ** 2).mean()), 4))
out["by_t"] = {}
for t in [1, 3, 6, 12, 24]:
    m = te2.t.values == t
    if m.sum() == 0:
        continue
    y = yte2[m]
    out["by_t"][str(t)] = dict(n=int(m.sum()), pos_rate=round(float(y.mean()), 4),
                               auc_base=round(float(roc_auc_score(y, base2[m])), 4),
                               auc_hgb=round(float(roc_auc_score(y, pg2[m])), 4),
                               d_auc=round(float(roc_auc_score(y, pg2[m]) - roc_auc_score(y, base2[m])), 4))
# early decision only (t<=6)
me = te2.t.values <= 6
out["early_t_le_6"] = dict(n=int(me.sum()), pos_rate=round(float(yte2[me].mean()), 4),
                           auc_base=round(float(roc_auc_score(yte2[me], base2[me])), 4),
                           auc_hgb=round(float(roc_auc_score(yte2[me], pg2[me])), 4),
                           d_auc=round(float(roc_auc_score(yte2[me], pg2[me]) - roc_auc_score(yte2[me], base2[me])), 4))
# t-only baseline (score=t)
out["auc_t_only"] = round(float(roc_auc_score(yte2, te2.t.values)), 4)
json.dump(out, open(f"{OUT}/d2_audit.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
print("D2-DONE")
