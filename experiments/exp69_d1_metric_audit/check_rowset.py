# -*- coding: utf-8 -*-
"""exp69 check_rowset.py — reproduce exp55's 395-row test set and compare to exp67 units."""
import json
import sys

import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp69_d1_metric_audit"
sys.path.insert(0, f"{V2}/experiments/exp_ml_scan_20261027")
import run_scan as rs  # noqa

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
u55 = pd.DataFrame(rows, columns=["bvid", "fs", "n01", "n13", "n36", "c0", "y", "dur"])
te55 = u55[u55.fs >= 20261021]
u67 = pd.read_csv(f"{V2}/experiments/exp67_evidence_audit/d1_units.csv")
te67 = u67[u67.fs >= 20261021]
num = ["fs", "n01", "n13", "n36", "c0", "y", "dur"]
a = te55[num].astype(float).round(8).value_counts().sort_index()
b = te67[num].astype(float).round(8).value_counts().sort_index()

fe = pd.read_csv(f"{V2}/experiments/exp68_d1_residual_bridge/features.csv", encoding="utf-8-sig")
fe = fe[fe.split == "test"]
m = te67.merge(fe, on=["bvid", "batch"], suffixes=("_u", "_f"))
res = dict(
    n55=int(len(te55)), n67=int(len(te67)),
    bvid_sets_equal=bool(set(te55.bvid) == set(te67.bvid)),
    bvid_dups_55=int(te55.bvid.duplicated().sum()), bvid_dups_67=int(te67.bvid.duplicated().sum()),
    numeric_multiset_equal=bool(a.equals(b)),
    persistence_mae_55=round(float(np.mean(np.abs(te55.y.values - 4 * te55.c0.values))), 2),
    persistence_mae_67=round(float(np.mean(np.abs(te67.y.values - 4 * te67.c0.values))), 2),
    features_c0_maxdiff=float(np.max(np.abs(m.c0_u - m.c0_f))),
    features_y_maxdiff=float(np.max(np.abs(m.y_u - m.y_f))),
    features_dur_maxdiff=float(np.max(np.abs(m.dur_u - m.dur_f))),
)
json.dump(res, open(f"{OUT}/rowset_check.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
print("EXP69-ROWSET-DONE")
