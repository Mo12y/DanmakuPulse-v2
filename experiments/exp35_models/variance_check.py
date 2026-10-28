import os
import sys

import numpy as np

sys.path.insert(0, "/home/molly/projects/danmaku-v2/jobs")
sys.path.insert(0, "/home/molly/projects/danmaku-v2/experiments/exp35_models")
import m2_l6_burst_risk as M  # noqa: E402
from lambda_strata import reconstruct_points, strata_of  # noqa: E402

OUT = "/home/molly/projects/danmaku-v2/data-out/exp35_models"
SEEDS = (20261026, 20261027, 20261028, 20261029, 20261030)

d, lam, thr, y, vid, fut = reconstruct_points()
sid = strata_of(lam)
mask = sid == 2          # λ>=1
print("λ>=1 层：分数方差分解（跨视频 vs 视频内），5 seeds 平均")
print(f"{'model':7s} {'between_var':>12s} {'within_var':>11s} {'ICC':>7s} "
      f"{'corr(视频均值分数, 视频爆发率)':>26s}")
for tag in ("base", "ET_l10", "MLP"):
    iccs, cors, vws = [], [], []
    for s in SEEDS:
        z = np.load(os.path.join(OUT, f"rcv_oof_seed{s}.npz"))
        p = -d["X6"][:, 0] if tag == "base" else z[tag]
        mus, within, rates = [], [], []
        for v in np.unique(vid[mask]):
            m = mask & (vid == v)
            mus.append(float(p[m].mean()))
            within.append(float(p[m].var(ddof=0)))
            rates.append(float(y[m].mean()))
        mus, within, rates = np.array(mus), np.array(within), np.array(rates)
        vb, vw = mus.var(ddof=0), within.mean()
        iccs.append(vb / (vb + vw) if (vb + vw) > 0 else np.nan)
        cors.append(float(np.corrcoef(mus, rates)[0, 1]))
        vws.append(vw)
    print(f"{tag:7s} {np.nanmean(iccs):12.3f} {np.mean(vws):11.4f} "
          f"{np.nanmean(iccs):7.3f} {np.mean(cors):26.3f}")
