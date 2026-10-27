# -*- coding: utf-8 -*-
"""validate_cohort_builder.py · 用内联 cohort 构建代码回放 D-train，比对冻结点表（100% 一致才放行）"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
sys.path.insert(0, f"{V2}/experiments/exp_ml_scan_20261027")
import run_scan as rs  # noqa: E402

DIMS_FEATS = ["bvid", "batch", "t", "lam", "y", "y_count", "since", "censored",
              "log1p_lam", "slope", "prog", "log1p_duration"]


def inline_cohort(d):
    """与 audit_scan.py 的 D4 cohort 构建同构（去掉 bvid 过滤）。d 需含 bvid/batch/duration/time_offset/part_count"""
    cw = []
    for (bv, bt), g in d.groupby(["bvid", "batch"]):
        dur = float(g["duration"].iloc[0]); Kf = int(dur // 5)
        pc = g["part_count"].iloc[0]
        if Kf < 18 or (pd.notna(pc) and pc >= 2):
            continue
        off = g["time_offset"].values.astype(float)
        edge = np.isclose(off, dur); off = off[~edge]
        b = (off // 5).astype(int); b = b[b < Kf]
        c = np.bincount(b, minlength=Kf)
        for t in range(12, Kf - 5, 6):
            lam = float(c[t - 12:t].mean()); thr = max(3.0, lam + 2.0 * np.sqrt(max(lam, 0.0)))
            sinc, cens = 12, 1
            for dd in range(1, 13):
                kk = t - dd
                if kk < 12:
                    break
                lk = float(c[kk - 12:kk].mean()); tk = max(3.0, lk + 2.0 * np.sqrt(max(lk, 0.0)))
                if c[kk] >= tk:
                    sinc, cens = dd, 0
                    break
            cw.append((bv, bt, t, lam, int(c[t:t + 6].max() >= thr), float(c[t:t + 6].sum()), sinc, cens,
                       float(np.log1p(lam)), float(np.log1p(c[t - 6:t].mean()) - np.log1p(c[t - 12:t - 6].mean())),
                       t / Kf, float(np.log1p(dur))))
    return pd.DataFrame(cw, columns=DIMS_FEATS)


# 用 D-train 原始数据构造 daily 格式回放
OLD_WH = "/home/molly/projects/danmaku/data/warehouse/ods"
fr = []
for p in sorted(glob.glob(f"{OLD_WH}/dt=*")):
    if not re.search(r"2026090[27]", p):
        continue
    fs = [f for f in glob.glob(f"{p}/**/*.parquet", recursive=True) if not os.path.basename(f).startswith(".")]
    x = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    x["batch"] = int(re.search(r"(2026\d{4})", os.path.basename(p)).group(1))
    x["part_count"] = np.nan
    fr.append(x[["bvid", "batch", "duration", "time_offset", "part_count"]])
a = pd.concat(fr, ignore_index=True)
for c in ["duration", "time_offset"]:
    a[c] = pd.to_numeric(a[c], errors="coerce")
a = a.dropna(subset=["duration", "time_offset"])
a = a[(a["duration"] > 0) & (a["time_offset"] >= 0)]

P_re = inline_cohort(a)
P_dt = rs.build_points()
m = P_re.merge(P_dt[DIMS_FEATS], on=["bvid", "batch", "t"], suffixes=("", "_ref"))
lam_ok = float((np.abs(m.lam - m.lam_ref) < 1e-6).mean())
y_ok = float((m.y == m.y_ref).mean())
yc_ok = float((np.abs(m.y_count - m.y_count_ref) < 1e-6).mean())
print(f"inline-builder replay: n={len(P_re)} matched={len(m)} lam={lam_ok:.6f} y={y_ok:.6f} y_count={yc_ok:.6f}")
print("VERDICT:", "PASS" if (lam_ok == 1.0 and y_ok == 1.0) else "FAIL")
