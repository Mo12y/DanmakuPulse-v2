# -*- coding: utf-8 -*-
"""exp68 build_features.py — time-safe first-6h features for D1 residual bridge.

Only uses rows with 0 <= rel < 6h in the same (bvid,batch) snapshot, plus metadata known at publish.
Writes: features.csv, feature_manifest.csv, feature_audit.json.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp68_d1_residual_bridge"
os.makedirs(OUT, exist_ok=True)
COLS = ["bvid", "duration", "pubdate", "time_offset", "send_time", "part_count", "status",
        "tname", "text_len", "mode", "color", "user_hash"]

parts = []
for f in sorted(glob.glob(f"{V2}/data/cli_collected/hotlist_top200_202610*.csv")):
    day = int(re.search(r"(2026\d{4})", os.path.basename(f)).group(1))
    if not (20261012 <= day <= 20261026):
        continue
    df = pd.read_csv(f, usecols=COLS, encoding="utf-8-sig", low_memory=False)
    df["batch"] = day
    df["source_path"] = f
    parts.append(df)
d = pd.concat(parts, ignore_index=True)
for c in ["duration", "pubdate", "time_offset", "send_time", "part_count", "text_len", "mode", "color"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
d = d[(d["status"] == "ok") & d[["pubdate", "duration", "send_time"]].notna().all(axis=1) & (d["duration"] > 0)]
d = d[~(d["part_count"].fillna(0) >= 2)].copy()
d["rel"] = (d["send_time"] - d["pubdate"]) / 3600.0
d["uh"] = d["user_hash"].fillna("").astype(str).str.strip()

e = d[(d["rel"] >= 0) & (d["rel"] < 6)].copy()
e["bin"] = np.floor(e["rel"] * 2).astype(int).clip(0, 11)

base = e.groupby(["bvid", "batch"]).agg(
    n6=("bvid", "size"), tname=("tname", "first"), pubdate=("pubdate", "first"),
    duration=("duration", "first"), part_count=("part_count", "first"),
    source_path=("source_path", "first"))
idx = base.index

bc = (e.groupby(["bvid", "batch", "bin"]).size().unstack(fill_value=0)
      .reindex(columns=range(12), fill_value=0).reindex(idx, fill_value=0))
cnt = bc.values.astype(float)
n6 = cnt.sum(1)
p = cnt / np.maximum(n6[:, None], 1e-9)
entropy = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-12)), 0.0)).sum(1) / np.log(12)
mean = cnt.mean(1)
burst = np.where(mean > 0, cnt.std(1) / np.maximum(mean, 1e-9), 0.0)
peak_pos = cnt.argmax(1) / 11.0
peak_share = cnt.max(1) / np.maximum(n6, 1e-9)
zero_share = (cnt == 0).mean(1)


def gini_mat(M):
    n = M.shape[1]
    X = np.sort(M, axis=1)
    w = np.arange(1, n + 1)
    s = X.sum(1)
    num = 2 * (X * w).sum(1) - (n + 1) * s
    return np.where(s > 0, num / (n * np.maximum(s, 1e-9)), 0.0)


gini_bins = gini_mat(cnt)
sparse = n6 < 2
entropy[sparse] = 0.0
burst[sparse] = 0.0
peak_pos[sparse] = 0.0
peak_share[sparse] = 1.0
gini_bins[sparse] = 0.0

F2 = pd.DataFrame(dict(entropy_norm=entropy, burst_cv=burst, peak_pos=peak_pos,
                       peak_share=peak_share, zero_bin_share=zero_share, gini_bins=gini_bins), index=idx)

F3 = e.groupby(["bvid", "batch"]).agg(
    mean_text_len=("text_len", "mean"),
    long_share=("text_len", lambda s: float((s >= 10).mean())),
    colored_share=("color", lambda s: float((s != 16777215).mean())))

eu = e[e["uh"] != ""]
g4 = eu.groupby(["bvid", "batch"]).agg(n_user_rows=("uh", "size"), nu=("uh", "nunique"))
uc = eu.groupby(["bvid", "batch", "uh"]).size().rename("k").reset_index()
uc["p"] = uc["k"] / uc.groupby(["bvid", "batch"])["k"].transform("sum")
hhi = uc.groupby(["bvid", "batch"])["p"].apply(lambda s: float((s ** 2).sum()))
top1 = uc.groupby(["bvid", "batch"])["p"].max()
F4 = pd.DataFrame(dict(n_user_rows=g4["n_user_rows"], nu=g4["nu"], user_hhi=hhi, user_top1_share=top1))
F4["user_unique_share"] = F4["nu"] / F4["n_user_rows"]
F4["user_cov"] = F4["n_user_rows"] / base["n6"]
F4 = F4[["user_cov", "user_hhi", "user_top1_share", "user_unique_share"]]

sec = e.groupby(["bvid", "batch", "send_time"]).agg(rows=("bvid", "size"), nu=("uh", "nunique")).reset_index()
multi_rows = sec[sec["nu"] >= 2].groupby(["bvid", "batch"])["rows"].sum()
top_sec = sec.groupby(["bvid", "batch"])["rows"].max()
ru = eu.groupby(["bvid", "batch", "uh"]).size()
repeat_rows = ru[ru >= 2].groupby(level=[0, 1]).sum()

e2 = e.sort_values(["bvid", "batch", "time_offset"]).copy()
e2["rk"] = e2.groupby(["bvid", "batch"]).cumcount() + 1
e2["nn"] = e2.groupby(["bvid", "batch"])["time_offset"].transform("size")
e2["sx"] = e2.groupby(["bvid", "batch"])["time_offset"].transform("sum")
e2["rx"] = e2["rk"] * e2["time_offset"]
agg = e2.groupby(["bvid", "batch"]).agg(nn=("nn", "first"), sx=("sx", "first"),
                                        srx=("rx", lambda s: float(s.sum())))
pg = ((2 * agg["srx"] - (agg["nn"] + 1) * agg["sx"]) / (agg["nn"] * np.maximum(agg["sx"], 1e-9))).clip(0, 1)

F5 = pd.DataFrame(dict(
    multi_user_second_share=multi_rows.reindex(idx).fillna(0) / base["n6"],
    repeat_user_row_share=repeat_rows.reindex(idx).fillna(0) / base["n6"],
    progress_gini=pg.reindex(idx).fillna(0),
    second_top1_share=top_sec.reindex(idx).fillna(1) / base["n6"]))

feat = base.join(F2).join(F3).join(F4).join(F5).reset_index()
got = d.groupby(["bvid", "batch"]).size().rename("got_snapshot")
feat = feat.merge(got.reset_index(), on=["bvid", "batch"], how="left")

u = pd.read_csv(f"{V2}/experiments/exp67_evidence_audit/d1_units.csv")[
    ["bvid", "batch", "fs", "c0", "y", "dur", "age"]]
m = u.merge(feat, on=["bvid", "batch"], how="left", validate="one_to_one")
assert len(m) == len(u) == 1313, (len(m), len(u))
m["split"] = np.where(m["fs"] <= 20261020, "train", "test")
m["pub_dt"] = pd.to_datetime(m["pubdate"], unit="s", utc=True).dt.tz_convert("Asia/Shanghai")
m["pub_hour"] = m["pub_dt"].dt.hour
m["pub_dow"] = m["pub_dt"].dt.dayofweek
m["log_dur"] = np.log1p(m["duration"])
m["part_count_bin"] = (m["part_count"].fillna(0) >= 1).astype(int)
m["pub_hour_sin"] = np.sin(2 * np.pi * m["pub_hour"] / 24)
m["pub_hour_cos"] = np.cos(2 * np.pi * m["pub_hour"] / 24)
m["pub_dow_sin"] = np.sin(2 * np.pi * m["pub_dow"] / 7)
m["pub_dow_cos"] = np.cos(2 * np.pi * m["pub_dow"] / 7)

feat_cols = ["entropy_norm", "burst_cv", "peak_pos", "peak_share", "zero_bin_share", "gini_bins",
             "mean_text_len", "long_share", "colored_share",
             "user_cov", "user_hhi", "user_top1_share", "user_unique_share",
             "multi_user_second_share", "repeat_user_row_share", "progress_gini", "second_top1_share"]
static_cols = ["log_dur", "part_count_bin", "pub_hour_sin", "pub_hour_cos", "pub_dow_sin", "pub_dow_cos"]
keep = ["bvid", "batch", "fs", "split", "age", "c0", "y", "dur", "n6", "tname", "pubdate",
        "source_path", "got_snapshot"] + static_cols + feat_cols
m[keep].to_csv(f"{OUT}/features.csv", index=False, encoding="utf-8-sig")

rows = []
for r in m.itertuples():
    for g in ["G0", "G1", "G2", "G3", "G4", "G5"]:
        nf = {"G0": 1, "G1": 6, "G2": 6, "G3": 3, "G4": 4, "G5": 4}[g]
        reason = ""
        if g == "G2" and r.n6 < 2:
            reason = "n6<2: shape constants"
        if g == "G4" and (pd.isna(r.user_cov) or r.user_cov == 0):
            reason = "no user_hash in first 6h"
        rows.append(dict(bvid=r.bvid, batch=r.batch, split=r.split, decision_time="6h after publish",
                         snapshot_age=round(float(r.age), 2), feature_group=g, n_features=nf,
                         source_path=r.source_path, time_safe="yes", missing_reason=reason))
man = pd.DataFrame(rows)
man.to_csv(f"{OUT}/feature_manifest.csv", index=False, encoding="utf-8-sig")

audit = dict(
    n_units=int(len(m)), n_train=int((m.split == "train").sum()), n_test=int((m.split == "test").sum()),
    matched=bool(m[feat_cols].notna().all(axis=1).all()),
    n6_quantiles={str(k): float(v) for k, v in m.n6.quantile([0, .01, .05, .25, .5, .75, .95, 1]).items()},
    n6_lt2=int((m.n6 < 2).sum()), n6_lt10=int((m.n6 < 10).sum()),
    user_cov_min=float(m.user_cov.min()), user_cov_median=float(m.user_cov.median()),
    missing=({c: int(m[c].isna().sum()) for c in feat_cols}),
    n_features={"G0": 1, "G1": 6, "G2": 6, "G3": 3, "G4": 4, "G5": 4},
    time_safe_scope="rows with 0<=rel<6h in the same (bvid,batch) snapshot; static metadata from publish",
    excluded=["full got/reported", "full-video peaks/means/text", "target window (6h,30h] rows",
              "coverage (circular)", "cross-batch repetition (needs future batches)"],
    snapshot_note="snapshot age>=31h; early rows may be affected by deletion/pool rolling, same as D1 c0",
    zero_variance_train=[c for c in feat_cols + static_cols if m.loc[m.split == "train", c].std() == 0],
)
json.dump(audit, open(f"{OUT}/feature_audit.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(audit, ensure_ascii=False, indent=1)[:2000])
print("EXP68-FEATURES-DONE")
