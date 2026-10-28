# -*- coding: utf-8 -*-
"""exp42 · 检验 C1 的数据可行性（只读；got<600 vs got>=600 分组 + min_offset 交叉核实）
统计量（预注册）：观测窗=[min(time_offset), duration]；60s 桶；
  S1 = 观测窗等分 K=5 段的段均值 CV（主）；S2 = (P90-P10)/median（副）；S3 = 零桶占比（副）；
  "明显变化"：S1>=0.5 / S1>=1.0 双阈值（先于分组对比写死）。
混杂：时长桶(<10min/10-30/>30)分层 + 分区 top5 列示；效应量 CI = 视频级 bootstrap(B=2000)。
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp42_pool_feasibility"
os.makedirs(OUT, exist_ok=True)
SEED = 20261027
COLS = ["bvid", "tname", "duration", "pubdate", "time_offset", "send_time",
        "danmaku_got", "part_count", "status"]


def seg_cv(off, dur, obs_start, K=5):
    """观测窗 [obs_start, dur] 等分 K 段的段均值 CV；桶=60s"""
    lo, hi = int(obs_start // 60), int(dur // 60)
    if hi - lo < K:  # 观测窗不足 K 桶
        return None
    b = (np.clip(off, 0, None) // 60).astype(int)
    c = np.bincount(b[(b >= lo) & (b < hi)] - lo, minlength=hi - lo)
    seg = np.array_split(c, K)
    means = np.array([s.mean() for s in seg])
    m = means.mean()
    return float(means.std() / m) if m > 0 else None


def robust_amp(off, dur, obs_start):
    lo, hi = int(obs_start // 60), int(dur // 60)
    if hi - lo < 10:
        return None
    b = (np.clip(off, 0, None) // 60).astype(int)
    c = np.bincount(b[(b >= lo) & (b < hi)] - lo, minlength=hi - lo)
    med = np.median(c)
    return float((np.quantile(c, 0.9) - np.quantile(c, 0.1)) / med) if med > 0 else None


def zero_share(off, dur, obs_start):
    lo, hi = int(obs_start // 60), int(dur // 60)
    if hi - lo < 10:
        return None
    b = (np.clip(off, 0, None) // 60).astype(int)
    c = np.bincount(b[(b >= lo) & (b < hi)] - lo, minlength=hi - lo)
    return float((c == 0).mean())


def main():
    parts = []
    for f in sorted(glob.glob(f"{V2}/data/cli_collected/hotlist_top200_202610*.csv")):
        day = int(re.search(r"(2026\d{4})", os.path.basename(f)).group(1))
        if day > 20261026:
            continue
        d = pd.read_csv(f, usecols=COLS, encoding="utf-8-sig", low_memory=False)
        d["batch"] = day
        parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    for c in ["duration", "pubdate", "time_offset", "send_time", "danmaku_got", "part_count"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a["status"] == "ok") & a[["pubdate", "duration", "send_time"]].notna().all(axis=1) & (a["duration"] > 0)]
    a = a[~(a["part_count"].fillna(0) >= 2)]

    g = a.groupby(["bvid", "batch"])
    tmp = g.agg(dur=("duration", "first"), got=("danmaku_got", "first"), pub=("pubdate", "first"),
                maxsend=("send_time", "max"), tname=("tname", "first"))
    tmp["age_h"] = (tmp["maxsend"] - tmp["pub"]) / 3600.0
    best = tmp.reset_index().sort_values("age_h", ascending=False).drop_duplicates("bvid")  # 每 bvid 取时龄最长快照

    recs = []
    for _, r in best.iterrows():
        bv, bt = r.bvid, int(r.batch)
        s = a[(a.bvid == bv) & (a.batch == bt)]
        dur = float(r.dur); off = s["time_offset"].values.astype(float)
        os_ = float(off.min()) if len(off) else dur
        cv = seg_cv(off, dur, os_); ra = robust_amp(off, dur, os_); zs = zero_share(off, dur, os_)
        n3 = len(off)
        seg3 = []
        for i in range(3):
            lo, hi = dur * i / 3, dur * (i + 1) / 3
            dens = float(((off >= lo) & (off < hi)).sum()) / max((hi - lo) / 60, 1e-9)
            seg3.append(dens)
        recs.append(dict(bvid=bv, batch=bt, tname=str(r.tname), dur=dur,
                         got=int(r.got), n=n3, dpm=n3 / (dur / 60), age_h=round(float(r.age_h), 1),
                         obs_start_s=os_, obs_start_frac=os_ / dur,
                         S1=cv, S2=ra, S3=zs, dens=[round(x, 2) for x in seg3]))
    df = pd.DataFrame(recs)
    df["grp"] = np.where(df.got < 600, "G1_got<600", "G2_got>=600")
    df["durbin"] = pd.cut(df.dur, [0, 600, 1800, 1e9], labels=["<10min", "10-30min", ">30min"])

    def boot_med_diff(x, y, B=2000, seed=SEED):
        rng = np.random.default_rng(seed)
        xs = np.array(x, float); ys = np.array(y, float)
        ds = []
        for _ in range(B):
            ds.append(np.median(rng.choice(xs, len(xs), True)) - np.median(rng.choice(ys, len(ys), True)))
        return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

    out = {"n_units": int(len(tmp)), "n_videos": int(len(df)),
           "got_dist": {k: int(v) for k, v in df.got.describe().round(0).items()}}
    res = {}
    for grp, sub in df.groupby("grp"):
        s1 = sub.S1.dropna()
        res[grp] = dict(n=int(len(sub)),
                        dur_med=round(float(sub.dur.median()), 0),
                        dpm_med=round(float(sub.dpm.median()), 2),
                        got_med=int(sub.got.median()),
                        obs_start_frac_p50=round(float(sub.obs_start_frac.median()), 3),
                        obs_start_frac_p90=round(float(sub.obs_start_frac.quantile(0.9)), 3),
                        full_timeline_share=round(float((sub.obs_start_s < 60).mean()), 3),
                        S1_med=round(float(s1.median()), 3), S1_q=[round(float(s1.quantile(q)), 3) for q in (.25, .75)],
                        S2_med=round(float(sub.S2.median()), 3),
                        S3_med=round(float(sub.S3.dropna().median()), 3),
                        share_S1_ge_0p5=round(float((s1 >= 0.5).mean()), 3),
                        share_S1_ge_1p0=round(float((s1 >= 1.0).mean()), 3),
                        n3_med=[round(x, 2) for x in np.median(np.array(sub.dens.tolist()), axis=0)])
    out["groups"] = res
    # Q1 效应量（S1 中位差 G2-G1）+ 分层（时长桶）
    x1 = df[df.grp == "G1_got<600"].S1.dropna(); x2 = df[df.grp == "G2_got>=600"].S1.dropna()
    lo, hi_ = boot_med_diff(x2, x1)
    out["Q1_S1_meddiff_G2_minus_G1"] = dict(diff=round(float(np.median(x2) - np.median(x1)), 3), ci=[round(lo, 3), round(hi_, 3)])
    strat = {}
    for db, sub in df.dropna(subset=["S1"]).groupby("durbin", observed=True):
        a_ = sub[sub.grp == "G1_got<600"].S1; b_ = sub[sub.grp == "G2_got>=600"].S1
        if len(a_) >= 10 and len(b_) >= 10:
            strat[str(db)] = dict(n1=int(len(a_)), n2=int(len(b_)),
                                  med1=round(float(a_.median()), 3), med2=round(float(b_.median()), 3))
    out["Q1_stratified_by_durbin"] = strat
    out["tname_top"] = df.tname.value_counts().head(5).to_dict()
    # Q2 可行集
    full = df[df.obs_start_s < 60]
    out["Q2"] = dict(
        full_timeline_n=int(len(full)),
        full_rich_n=int(((full.S1 >= 1.0)).sum()), full_rich_n_05=int((full.S1 >= 0.5).sum()),
        obswin_n=int((df.obs_start_s >= 60).sum()),
        obswin_win_ge300s_n=int(((df.obs_start_s >= 60) & ((df.dur - df.obs_start_s) >= 300)).sum()),
        obswin_rich_n=int(((df.obs_start_s >= 60) & (df.S1 >= 1.0)).sum()),
        got_ge600_full_share=round(float((df[(df.grp == "G2_got>=600")].obs_start_s < 60).mean()), 3),
        got_lt600_full_share=round(float((df[(df.grp == "G1_got<600")].obs_start_s < 60).mean()), 3))
    json.dump(out, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))

    # 图 1：S1 ECDF 两组
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for grp, c in [("G1_got<600", "#4a7ddf"), ("G2_got>=600", "#c93d68")]:
        v = np.sort(df[df.grp == grp].S1.dropna().values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, label=f"{grp} (n={len(v)})")
    ax.axvline(0.5, ls="--", c="#999", lw=1); ax.axvline(1.0, ls="--", c="#999", lw=1)
    ax.set_xlabel("段均值变异系数 S1（观测窗内，K=5 段）"); ax.set_ylabel("累计比例")
    ax.set_title("两组节奏变化程度（S1）的 ECDF——虚线为“明显变化”双阈值")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig1_S1_ecdf.png", dpi=130); plt.close(fig)
    # 图 2：obs_start_frac 分布
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for grp, c in [("G1_got<600", "#4a7ddf"), ("G2_got>=600", "#c93d68")]:
        v = np.sort(df[df.grp == grp].obs_start_frac.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, label=f"{grp} (n={len(v)})")
    ax.axvline(0.05, ls="--", c="#999", lw=1)
    ax.set_xlabel("观测起点占视频时长比例 min(time_offset)/duration"); ax.set_ylabel("累计比例")
    ax.set_title("时间线完整性：观测起点分布（虚线=前段缺失 5% 线）")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/fig2_obsstart.png", dpi=130); plt.close(fig)
    print("EXP42-DONE →", OUT)


if __name__ == "__main__":
    main()
