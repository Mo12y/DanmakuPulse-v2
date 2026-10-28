# -*- coding: utf-8 -*-
"""exp43 · got>=600 组时间线完整性窄核验 + 合格集 5 支盲标材料（只读）
核验三层概念（用户裁定）：A"从片头开始观测" B"观测窗内无明显空桶" C"已证明全量采集"（只能逼近）
核验项：
  V1 空洞游程：60s 桶最大零游程、>=3min 空洞存在率、>=5min 存在率；10s 桶复核
  V2 got 堆积：600 倍数计数（触顶嫌疑）；got>=9600 计数
  V3 got/reported 比值分层 × 空洞
  V4 send_time 三重校验：min(send_time)-pubdate 分布；|send_time-pubdate-offset|<=2s 行占比；首/末 60s 密度
合格集（预注册）：got>=600 & duration>=300s & 60s 桶无>=3min 零游程 & 10s 桶无>=5min 零游程 & got<9600
选样：S1 三分位 T3=2 支 / T2=2 支 / T1=1 支，跨分区，seed=20261027
"""
import glob
import json
import os
import random
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp43_timeline_verify"
SEED = 20261027
os.makedirs(OUT, exist_ok=True)
COLS = ["bvid", "title", "tname", "duration", "pubdate", "time_offset", "send_time",
        "mode", "text", "danmaku_got", "danmaku_reported", "part_count", "status"]


def max_zero_run(c):
    best = cur = 0
    for v in c:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


def holes(off, dur):
    lo, hi = 0, int(dur // 60)
    b = (np.clip(off, 0, None) // 60).astype(int)
    c60 = np.bincount(b[(b >= 0) & (b < hi)] - 0, minlength=hi)
    b10 = (np.clip(off, 0, None) // 10).astype(int)
    h10 = int(dur // 10)
    c10 = np.bincount(b10[(b10 >= 0) & (b10 < h10)] - 0, minlength=h10)
    r60 = max_zero_run(c60) * 60
    r10 = max_zero_run(c10) * 10
    return dict(run60_s=int(r60), hole3min=bool(r60 >= 180), hole5min=bool(r60 >= 300),
                run10_s=int(r10), hole5min_10s=bool(r10 >= 300), n60=int(hi), n10=int(h10))


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
    for c in ["duration", "pubdate", "time_offset", "send_time", "danmaku_got", "danmaku_reported", "part_count"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a["status"] == "ok") & a[["pubdate", "duration", "send_time"]].notna().all(axis=1) & (a["duration"] > 0)]
    a = a[~(a["part_count"].fillna(0) >= 2)]

    g = a.groupby(["bvid", "batch"])
    tmp = g.agg(dur=("duration", "first"), got=("danmaku_got", "first"), rep=("danmaku_reported", "first"),
                pub=("pubdate", "first"), maxsend=("send_time", "max"), tname=("tname", "first"),
                title=("title", "first"))
    tmp["age_h"] = (tmp["maxsend"] - tmp["pub"]) / 3600.0
    G = tmp.reset_index()
    G = G[G.got >= 600].sort_values("age_h", ascending=False).drop_duplicates("bvid")
    R = {"n_got_ge600_videos": int(len(G))}

    # V1/V3/V4 逐视频
    recs, got_vals = [], []
    for _, r in G.iterrows():
        bv, bt = r.bvid, int(r.batch)
        s = a[(a.bvid == bv) & (a.batch == bt)]
        dur = float(r.dur); pub = float(r.pub)
        off = s["time_offset"].values.astype(float)
        st = s["send_time"].values.astype(float)
        h = holes(off, dur)
        gap_st = float(st.min() - pub) if len(st) else None
        mis = float(np.mean(np.abs(st - pub - off) <= 2)) if len(st) else None
        d60 = dur / 60
        dens_first = float(((off >= 0) & (off < 60)).sum()) / 1.0
        dens_last = float(((off >= dur - 60) & (off <= dur)).sum()) / 1.0
        recs.append(dict(bvid=bv, batch=bt, tname=str(r.tname), dur=dur, got=int(r.got),
                         rep=int(r.rep) if pd.notna(r.rep) else None, age_h=round(float(r.age_h), 1),
                         ratio=round(float(r.got / r.rep), 3) if pd.notna(r.rep) and r.rep > 0 else None,
                         st_gap_first_s=round(gap_st, 1) if gap_st is not None else None,
                         st_offset_mismatch_share=round(1 - mis, 4) if mis is not None else None,
                         dens_first60=dens_first, dens_last60=dens_last, **h))
        got_vals.append(int(r.got))
    D = pd.DataFrame(recs)
    # V2 got 堆积
    R["got_multiples_of_600"] = {str(k): int(v) for k, v in Counter(D.got // 600 * 600 if False else D[D.got % 600 == 0].got).items()}
    R["got_at_9600"] = int((D.got >= 9600).sum())
    # V1 汇总
    R["V1_holes"] = dict(hole3min_share=round(float(D.hole3min.mean()), 4),
                         hole5min_share=round(float(D.hole5min.mean()), 4),
                         hole5min_10s_share=round(float(D.hole5min_10s.mean()), 4),
                         run60_p50=round(float(D.run60_s.median()), 1), run60_p90=round(float(D.run60_s.quantile(.9)), 1),
                         run60_max=int(D.run60_s.max()),
                         n_videos_with_any_hole=int((D.run60_s > 0).sum()))
    # V3 ratio 分层 × 空洞
    D["ratio_bin"] = pd.cut(D.ratio, [0, 0.5, 0.9, 1.01], labels=["<0.5", "0.5-0.9", ">=0.9"])
    R["V3_ratio"] = {str(k): dict(n=int(len(s)), hole3min_share=round(float(s.hole3min.mean()), 4),
                                  run60_p50=round(float(s.run60_s.median()), 1))
                     for k, s in D.groupby("ratio_bin", observed=True)}
    # V4 send_time
    R["V4_sendtime"] = dict(
        st_gap_first_p50=round(float(D.st_gap_first_s.dropna().median()), 1),
        st_gap_first_p90=round(float(D.st_gap_first_s.dropna().quantile(.9)), 1),
        st_gap_first_gt60s_share=round(float((D.st_gap_first_s > 60).mean()), 4),
        offset_mismatch_share_med=round(float(D.st_offset_mismatch_share.median()), 4),
        dens_first60_med=round(float(D.dens_first60.median()), 1),
        dens_last60_med=round(float(D.dens_last60.median()), 1))
    # 图
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for grp, c in [(D[D.got < 600] if False else D[D.ratio >= 0.9], "#4a7ddf"), (D[D.ratio < 0.9], "#c93d68")]:
        v = np.sort(grp.run60_s.values)
        if len(v):
            ax[0].step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c,
                       label=f"ratio{'≥0.9' if c=='#4a7ddf' else '<0.9'} (n={len(v)})")
    ax[0].axvline(180, ls="--", c="#999", lw=1)
    ax[0].set_xlabel("60s 桶最大零游程（秒）"); ax[0].set_ylabel("累计比例"); ax[0].legend()
    ax[0].set_title("got≥600 组：空洞游程 ECDF（虚线=3min）")
    ax[1].hist(D.ratio.dropna(), bins=30, color="#88a7d8")
    ax[1].set_xlabel("got / danmaku_reported"); ax[1].set_ylabel("视频数")
    ax[1].set_title("实取/标称 比值分布")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_holes_ratio.png", dpi=130); plt.close(fig)

    # 合格集（预注册）
    Q = D[(D.dur >= 300) & (~D.hole3min) & (~D.hole5min_10s) & (D.got < 9600)].copy()
    R["qualified"] = dict(n=int(len(Q)),
                          with_S1_note="S1 由 exp42 口径在核验时重算于下",
                          dur_med=round(float(Q.dur.median()), 0), got_med=int(Q.got.median()))
    # S1 重算（exp42 口径）
    s1 = []
    for _, r in Q.iterrows():
        s = a[(a.bvid == r.bvid) & (a.batch == r.batch)]
        off = s["time_offset"].values.astype(float)
        lo, hi = 0, int(r.dur // 60)
        c = np.bincount((np.clip(off, 0, None) // 60).astype(int), minlength=hi)[:hi]
        seg = np.array_split(c, 5)
        means = np.array([x.mean() for x in seg])
        s1.append(means.std() / means.mean() if means.mean() > 0 else np.nan)
    Q["S1"] = s1
    Q = Q.dropna(subset=["S1"])
    R["qualified"]["n_with_S1"] = int(len(Q))
    qs = Q.S1.quantile([1 / 3, 2 / 3])
    Q["tier"] = pd.cut(Q.S1, [-1, qs.iloc[0], qs.iloc[1], 1e9], labels=["T1_low", "T2_mid", "T3_high"])
    # 选样：S1 三分位 T3×2 / T2×2 / T1×1；分区分散（优先当前已选次数少的分区）+ got 降序前 20 内随机
    rng = random.Random(SEED)
    picks, used = [], set()
    for tier, ntier in [("T3_high", 2), ("T2_mid", 2), ("T1_low", 1)]:
        for _ in range(ntier):
            pool = Q[(Q.tier == tier) & (~Q.bvid.isin(used))]
            if pool.empty:
                break
            tn_count = Counter(p[1].tname for p in picks)
            pool = pool.assign(tn=pool.tname.map(lambda t: tn_count.get(t, 0))).sort_values("tn", ascending=True)
            row = pool.head(20).sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
            picks.append((tier, row)); used.add(row.bvid)

    # 材料（exp41 同构）
    items = []
    for tier, row in picks:
        bv, bt = row.bvid, int(row.batch)
        s = a[(a.bvid == bv) & (a.batch == bt)]
        dur = float(row.dur); pub = float(s["pubdate"].iloc[0])
        nb = int(dur // 60) + (1 if dur % 60 else 0)
        idx = np.minimum((np.clip(s["time_offset"].values, 0, None) // 60).astype(int), nb - 1)
        bins = np.bincount(idx, minlength=nb)
        emph = np.bincount(idx, weights=(s["mode"].fillna(1).values != 1).astype(float), minlength=nb)
        wins = []
        for w0 in range(0, nb, 5):
            sel = s[(s["time_offset"] >= w0 * 60) & (s["time_offset"] < min(w0 + 5, nb) * 60)]
            tc = Counter(t for t in sel["text"].dropna().astype(str) if 1 <= len(t) <= 30)
            wins.append({"win_s": w0 * 60, "top": [{"t": t, "n": n} for t, n in tc.most_common(3)]})
        items.append(dict(bvid=bv, batch=bt, title=str(s["title"].iloc[0])[:50], tname=str(row.tname),
                          duration_s=int(dur), age_h=round(float(row.age_h), 1), danmaku_got=int(row.got),
                          n_danmaku=int(len(s)), tier=tier,
                          bins=[{"c": int(bins[i]), "e": int(emph[i])} for i in range(nb)], windows=wins))
    man = dict(seed=SEED,
               sampling="got>=600 完整性合格集（60s 无>=3min 空洞 & 10s 无>=5min 空洞 & got<9600 & dur>=300s）内按 S1 三分位 T3×2/T2×2/T1×1 抽样",
               qualified_n=R["qualified"]["n_with_S1"],
               categories=["opening_gather", "steady_watch", "resonance_burst", "lull", "tail"],
               note="探索性盲标（新 5 支，got>=600 组）；不得当作正式验证结果",
               picked=[dict(bvid=i["bvid"], batch=i["batch"], tname=i["tname"], duration_s=i["duration_s"],
                            age_h=i["age_h"], got=i["danmaku_got"], tier=i["tier"]) for i in items],
               items=items)
    json.dump(man, open(f"{OUT}/sample_manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(f"{OUT}/material_data.js", "w", encoding="utf-8").write("window.MATERIAL=" + json.dumps(man, ensure_ascii=False) + ";")
    json.dump(R, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: R[k] for k in ["n_got_ge600_videos", "got_at_9600", "V1_holes", "V3_ratio", "V4_sendtime", "qualified"]},
                     ensure_ascii=False, indent=1))
    print("picked:", [(i["bvid"], i["tname"], i["duration_s"], i["tier"], i["danmaku_got"]) for i in items])
    print("EXP43-DONE")


if __name__ == "__main__":
    main()
