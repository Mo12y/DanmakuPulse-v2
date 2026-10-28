# -*- coding: utf-8 -*-
"""exp48 · 状态转移的增量价值（exp47 起点；只读分析）
基线阶梯（同一测试序列、同一评分=逐桶平均 log 概率，视频级 bootstrap）：
  B1 全局 i.i.d.（训练边缘）
  B2 视频自身边际（leave-one-out：每桶用该视频其余桶的频率+Laplace）←"知道整体活跃度与状态构成"
  M1 Markov-1（训练转移矩阵）
  M2 Markov-2
核心指标 = M1 − B2（纯转移增量）；M2 − M1（记忆长度增量）。
对照 1 置换：每支测试视频 20 次桶序置换（保持状态构成、破坏顺序）→ M1 似然应回落；
  限制：置换不改变边际分布，属"顺序效应"的隔离检验，非因果断言。
对照 2 K 敏感性：K∈{3,4,5}（阈值均为训练分位数，不用测试集估计）。
失败分析：按时长桶与 got 分组的 M1−B2 增量；最差 3 支。
数据/划分与 exp47 完全一致：got>=600 & dur>=300s、>=15 桶、bvid 80/20、seed=20261027。
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
OUT = f"{V2}/experiments/exp48_markov_increment"
SEED = 20261027
os.makedirs(OUT, exist_ok=True)
COLS = ["bvid", "tname", "duration", "pubdate", "time_offset", "send_time", "danmaku_got", "part_count", "status"]


def load():
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
    tmp = g.agg(dur=("duration", "first"), got=("danmaku_got", "first"), maxsend=("send_time", "max"),
                pub=("pubdate", "first"), tname=("tname", "first"))
    tmp["age_h"] = (tmp["maxsend"] - tmp["pub"]) / 3600.0
    G = tmp.reset_index()
    G = G[(G.got >= 600) & (G.dur >= 300)].sort_values("age_h", ascending=False).drop_duplicates("bvid")
    seqs = []
    for _, r in G.iterrows():
        s = a[(a.bvid == r.bvid) & (a.batch == r.batch)]
        nb = int(float(r.dur) // 60)
        if nb < 15:
            continue
        c = np.bincount(np.minimum((np.clip(s["time_offset"].values.astype(float), 0, None) // 60).astype(int), nb - 1),
                        minlength=nb).astype(float)
        seqs.append(dict(bvid=r.bvid, tname=str(r.tname), dur=float(r.dur), got=int(r.got), c=c))
    return seqs


def run_K(seqs, K):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(seqs))
    ntr = int(len(seqs) * 0.8)
    TR, TE = [seqs[i] for i in idx[:ntr]], [seqs[i] for i in idx[ntr:]]
    alltr = np.concatenate([s["c"] for s in TR])
    edges = np.quantile(alltr, [i / K for i in range(1, K)])
    edges = np.maximum.accumulate(np.maximum(edges, 0.5))
    for s in TR + TE:
        s["st"] = np.searchsorted(edges, s["c"], side="right")
    cnt = np.bincount(np.concatenate([s["st"] for s in TR]), minlength=K) + 1.0
    p_marg = cnt / cnt.sum()
    M1 = np.ones((K, K)); M2 = np.ones((K * K, K))
    for s in TR:
        st = s["st"]
        for t in range(1, len(st)):
            M1[st[t - 1], st[t]] += 1
        for t in range(2, len(st)):
            M2[st[t - 2] * K + st[t - 1], st[t]] += 1
    M1n = M1 / M1.sum(axis=1, keepdims=True)
    M2n = M2 / M2.sum(axis=1, keepdims=True)

    res = []
    for s in TE:
        st = s["st"]; n = len(st)
        cnts = np.bincount(st, minlength=K).astype(float)
        l_iid = np.log(p_marg[st[1:]]).mean()
        # B2 视频边际 LOO：每桶用其余桶频率（含 +1 平滑）
        loo = (cnts[None, :] * 0 + cnts)[None, :].repeat(n, axis=0) - np.eye(K)[st]
        q_loo = (loo + 1.0) / (n - 1 + K)
        l_marg = np.log(q_loo[np.arange(n), st]).mean()
        l_m1 = np.log(M1n[st[:-1], st[1:]]).mean()
        l_m2 = np.log(M2n[st[:-2] * K + st[1:-1], st[2:]]).mean() if n >= 3 else np.nan
        # 置换对照：20 次桶序置换 → M1 nats
        prng = np.random.default_rng(SEED + hash(s["bvid"]) % 1000)
        shuf = []
        for _ in range(20):
            ps = prng.permutation(st)
            shuf.append(np.log(M1n[ps[:-1], ps[1:]]).mean())
        res.append(dict(bvid=s["bvid"], dur=s["dur"], got=s["got"], n=n,
                        l_iid=l_iid, l_marg=l_marg, l_m1=l_m1, l_m2=l_m2,
                        d_m1_marg=l_m1 - l_marg, d_m1_iid=l_m1 - l_iid, d_m2_m1=l_m2 - l_m1,
                        m1_shuf_mean=float(np.mean(shuf)), m1_shuf_std=float(np.std(shuf))))
    return pd.DataFrame(res), len(TR), edges


def boot_ci(diff, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    d = diff[~np.isnan(diff)]
    ms = [np.mean(rng.choice(d, len(d), True)) for _ in range(B)]
    return float(np.mean(d)), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def main():
    seqs = load()
    print("sequences:", len(seqs))
    R = {"n_videos": len(seqs)}
    for K in (3, 4, 5):
        df, ntr, edges = run_K([dict(s) for s in seqs], K)
        k = f"K{K}"
        m1m = boot_ci(df.d_m1_iid.values)
        m1b = boot_ci(df.d_m1_marg.values)
        m2m = boot_ci(df.d_m2_m1.values)
        # 置换对照（仅 K=4 主口径详细记录）
        shuf_drop = (df.m1_shuf_mean - df.m1_shuf_std * 1.0)  # 保守下界参考
        R[k] = dict(edges=[round(float(e), 1) for e in edges], n_train=ntr, n_test=int(len(df)),
                    B1_iid_med=round(float(df.l_iid.median()), 4),
                    B2_video_marg_med=round(float(df.l_marg.median()), 4),
                    M1_med=round(float(df.l_m1.median()), 4), M2_med=round(float(df.l_m2.median()), 4),
                    d_M1_vs_B2=round(float(df.d_m1_marg.mean()), 4),
                    d_M1_vs_B2_ci=[round(m1b[1], 4), round(m1b[2], 4)],
                    d_M1_vs_B1=round(float(df.d_m1_iid.mean()), 4),
                    d_M1_vs_B1_ci=[round(m1m[1], 4), round(m1m[2], 4)],
                    d_M2_vs_M1=round(float(df.d_m2_m1.mean()), 4),
                    d_M2_vs_M1_ci=[round(m2m[1], 4), round(m2m[2], 4)],
                    share_M1_gt_B2=round(float((df.d_m1_marg > 0).mean()), 3),
                    m1_shuf_mean_med=round(float(df.m1_shuf_mean.median()), 4),
                    m1_orig_med=round(float(df.l_m1.median()), 4))
    K4 = R["K4"]
    df4, _, _ = run_K([dict(s) for s in seqs], 4)
    # 分组失败分析（K=4）
    df4["durbin"] = pd.cut(df4.dur, [0, 600, 1800, 1e9], labels=["<10min", "10-30min", ">30min"])
    df4["gotbin"] = pd.cut(df4.got, [0, 1200, 2400, 1e9], labels=["1.2k", "1.2-2.4k", ">2.4k"])
    R["groups_dur"] = {str(k): dict(n=int(len(s)), d_m1_marg_med=round(float(s.d_m1_marg.median()), 4),
                                    share_pos=round(float((s.d_m1_marg > 0).mean()), 3))
                       for k, s in df4.groupby("durbin", observed=True)}
    R["groups_got"] = {str(k): dict(n=int(len(s)), d_m1_marg_med=round(float(s.d_m1_marg.median()), 4),
                                    share_pos=round(float((s.d_m1_marg > 0).mean()), 3))
                       for k, s in df4.groupby("gotbin", observed=True)}
    worst = df4.nsmallest(3, "d_m1_marg")
    R["worst3"] = worst[["bvid", "dur", "got", "d_m1_marg"]].to_dict("records")

    # 图
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].boxplot([df4.l_iid, df4.l_marg, df4.l_m1, df4.l_m2], tick_labels=["B1 全局iid", "B2 视频边际", "M1", "M2"], showfliers=False)
    ax[0].set_ylabel("每桶平均 log 概率"); ax[0].set_title("四层基线阶梯（K=4，测试视频）")
    sc = ax[1].scatter(df4.dur / 60, df4.d_m1_marg, c=df4.got, cmap="viridis", s=18, alpha=0.8)
    ax[1].axhline(0, ls="--", c="#999", lw=1)
    ax[1].set_xlabel("视频时长（分钟）"); ax[1].set_ylabel("M1−B2 每桶 nats")
    ax[1].set_title("纯转移增量 vs 时长（色=got）"); fig.colorbar(sc, ax=ax[1], label="got")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_increment.png", dpi=130); plt.close(fig)

    json.dump(R, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: R[k] for k in ["K3", "K4", "K5", "groups_dur", "groups_got", "worst3"]},
                     ensure_ascii=False, indent=1))
    print("EXP48-DONE")


if __name__ == "__main__":
    main()
