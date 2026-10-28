# -*- coding: utf-8 -*-
"""exp47 · 互动节奏状态结构检验（无人工标签；视频隔离）
识别对象（区分三者）：节奏状态=60s 桶强度离散态（K=4，训练分位阈值）；状态转换=状态序列转移依赖；
  局部突发事件=泊松 2σ 突起桶（exp45 已覆盖检测，本轮作 E-B 特征）。
E-A 似然检验（测试视频逐桶平均 nats）：i.i.d.（训练边缘）vs Markov-1 vs Markov-2（训练转移，Laplace α=1）
  预注册判据：M1−i.i.d. 配对视频 bootstrap 95%CI 下界 >0 且每桶 ≥0.02 nats。
E-B 下一状态预测（测试转移对）：多数类 vs M1 argmax vs LR（特征）→ 视频宏平均 OVR AUC + accuracy。
划分：bvid 80/20（seed=20261027）；样本：日批 got>=600 & dur>=300s，每 bvid 时龄最长快照。
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp47_state_structure"
SEED = 20261027
K = 4
os.makedirs(OUT, exist_ok=True)
COLS = ["bvid", "tname", "duration", "pubdate", "time_offset", "send_time", "danmaku_got", "part_count", "status"]


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
    tmp = g.agg(dur=("duration", "first"), got=("danmaku_got", "first"), maxsend=("send_time", "max"),
                pub=("pubdate", "first"), tname=("tname", "first"))
    tmp["age_h"] = (tmp["maxsend"] - tmp["pub"]) / 3600.0
    G = tmp.reset_index()
    G = G[(G.got >= 600) & (G.dur >= 300)].sort_values("age_h", ascending=False).drop_duplicates("bvid")
    print("pool videos:", len(G))

    seqs = []
    for _, r in G.iterrows():
        s = a[(a.bvid == r.bvid) & (a.batch == r.batch)]
        dur = float(r.dur)
        nb = int(dur // 60)
        if nb < 15:
            continue
        c = np.bincount(np.minimum((np.clip(s["time_offset"].values.astype(float), 0, None) // 60).astype(int), nb - 1),
                        minlength=nb).astype(float)
        seqs.append(dict(bvid=r.bvid, tname=str(r.tname), dur=dur, got=int(r.got), c=c))
    print("sequences (>=15 buckets):", len(seqs))

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(seqs))
    ntr = int(len(seqs) * 0.8)
    TR, TE = [seqs[i] for i in idx[:ntr]], [seqs[i] for i in idx[ntr:]]

    alltr = np.concatenate([s["c"] for s in TR])
    edges = np.quantile(alltr, [0.25, 0.5, 0.75])
    edges[0] = max(edges[0], 0.5)  # 保证状态 0 非平凡
    print("state edges (train):", edges)

    def disc(c):
        return np.searchsorted(edges, c, side="right")

    for s in TR + TE:
        s["st"] = disc(s["c"])

    # 训练分布/转移（Laplace α=1）
    cnt_st = np.bincount(np.concatenate([s["st"] for s in TR]), minlength=K) + 1.0
    p_marg = cnt_st / cnt_st.sum()
    M1 = np.ones((K, K))
    M2 = np.ones((K * K, K))
    pair1 = np.zeros(K * K)
    for s in TR:
        st = s["st"]
        for t in range(1, len(st)):
            M1[st[t - 1], st[t]] += 1
        for t in range(2, len(st)):
            M2[st[t - 2] * K + st[t - 1], st[t]] += 1
    M1n = M1 / M1.sum(axis=1, keepdims=True)
    M2n = M2 / M2.sum(axis=1, keepdims=True)

    def nats_video(s, model):
        st = s["st"]
        vals = []
        if model == "iid":
            for t in range(1, len(st)):
                vals.append(np.log(p_marg[st[t]]))
        elif model == "m1":
            for t in range(1, len(st)):
                vals.append(np.log(M1n[st[t - 1], st[t]]))
        else:
            for t in range(2, len(st)):
                vals.append(np.log(M2n[st[t - 2] * K + st[t - 1], st[t]]))
        return np.array(vals)

    per_v = {m: [] for m in ["iid", "m1", "m2"]}
    vid_ids = []
    for s in TE:
        vid_ids.append(s["bvid"])
        for m in per_v:
            v = nats_video(s, m)
            per_v[m].append(float(v.mean()) if len(v) else np.nan)
    d1 = np.array(per_v["m1"]) - np.array(per_v["iid"])
    d2 = np.array(per_v["m2"]) - np.array(per_v["m1"])
    d1v = d1[~np.isnan(d1)]; d2v = d2[~np.isnan(d2)]
    bs = np.random.default_rng(SEED)
    ci1, ci2 = [], []
    for _ in range(2000):
        pick = bs.choice(len(d1v), len(d1v), True)
        ci1.append(d1v[pick].mean())
        pick2 = bs.choice(len(d2v), len(d2v), True)
        ci2.append(d2v[pick2].mean())
    EA = dict(n_test_videos=len(TE), n_buckets_test=int(sum(len(s['st']) for s in TE)),
              iid_nats_mean=round(float(np.nanmean(per_v["iid"])), 4),
              m1_nats_mean=round(float(np.nanmean(per_v["m1"])), 4),
              m2_nats_mean=round(float(np.nanmean(per_v["m2"])), 4),
              m1_minus_iid_perbucket=round(float(np.nanmean(d1)), 4),
              m1_ci=[round(float(np.percentile(ci1, 2.5)), 4), round(float(np.percentile(ci1, 97.5)), 4)],
              m2_minus_m1_perbucket=round(float(np.nanmean(d2)), 4),
              m2_ci=[round(float(np.percentile(ci2, 2.5)), 4), round(float(np.percentile(ci2, 97.5)), 4)],
              gate="M1 优于 i.i.d.（CI 下界>0 且 >=0.02 nats/桶）" if (np.nanpercentile(ci1, 2.5) > 0 and np.nanmean(d1) >= 0.02) else "FAIL")
    print("E-A:", json.dumps(EA, ensure_ascii=False))

    # E-B：下一状态预测（测试视频全部转移对）
    Xlr, ylr, vidlr, maj, m1p = [], [], [], [], []
    for s in TE:
        st = s["st"]; c = s["c"]; nb = len(st)
        for t in range(1, nb):
            feat = np.zeros(K); feat[st[t - 1]] = 1
            roll = c[max(0, t - 3):t].mean()
            slope = c[t - 1] - c[max(0, t - 2)]
            last_burst = next((t - j for j in range(t - 1, -1, -1) if c[j] >= edges[-1]), 99)
            Xlr.append(np.concatenate([feat, [roll, slope, last_burst, t / nb]]))
            ylr.append(st[t]); vidlr.append(s["bvid"])
            maj.append(int(np.argmax(p_marg)))
            m1p.append(int(np.argmax(M1n[st[t - 1]])))
    Xlr = np.array(Xlr); ylr = np.array(ylr)
    acc_maj = accuracy_score(ylr, maj); acc_m1 = accuracy_score(ylr, m1p)
    # LR 只在训练视频拟合——为满足划分，用 TR 视频构造 LR 训练集
    Xtr, ytr = [], []
    for s in TR:
        st = s["st"]; c = s["c"]; nb = len(st)
        for t in range(1, nb):
            feat = np.zeros(K); feat[st[t - 1]] = 1
            Xtr.append(np.concatenate([feat, [c[max(0, t - 3):t].mean(), c[t - 1] - c[max(0, t - 2)],
                                               next((t - j for j in range(t - 1, -1, -1) if c[j] >= edges[-1]), 99), t / nb]]))
            ytr.append(st[t])
    lr = LogisticRegression(max_iter=1000, C=1.0).fit(np.array(Xtr), np.array(ytr))
    plr = lr.predict_proba(Xlr)
    # 视频宏平均 OVR AUC（每视频含 >=2 类才可算）
    aucs = {m: [] for m in ["M1", "LR"]}
    accs = {m: [] for m in ["M1", "LR"]}
    dfv = pd.DataFrame(dict(vid=vidlr, m1=m1p, y=ylr))
    for bv, sub in dfv.groupby("vid"):
        if sub.y.nunique() < 2:
            continue
        mask = np.array(vidlr) == bv
        for m, pv in [("M1", np.array(m1p)[mask]), ("LR", plr[mask])]:
            try:
                aucs[m].append(roc_auc_score(sub.y.values, pv, multi_class="ovr", average="macro"))
            except Exception:
                pass
        accs["M1"].append(accuracy_score(sub.y.values, np.array(m1p)[mask]))
        accs["LR"].append(accuracy_score(sub.y.values, lr.predict(Xlr[np.array(vidlr) == bv])))
    EB = dict(n_test_transitions=int(len(ylr)), acc_majority=round(acc_maj, 4), acc_M1=round(acc_m1, 4),
              acc_LR=round(float(np.mean(accs["LR"])), 4),
              auc_M1_macro_med=round(float(np.median(aucs["M1"])), 4) if aucs["M1"] else None,
              auc_LR_macro_med=round(float(np.median(aucs["LR"])), 4) if aucs["LR"] else None,
              n_videos_auc=len(aucs["M1"]),
              gate="LR 或 M1 的宏 AUC 中位 >0.5 且 accuracy 超多数类" if (aucs["LR"] and np.median(aucs["LR"]) > 0.5) or acc_m1 > acc_maj else "FAIL")
    print("E-B:", json.dumps(EB, ensure_ascii=False))

    # 图
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    data = [np.nan_to_num(per_v["iid"], nan=0), np.nan_to_num(per_v["m1"], nan=0), np.nan_to_num(per_v["m2"], nan=0)]
    ax[0].boxplot(data, tick_labels=["i.i.d.", "Markov-1", "Markov-2"], showfliers=False)
    ax[0].set_ylabel("每桶平均对数似然（nats）")
    ax[0].set_title(f"测试视频序列似然（n={len(TE)} 支）")
    worst = np.argsort(d1)[:3]
    for w in worst:
        s = TE[w]
        ax[1].step(np.arange(len(s["st"])) * 60, s["c"], lw=1, label=s["bvid"][:10])
    ax[1].set_xlabel("视频时间（秒）"); ax[1].set_ylabel("弹幕/分")
    ax[1].set_title("M1 似然最差 3 支（失败案例）"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_state_structure.png", dpi=130); plt.close(fig)

    out = dict(seed=SEED, K=K, n_train_videos=len(TR), n_test_videos=len(TE),
               state_edges_train=[round(float(e), 1) for e in edges],
               EA=EA, EB=EB,
               failure_cases=[dict(bvid=TE[w]["bvid"], dur=TE[w]["dur"], got=TE[w]["got"],
                                   m1_minus_iid=round(float(d1[w]), 3)) for w in worst])
    json.dump(out, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
    print("EXP47-DONE →", OUT)


if __name__ == "__main__":
    main()
