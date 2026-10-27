# -*- coding: utf-8 -*-
"""验证：exp48 的 B2 计算是否存在广播错误；exp48.1 的 B2 是否等于手算 LOO。

用法：~/danmaku-venv/bin/python verify_b2_broadcast.py
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def exp48_l_marg(st, K):
    """逐行照抄 exp48_markov_increment.py 的 B2 实现"""
    n = len(st)
    cnts = np.bincount(st, minlength=K).astype(float)
    loo = (cnts[None, :] * 0 + cnts)[None, :].repeat(n, axis=0) - np.eye(K)[st]
    q_loo = (loo + 1.0) / (n - 1 + K)
    return np.log(q_loo[np.arange(n), st]).mean(), loo.shape, q_loo[np.arange(n), st].shape


def exp48_1_l_marg(st, K):
    """exp48.1 的 B2：t=1..n-1 上逐位置 LOO"""
    n = len(st)
    cnts = np.bincount(st, minlength=K).astype(float)
    loo = cnts[None, :] - np.eye(K)[st]
    q = (loo + 1.0) / (n - 1 + K)
    tA = np.arange(1, n)
    return np.log(q[tA, st[tA]]).mean(), loo.shape


def hand_loo(st, K):
    """暴力手算：对每个 t，删掉自己那一个计数后重算频率"""
    n = len(st)
    out = []
    for t in range(1, n):
        c = np.bincount(np.delete(st, t), minlength=K).astype(float)
        p = (c + 1.0) / ((n - 1) + K)
        out.append(np.log(p[st[t]]))
    return float(np.mean(out))


print("=" * 78)
print("A. 广播形状检查（exp48 原实现）")
st = np.array([0, 1, 2, 3, 1, 0, 2, 2, 3, 1, 0, 1])
K = 4
v48, shp_loo, shp_terms = exp48_l_marg(st, K)
print(f"  n={len(st)} K={K}")
print(f"  loo 形状 = {shp_loo}   ← 期望 (n,K)=(12,4)，实际 3 维说明发生了跨轴广播")
print(f"  q_loo[arange(n), st] 形状 = {shp_terms} ← 期望 (n,)，实际 2 维 ⇒ .mean() 把 n*K 个元素一起平均")
print(f"  exp48 B2 = {v48:.4f}")

v481, shp2 = exp48_1_l_marg(st, K)
vh = hand_loo(st, K)
print(f"  exp48.1 B2 = {v481:.4f}  (loo 形状 {shp2})")
print(f"  手算 LOO   = {vh:.4f}")
print(f"  一致？{abs(v481 - vh) < 1e-12}")

print()
print("=" * 78)
print("B. exp48 平均的到底是什么")
n = len(st)
cnts = np.bincount(st, minlength=K).astype(float)
loo = (cnts[None, :] * 0 + cnts)[None, :].repeat(n, axis=0) - np.eye(K)[st]
q = (loo + 1.0) / (n - 1 + K)
term = q[np.arange(n), st]
print("  term[t, s] = (cnts[s] - 1{st[st[t]]==s} + 1)/(n-1+K) —— 与 t 无关，只随 s 走")
print("  第一行 term[0] :", term[0])
print("  观测态那几列的值 :", np.array([(cnts[s]) / (n - 1 + K) for s in st]))
print(f"  ⇒ exp48 实际平均的是「所有 K 个状态」的 log 伪概率，含该视频从未出现的状态")
print(f"     例：S0 计数 {int(cnts[0])}，其伪概率 {(cnts[0]+1)/(n-1+K):.4f} → log {np.log((cnts[0]+1)/(n-1+K)):.4f}")

print()
print("=" * 78)
print("C. 真实数据上两种口径的差距（K=4，取一支真实视频）")
import glob, os, re
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
COLS = ["bvid", "duration", "pubdate", "time_offset", "send_time", "danmaku_got", "part_count", "status"]
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
tmp = g.agg(dur=("duration", "first"), got=("danmaku_got", "first"),
            maxsend=("send_time", "max"), pub=("pubdate", "first"))
tmp["age_h"] = (tmp["maxsend"] - tmp["pub"]) / 3600.0
G = tmp.reset_index()
G = G[(G.got >= 600) & (G.dur >= 300)].sort_values("age_h", ascending=False).drop_duplicates("bvid")
edges = np.array([20.0, 45.0, 92.0])
shown = 0
for _, r in G.iterrows():
    nb = int(float(r.dur) // 60)
    if nb < 15:
        continue
    s = a[(a.bvid == r.bvid) & (a.batch == r.batch)]
    c60 = np.bincount(np.minimum((np.clip(s["time_offset"].values.astype(float), 0, None) // 60)
                                 .astype(int), nb - 1), minlength=nb).astype(float)
    st = np.searchsorted(edges, c60, side="right")
    if len(set(st.tolist())) < 4:
        continue
    v48, _, _ = exp48_l_marg(st, 4)
    v481, _ = exp48_1_l_marg(st, 4)
    print(f"  {r.bvid}  n={nb:3d}  B2(exp48)={v48:7.4f}  B2(exp48.1)={v481:7.4f}  差={v481 - v48:+.4f}")
    shown += 1
    if shown >= 5:
        break
print("\nVERIFY-DONE")
