# -*- coding: utf-8 -*-
"""exp48.1 · 状态转移增量价值——评分对齐修正版（不改动、不覆盖 exp48）

为什么要修正（exp48 已定位的实现问题）：
  1) 评分位置不配对：B1/M1 在 t=1..n-1 上取均值，M2 在 t=2..n-1，而 B2（视频边际 LOO）
     在 t=0..n-1 全部 n 个位置上取均值 —— 三个均值分母不同，差值不是同位置配对差。
  2) 置换对照的每视频种子用 Python 内置 hash(bvid)，受 PYTHONHASHSEED 影响，跨进程不可复现。
  3) 置换只做描述性对照，未给出正式置换检验的统计量/零假设/p 值。

本版做法：
  · 两个评分块，各自内部严格配对（同视频、同目标位置）：
      Block A（t=1..n-1）：B1 全局 iid / B2 视频边际(LOO, 离线) / B2b 视频前缀(因果在线) / M1
      Block B（t=2..n-1）：M1 / M2
    M1 在两个块里各算一次（各自与同块基线配对）；不跨块混用差值。
  · B2 的语义写清：它用到整段视频的状态组成（含目标位置之后的桶），属**离线、条件于整段组成的
    对照基线**，不等于在线预测能力；B2b 只用到 t 之前的桶，才是可在线获得的对照。
  · 种子改为 SHA-256 派生（记录派生摘要）。
  · 正式置换检验：统计量 Θ = 测试视频 M1 每桶平均 log 概率的均值；零假设=每支测试视频内部状态
    顺序可交换；做法=各视频独立整段置换（保持状态组成不变），R=2000，p=(1+#{Θ*≥Θ})/(R+1)。
    局限：M1 参数由训练集固定，因此检验的是"训练得到的转移结构是否优于同组成的随机顺序"，
    不是因果断言。

数据与划分与 exp47/exp48 完全一致（got>=600 & dur>=300s、>=15 桶、bvid 80/20、seed=20261027），
阈值/转移矩阵只用训练视频估计。只读分析，不写生产、不改数据。
"""
import glob
import hashlib
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
OUT = f"{V2}/experiments/exp48_1_scoring_aligned"
SEED = 20261027
K_LIST = (3, 4, 5)
BOOT_B = 2000
PERM_R = 2000
COLS = ["bvid", "tname", "duration", "pubdate", "time_offset", "send_time",
        "danmaku_got", "part_count", "status"]
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- 稳定种子
def seed_of(bvid, salt):
    """SHA-256 派生（跨进程稳定）；返回 uint64 范围内的整数。"""
    h = hashlib.sha256(f"{salt}:{bvid}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % (2 ** 63 - 1)


# ---------------------------------------------------------------- 数据（与 exp48 逐行一致）
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
    a = a[(a["status"] == "ok") & a[["pubdate", "duration", "send_time"]].notna().all(axis=1)
          & (a["duration"] > 0)]
    a = a[~(a["part_count"].fillna(0) >= 2)]
    g = a.groupby(["bvid", "batch"])
    tmp = g.agg(dur=("duration", "first"), got=("danmaku_got", "first"),
                maxsend=("send_time", "max"), pub=("pubdate", "first"), tname=("tname", "first"))
    tmp["age_h"] = (tmp["maxsend"] - tmp["pub"]) / 3600.0
    G = tmp.reset_index()
    G = G[(G.got >= 600) & (G.dur >= 300)].sort_values("age_h", ascending=False).drop_duplicates("bvid")
    seqs = []
    for _, r in G.iterrows():
        s = a[(a.bvid == r.bvid) & (a.batch == r.batch)]
        nb = int(float(r.dur) // 60)
        if nb < 15:
            continue
        c = np.bincount(np.minimum((np.clip(s["time_offset"].values.astype(float), 0, None) // 60)
                                   .astype(int), nb - 1), minlength=nb).astype(float)
        seqs.append(dict(bvid=r.bvid, tname=str(r.tname), dur=float(r.dur), got=int(r.got), c=c))
    return seqs


# ---------------------------------------------------------------- 拟合（只用训练视频）
def split(seqs):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(seqs))
    ntr = int(len(seqs) * 0.8)
    return [seqs[i] for i in idx[:ntr]], [seqs[i] for i in idx[ntr:]]


def encode(TR, TE, K):
    """阈值=训练池化桶计数的等分位；单调化 + 下界 0.5。返回诊断信息。"""
    alltr = np.concatenate([s["c"] for s in TR])
    raw = np.quantile(alltr, [i / K for i in range(1, K)])
    edges = np.maximum.accumulate(np.maximum(raw, 0.5))
    # 钳位/单调化后出现重复阈值 = 某个状态区间宽度为 0（该状态只能靠并列值落位）
    tie_collapsed = int(K - 1 - np.sum(np.diff(np.concatenate([[-np.inf], edges])) > 0))
    for s in TR + TE:
        s["st"] = np.searchsorted(edges, s["c"], side="right").astype(int)
    tr_occ = np.bincount(np.concatenate([s["st"] for s in TR]), minlength=K)
    te_occ = np.bincount(np.concatenate([s["st"] for s in TE]), minlength=K)
    return edges, tr_occ, te_occ, dict(raw_edges=[float(x) for x in raw],
                                       clamped_edges=[float(x) for x in edges],
                                       monotone_forced=int(np.sum(~np.isclose(raw, edges))),
                                       tie_collapsed=tie_collapsed)


def fit(TR, K):
    """训练集估计：边缘 + M1 + M2（全部 Laplace +1）。返回概率矩阵与零计数诊断。"""
    cnt = np.bincount(np.concatenate([s["st"] for s in TR]), minlength=K).astype(float) + 1.0
    p_marg = cnt / cnt.sum()
    M1 = np.ones((K, K))
    M2 = np.ones((K * K, K))
    for s in TR:
        st = s["st"]
        for t in range(1, len(st)):
            M1[st[t - 1], st[t]] += 1
        for t in range(2, len(st)):
            M2[st[t - 2] * K + st[t - 1], st[t]] += 1
    M1n = M1 / M1.sum(axis=1, keepdims=True)
    M2n = M2 / M2.sum(axis=1, keepdims=True)
    diag = dict(
        m1_zero_cells=int(np.sum(M1 == 1.0)),           # 只有 Laplace 伪计数的格子
        m2_zero_cells=int(np.sum(M2 == 1.0)),
        m2_row_unseen=int(np.sum(M2.sum(axis=1) == 1.0 + K)),   # 整行无观测（回退均匀）
        min_p_m1=float(M1n.min()), max_p_m1=float(M1n.max()),
        min_p_m2=float(M2n.min()))
    return p_marg, M1n, M2n, diag


# ---------------------------------------------------------------- 逐视频配对评分
def score_one(s, K, p_marg, M1n, M2n):
    st = s["st"]
    n = len(st)
    cnts = np.bincount(st, minlength=K).astype(float)
    tA = np.arange(1, n)                     # Block A
    tB = np.arange(2, n)                     # Block B
    # B1 全局 iid（训练边缘）
    logB1 = np.log(p_marg[st[tA]])
    # B2 视频边际 LOO（用整段组成；排除自身计数；离线、含未来桶）
    loo = cnts[None, :] - np.eye(K)[st]
    q_loo = (loo + 1.0) / (n - 1 + K)
    logB2 = np.log(q_loo[tA, st[tA]])
    # B2b 因果前缀（只用 t 之前；t=1 时仅有 st[0]；分子为位置 0..t-1 的计数）
    pre = np.cumsum(np.eye(K)[st], axis=0)          # pre[i] = 位置 0..i 的累计
    q_pre = np.full((n, K), np.nan)
    q_pre[1:] = (pre[:-1] + 1.0) / (np.arange(1, n)[:, None] + K)
    assert np.all(np.isfinite(q_pre[tA, st[tA]])), "B2b 前缀概率非有限"
    logB2b = np.log(q_pre[tA, st[tA]])
    # M1
    logM1A = np.log(M1n[st[tA - 1], st[tA]])
    logM1B = np.log(M1n[st[tB - 1], st[tB]])
    # M2
    logM2B = np.log(M2n[st[tB - 2] * K + st[tB - 1], st[tB]])
    for v in (logB1, logB2, logB2b, logM1A, logM1B, logM2B):
        assert np.all(np.isfinite(v)), "出现非有限 log 概率"
    out = dict(bvid=s["bvid"], dur=s["dur"], got=s["got"], n=n, tname=s["tname"],
               lB1=float(logB1.mean()), lB2=float(logB2.mean()), lB2b=float(logB2b.mean()),
               lM1A=float(logM1A.mean()), lM1B=float(logM1B.mean()), lM2B=float(logM2B.mean()),
               d_M1_B1=float((logM1A - logB1).mean()),
               d_M1_B2=float((logM1A - logB2).mean()),
               d_M1_B2b=float((logM1A - logB2b).mean()),
               d_B2_B1=float((logB2 - logB1).mean()),
               d_M2_M1=float((logM2B - logM1B).mean()),
               nA=int(len(tA)), nB=int(len(tB)))
    return out


def boot_ci(d, B=BOOT_B, seed=SEED):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    rng = np.random.default_rng(seed)
    ms = [float(np.mean(rng.choice(d, len(d), True))) for _ in range(B)]
    return float(np.mean(d)), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


# ---------------------------------------------------------------- 正式置换检验
def perm_test(TE, M1n, R=PERM_R, seed_salt="exp48.1-perm"):
    """Θ = 测试视频 M1 每桶平均 log 概率的均值；H0=各视频内部顺序可交换。
    各视频独立整段置换（保持状态组成），R 次；p=(1+#{Θ*≥Θ})/(R+1)。"""
    obs_each = np.array([np.log(M1n[s["st"][:-1], s["st"][1:]]).mean() for s in TE])
    obs = float(obs_each.mean())
    seeds = [seed_of(s["bvid"], seed_salt) for s in TE]
    null = np.empty(R)
    for r in range(R):
        vals = np.empty(len(TE))
        for i, s in enumerate(TE):
            st = s["st"]
            prng = np.random.default_rng(seeds[i] + r * 7919)
            ps = prng.permutation(st)
            vals[i] = np.log(M1n[ps[:-1], ps[1:]]).mean()
        null[r] = vals.mean()
    p = (1.0 + float(np.sum(null >= obs))) / (R + 1.0)
    return dict(stat=obs, p_value=p, R=R, perm="within-video full permutation (composition preserved)",
                null_mean=float(null.mean()), null_sd=float(null.std()),
                null_p2_5=float(np.percentile(null, 2.5)), null_p97_5=float(np.percentile(null, 97.5)),
                obs_gap_sd=float((obs - null.mean()) / max(null.std(), 1e-12)),
                seed_digest=hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest()[:16])


# ---------------------------------------------------------------- 描述性置换（逐视频 20 次，稳定种子）
def shuf_descriptive(df, TE, M1n, reps=20):
    vals = []
    for s in TE:
        st = s["st"]
        prng = np.random.default_rng(seed_of(s["bvid"], "exp48.1-shuf"))
        vals.append(float(np.mean([np.log(M1n[ps[:-1], ps[1:]]).mean()
                                   for ps in (prng.permutation(st) for _ in range(reps))])))
    df["m1_shuf_mean"] = vals
    df["shuf_gap"] = df["lM1A"] - df["m1_shuf_mean"]
    return dict(orig_med=float(df.lM1A.median()), shuf_med=float(df.m1_shuf_mean.median()),
                gap_mean=float(df.shuf_gap.mean()), gap_med=float(df.shuf_gap.median()),
                share_gap_pos=float((df.shuf_gap > 0).mean()), reps_per_video=reps)


# ---------------------------------------------------------------- 主流程
def main():
    seqs = load()
    print(f"sequences={len(seqs)}", flush=True)
    R = {"meta": dict(exp="exp48.1", seed=SEED, boot_B=BOOT_B, perm_R=PERM_R,
                      n_sequences=len(seqs), generated_by="experiments/exp48_1_scoring_aligned.py",
                      fix_list=["评分位置配对(B2 由 t=0..n-1 改为 t=1..n-1)",
                                "新增因果在线基线 B2b（仅用 t 之前）",
                                "置换/抖动种子改 SHA-256（弃用内置 hash）",
                                "补正式置换检验(p 值)",
                                "补训练/测试状态占用与零计数诊断"])}
    R["data"] = dict(filter="status==ok & pubdate/duration/send_time 非空 & duration>0 & 非多P"
                            " & danmaku_got>=600 & duration>=300 & 分钟桶数>=15",
                     batch_window="20261012-20261026（与 exp47/48 相同）",
                     unit="视频（bvid 去重，取 age_h 最小的批次）",
                     split="bvid 级 80/20，seed=20261027，K=3/4/5 共用同一划分")

    frames = {}
    for K in K_LIST:
        TR, TE = split([dict(s) for s in seqs])
        edges, tr_occ, te_occ, enc_diag = encode(TR, TE, K)
        p_marg, M1n, M2n, mdiag = fit(TR, K)
        rows = [score_one(s, K, p_marg, M1n, M2n) for s in TE]
        df = pd.DataFrame(rows)
        frames[K] = df
        # 置换：正式（R 次）+ 描述性（20 次）
        perm = perm_test(TE, M1n)
        shuf = shuf_descriptive(df, TE, M1n)
        # 配对 bootstrap
        b_M1_B1 = boot_ci(df.d_M1_B1.values)
        b_M1_B2 = boot_ci(df.d_M1_B2.values)
        b_M1_B2b = boot_ci(df.d_M1_B2b.values)
        b_B2_B1 = boot_ci(df.d_B2_B1.values)
        b_M2_M1 = boot_ci(df.d_M2_M1.values)
        # B2 相对 B2b：离线信息（含未来桶）相对因果前缀多给多少
        b_B2_B2b = boot_ci((df.lB2 - df.lB2b).values)
        R[f"K{K}"] = dict(
            edges=[round(float(e), 1) for e in edges],
            n_train=len(TR), n_test=len(TE),
            n_buckets_A=int(df.nA.sum()), n_buckets_B=int(df.nB.sum()),
            encode_diag=enc_diag, train_state_occ=[int(x) for x in tr_occ],
            test_state_occ=[int(x) for x in te_occ],
            empty_state_train=int(np.sum(tr_occ == 0)), empty_state_test=int(np.sum(te_occ == 0)),
            fit_diag=mdiag,
            medians=dict(B1=round(float(df.lB1.median()), 4), B2=round(float(df.lB2.median()), 4),
                         B2b=round(float(df.lB2b.median()), 4), M1=round(float(df.lM1A.median()), 4),
                         M2=round(float(df.lM2B.median()), 4)),
            blockA=dict(
                d_M1_vs_B1=dict(mean=b_M1_B1[0], ci=[b_M1_B1[1], b_M1_B1[2]],
                                share_pos=float((df.d_M1_B1 > 0).mean())),
                d_M1_vs_B2=dict(mean=b_M1_B2[0], ci=[b_M1_B2[1], b_M1_B2[2]],
                                share_pos=float((df.d_M1_B2 > 0).mean())),
                d_M1_vs_B2b=dict(mean=b_M1_B2b[0], ci=[b_M1_B2b[1], b_M1_B2b[2]],
                                 share_pos=float((df.d_M1_B2b > 0).mean())),
                d_B2_vs_B1=dict(mean=b_B2_B1[0], ci=[b_B2_B1[1], b_B2_B1[2]]),
                d_B2_vs_B2b=dict(mean=b_B2_B2b[0], ci=[b_B2_B2b[1], b_B2_B2b[2]])),
            blockB=dict(d_M2_vs_M1=dict(mean=b_M2_M1[0], ci=[b_M2_M1[1], b_M2_M1[2]],
                                        share_pos=float((df.d_M2_M1 > 0).mean()))),
            perm_test=perm, shuf_descriptive=shuf)

    # ---- 分层（K=4 主口径） ----
    df4 = frames[4].copy()
    df4["durbin"] = pd.cut(df4.dur, [0, 600, 1800, 1e9], labels=["<10min", "10-30min", ">30min"])
    df4["gotbin"] = pd.cut(df4.got, [0, 1200, 2400, 1e9], labels=["got<=1.2k", "1.2-2.4k", ">2.4k"])
    df4["nbin"] = pd.qcut(df4.n, 3, labels=["短序列", "中序列", "长序列"])
    grp = {}
    for col in ("durbin", "gotbin", "nbin"):
        grp[col] = {str(k): dict(n=int(len(s)),
                                 d_M1_B2_mean=round(float(s.d_M1_B2.mean()), 4),
                                 d_M1_B2_med=round(float(s.d_M1_B2.median()), 4),
                                 d_M1_B2b_med=round(float(s.d_M1_B2b.median()), 4),
                                 share_pos=round(float((s.d_M1_B2 > 0).mean()), 3))
                    for k, s in df4.groupby(col, observed=True)}
    R["stratified_K4"] = grp
    worst = df4.nsmallest(5, "d_M1_B2")[["bvid", "dur", "got", "n", "d_M1_B2", "d_M1_B2b"]]
    R["worst5_K4"] = [{k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in r.items()} for r in worst.to_dict("records")]

    # ---- 与 exp48 原结果对照 ----
    try:
        old = json.load(open(f"{V2}/experiments/exp48_markov_increment/results.json", encoding="utf-8"))
        cmp = {}
        for K in K_LIST:
            k = f"K{K}"
            cmp[k] = dict(
                exp48=dict(B1_med=old[k]["B1_iid_med"], B2_med=old[k]["B2_video_marg_med"],
                           M1_med=old[k]["M1_med"], M2_med=old[k]["M2_med"],
                           d_M1_B2=old[k]["d_M1_vs_B2"], d_M1_B2_ci=old[k]["d_M1_vs_B2_ci"],
                           d_M1_B1=old[k]["d_M1_vs_B1"], d_M2_M1=old[k]["d_M2_vs_M1"]),
                exp48_1=dict(B1_med=R[k]["medians"]["B1"], B2_med=R[k]["medians"]["B2"],
                             B2b_med=R[k]["medians"]["B2b"], M1_med=R[k]["medians"]["M1"],
                             M2_med=R[k]["medians"]["M2"],
                             d_M1_B2=round(R[k]["blockA"]["d_M1_vs_B2"]["mean"], 4),
                             d_M1_B2_ci=[round(x, 4) for x in R[k]["blockA"]["d_M1_vs_B2"]["ci"]],
                             d_M1_B1=round(R[k]["blockA"]["d_M1_vs_B1"]["mean"], 4),
                             d_M2_M1=round(R[k]["blockB"]["d_M2_vs_M1"]["mean"], 4)),
                delta=dict(d_M1_B2=round(R[k]["blockA"]["d_M1_vs_B2"]["mean"]
                                         - old[k]["d_M1_vs_B2"], 4),
                           d_M1_B1=round(R[k]["blockA"]["d_M1_vs_B1"]["mean"]
                                         - old[k]["d_M1_vs_B1"], 4),
                           d_M2_M1=round(R[k]["blockB"]["d_M2_vs_M1"]["mean"]
                                         - old[k]["d_M2_vs_M1"], 4)))
        R["compare_exp48"] = cmp
    except Exception as e:                                    # noqa: BLE001
        R["compare_exp48"] = {"error": repr(e)}

    # ---- 图 ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].boxplot([df4.lB1, df4.lB2, df4.lB2b, df4.lM1A, df4.lM2B], showfliers=False,
                  tick_labels=["B1\n全局iid", "B2\n视频边际(离线)", "B2b\n视频前缀(在线)", "M1", "M2"])
    ax[0].set_ylabel("每桶平均 log 概率"); ax[0].set_title("五层对照（K=4，测试 70 支，同位置配对）")
    sc = ax[1].scatter(df4.dur / 60, df4.d_M1_B2, c=df4.got, cmap="viridis", s=18, alpha=0.85)
    ax[1].axhline(0, ls="--", c="#999", lw=1)
    ax[1].set_xlabel("视频时长（分钟）"); ax[1].set_ylabel("M1 − B2（每桶 nats）")
    ax[1].set_title("纯转移增量 vs 时长（色=got）"); fig.colorbar(sc, ax=ax[1], label="got")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_increment_aligned.png", dpi=130); plt.close(fig)

    df4.to_csv(f"{OUT}/per_video_K4.csv", index=False, encoding="utf-8-sig")
    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: R[k] for k in ("K3", "K4", "K5")}, ensure_ascii=False, indent=1)[:4000])
    print("EXP48_1_DONE")


if __name__ == "__main__":
    main()
