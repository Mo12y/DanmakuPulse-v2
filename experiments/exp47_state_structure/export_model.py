# -*- coding: utf-8 -*-
"""exp47→exp46 · 状态模型参数导出（重放 exp47 训练路径；一致性自检后写 model.json）
自检：重算 edges 必须等于 exp47_state_structure/results.json 的 state_edges_train，否则中止。
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
E47 = f"{V2}/experiments/exp47_state_structure"
SEED = 20261027
K = 4
COLS = ["bvid", "duration", "pubdate", "time_offset", "send_time", "danmaku_got", "part_count", "status"]


def main():
    res47 = json.load(open(f"{E47}/results.json", encoding="utf-8"))
    ref_edges = res47["state_edges_train"]

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
                pub=("pubdate", "first"))
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
        seqs.append(dict(bvid=r.bvid, c=c))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(seqs))
    TR = [seqs[i] for i in idx[:int(len(seqs) * 0.8)]]

    alltr = np.concatenate([s["c"] for s in TR])
    edges = np.quantile(alltr, [0.25, 0.5, 0.75])
    edges[0] = max(edges[0], 0.5)
    edges_l = [round(float(e), 4) for e in edges]
    assert np.allclose(edges_l, ref_edges), f"edges 不一致: {edges_l} vs {ref_edges}"

    M1 = np.ones((K, K))
    for s in TR:
        st = np.searchsorted(edges, s["c"], side="right")
        for t in range(1, len(st)):
            M1[st[t - 1], st[t]] += 1
    M1n = M1 / M1.sum(axis=1, keepdims=True)
    p_marg = (np.bincount(np.concatenate([np.searchsorted(edges, s["c"], side="right") for s in TR]), minlength=K) + 1.0)
    p_marg = p_marg / p_marg.sum()

    model = dict(
        version="exp47-v1", K=K, edges=[float(e) for e in edges],
        edges_source="训练 276 支（80% 划分 seed=20261027）桶计数 P25/P50/P75，下界钳 0.5",
        p_marg=[round(float(x), 6) for x in p_marg],
        M1=[[float(x) for x in row] for row in M1n],
        state_names=["S0 冷", "S1 温", "S2 热", "S3 爆"],
        trained_on="日批 20261012-1026 got>=600 & dur>=300s 训练划分（276 支）；数据窗与快照一致",
        usage="线上仅用本文件参数做状态编码与下一状态概率；禁止用单视频/测试数据重新拟合",
        self_check=dict(edges_match_exp47_results=True, rows_sum_to_1=bool(np.allclose(M1n.sum(axis=1), 1.0))))
    json.dump(model, open(f"{E47}/model.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("MODEL-DONE edges=", model["edges"], "rows_sum_ok=", model["self_check"]["rows_sum_to_1"])


if __name__ == "__main__":
    main()
