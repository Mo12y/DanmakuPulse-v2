# -*- coding: utf-8 -*-
"""exp44 · Part A：正式 30 支标注集（排除 exp43 探索 5 支）+ 材料 + 指南
准入（沿用 exp43 预注册）：got>=600 & dur>=300s & 60s 无>=3min 零游程 & 10s 无>=5min 零游程 & got<9600
分层：S1 三分位 × 时长桶（<10min/10-30/>30min）9 格配额（格占比×30，至少 2），格内随机 seed=20261027，分区分散
排除：exp43 探索 5 支
"""
import glob
import json
import os
import random
import re
from collections import Counter

import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp44_c1_algorithm"
SEED = 20261027
os.makedirs(OUT, exist_ok=True)
COLS = ["bvid", "title", "tname", "duration", "pubdate", "time_offset", "send_time",
        "mode", "text", "danmaku_got", "danmaku_reported", "part_count", "status"]
EXPLORED = {"BV1kBpi6aEDP", "BV1ZjpJ6kEqA", "BV1wAp76yEu1", "BV1LAJm6LEoE", "BV1ZSp46AEF8"}


def max_zero_run(c):
    best = cur = 0
    for v in c:
        cur = cur + 1 if v == 0 else 0
        best = max(best, cur)
    return best


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
    tmp = g.agg(dur=("duration", "first"), got=("danmaku_got", "first"), pub=("pubdate", "first"),
                maxsend=("send_time", "max"), tname=("tname", "first"), title=("title", "first"))
    tmp["age_h"] = (tmp["maxsend"] - tmp["pub"]) / 3600.0
    G = tmp.reset_index()
    G = G[(G.got >= 600)].sort_values("age_h", ascending=False).drop_duplicates("bvid")

    recs = []
    for _, r in G.iterrows():
        s = a[(a.bvid == r.bvid) & (a.batch == r.batch)]
        dur = float(r.dur); off = s["time_offset"].values.astype(float)
        hi = int(dur // 60)
        if hi < 5:
            continue
        c60 = np.bincount((np.clip(off, 0, None) // 60).astype(int), minlength=hi)[:hi]
        b10 = (np.clip(off, 0, None) // 10).astype(int); h10 = int(dur // 10)
        c10 = np.bincount(b10[(b10 >= 0) & (b10 < h10)], minlength=h10)
        if max_zero_run(c60) >= 3 or max_zero_run(c10) >= 30:
            continue
        seg = np.array_split(c60, 5)
        means = np.array([x.mean() for x in seg])
        recs.append(dict(bvid=r.bvid, batch=int(r.batch), tname=str(r.tname), title=str(r.title)[:50],
                         dur=dur, got=int(r.got), age_h=round(float(r.age_h), 1),
                         S1=float(means.std() / means.mean()) if means.mean() > 0 else np.nan))
    Q = pd.DataFrame(recs).dropna(subset=["S1"])
    Q = Q[~Q.bvid.isin(EXPLORED)].reset_index(drop=True)
    Q["durbin"] = pd.cut(Q.dur, [0, 600, 1800, 1e9], labels=["<10min", "10-30min", ">30min"])
    qs = Q.S1.quantile([1 / 3, 2 / 3])
    Q["tier"] = pd.cut(Q.S1, [-1, qs.iloc[0], qs.iloc[1], 1e9], labels=["T1_low", "T2_mid", "T3_high"])
    Q["cell"] = Q.tier.astype(str) + "|" + Q.durbin.astype(str)

    rng = random.Random(SEED)
    cell_n = Q.cell.value_counts()
    quota = {c: max(2, int(round(30 * n / len(Q)))) for c, n in cell_n.items()}
    tot = sum(quota.values())
    while tot > 30:
        c = max(quota, key=lambda k: quota[k] if quota[k] > 2 else -1)
        quota[c] -= 1; tot -= 1
    while tot < 30:
        c = max((k for k in quota if k in cell_n.index), key=lambda k: cell_n[k] - quota[k])
        quota[c] += 1; tot += 1

    picks, used = [], set()
    for cell, k in quota.items():
        pool = Q[(Q.cell == cell) & (~Q.bvid.isin(used))]
        tn_count = Counter(p["tname"] for p in picks)
        if len(pool):
            pool = pool.assign(tn=pool.tname.map(lambda t: tn_count.get(t, 0))).sort_values("tn", ascending=True)
            take = pool.head(max(k * 3, 10))
            rows = take.sample(min(k, len(take)), random_state=rng.randrange(1 << 30))
            for _, row in rows.iterrows():
                if len([p for p in picks if p["cell"] == cell]) >= k:
                    break
                picks.append(dict(bvid=row.bvid, batch=int(row.batch), tname=row.tname, title=row.title,
                                  duration_s=int(row.dur), got=int(row.got), age_h=round(float(row.age_h), 1),
                                  S1=round(float(row.S1), 3), tier=str(row.tier), durbin=str(row.durbin), cell=cell))
                used.add(row.bvid)
    # 补齐至 30
    while len(picks) < 30:
        rest = Q[~Q.bvid.isin(used)]
        if rest.empty:
            break
        tn_count = Counter(p["tname"] for p in picks)
        rest = rest.assign(tn=rest.tname.map(lambda t: tn_count.get(t, 0))).sort_values("tn", ascending=True)
        row = rest.head(20).sample(1, random_state=rng.randrange(1 << 30)).iloc[0]
        picks.append(dict(bvid=row.bvid, batch=int(row.batch), tname=row.tname, title=row.title,
                          duration_s=int(row.dur), got=int(row.got), age_h=round(float(row.age_h), 1),
                          S1=round(float(row.S1), 3), tier=str(row.tier), durbin=str(row.durbin),
                          cell=str(row.cell)))

    items = []
    for p in picks:
        s = a[(a.bvid == p["bvid"]) & (a.batch == p["batch"])]
        dur = float(s["duration"].iloc[0]); pub = float(s["pubdate"].iloc[0])
        nb = int(dur // 60) + (1 if dur % 60 else 0)
        idx = np.minimum((np.clip(s["time_offset"].values, 0, None) // 60).astype(int), nb - 1)
        bins = np.bincount(idx, minlength=nb)
        emph = np.bincount(idx, weights=(s["mode"].fillna(1).values != 1).astype(float), minlength=nb)
        wins = []
        for w0 in range(0, nb, 5):
            sel = s[(s["time_offset"] >= w0 * 60) & (s["time_offset"] < min(w0 + 5, nb) * 60)]
            tc = Counter(t for t in sel["text"].dropna().astype(str) if 1 <= len(t) <= 30)
            wins.append({"win_s": w0 * 60, "top": [{"t": t, "n": n} for t, n in tc.most_common(3)]})
        items.append(dict(bvid=p["bvid"], batch=p["batch"], title=p["title"], tname=p["tname"],
                          duration_s=int(dur), age_h=p["age_h"], danmaku_got=p["got"],
                          n_danmaku=int(len(s)), S1=p["S1"], tier=p["tier"],
                          bins=[{"c": int(bins[i]), "e": int(emph[i])} for i in range(nb)], windows=wins))

    guide = {
        "categories": {
            "opening_gather": "片头 0–2 分钟内的爆发式涌入，之后明显回落",
            "steady_watch": "随视频推进均匀出现的背景性弹幕，强度低且稳",
            "resonance_burst": "【相对判据】明显高于前后邻域的密集突起（复读/玩梗/名场面）——绝对弹幕多不算，必须有相对突起",
            "lull": "接近零弹幕的延续区间（≥3 分钟无弹幕才标冷场；短间隙不算）",
            "tail": "【收紧】位于视频最后 25% 且强度低于全片中位的低强度延续；若末段强度不低，按实际内容归类（共振/平稳），不得仅因位置靠后就标 tail"},
        "rules": ["边界允许 ±几秒粗糙度", "允许整支一个段或标「不确定」", "两人独立，互看结果即失效",
                  "完成后点顶栏「⬇ 导出」，下载 annotator_代号.json"]}

    man = dict(seed=SEED, n_target=30, qualified_pool_n=int(len(Q)),
               sampling="S1 三分位×时长桶 9 格配额（≥2/格），格内随机+分区分散；排除 exp43 探索 5 支",
               excluded_explored=sorted(EXPLORED), guide=guide,
               cell_quota={k: int(v) for k, v in quota.items()},
               picked=[{k: p[k] for k in ("bvid", "batch", "tname", "duration_s", "got", "S1", "tier", "durbin", "cell")} for p in picks],
               items=items)
    json.dump(man, open(f"{OUT}/sample30_manifest.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(f"{OUT}/material30_data.js", "w", encoding="utf-8").write("window.MATERIAL=" + json.dumps(man, ensure_ascii=False) + ";")
    print("Picked", len(picks), "videos. cells:", dict(Counter(p["cell"] for p in picks)))
    print("tnames:", dict(Counter(p["tname"] for p in picks)))
    print("S1 tiers:", dict(Counter(p["tier"] for p in picks)), "| durbins:", dict(Counter(p["durbin"] for p in picks)))
    print("SAMPLE30-DONE →", OUT)


if __name__ == "__main__":
    main()
