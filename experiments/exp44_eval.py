# -*- coding: utf-8 -*-
"""exp44 · Part B：三个自动分割算法 + exp43 5 支共识边界评估（无调参，一次性运行）
算法（版本 e44-v1，全部在 60s 桶 log1p 强度序列上，min_segment=2 桶）：
  EqualK3   : 固定 3 等长段（基线）
  SingleCP  : 单变点，位置=段内平方误差和最小化（穷举，基线）
  BinsegBIC : 递归二分；分裂接受条件 = ΔSSE > sigma2 * log(n)（BIC 准则，c=1；无人工调参）
参考边界 = 共识：A/B 标注边界 60s 内匹配对取均值（不一致的边界不进参考）。
指标：对共识的召回（主）、预测精度、匹配对绝对误差中位、段数差。
门槛（任务40 预注册）：召回 ≥0.60 且 匹配误差中位 ≤15s。
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
E43 = f"{V2}/experiments/exp43_timeline_verify"
OUT = f"{V2}/experiments/exp44_c1_algorithm"
os.makedirs(OUT, exist_ok=True)
SEED = 20261027
COLS = ["bvid", "duration", "pubdate", "time_offset", "send_time", "part_count", "status"]
MIN_SEG = 2


def series_of(a, bv, bt, dur):
    s = a[(a.bvid == bv) & (a.batch == bt)]
    nb = int(dur // 60)
    if nb < MIN_SEG * 3:
        nb = MIN_SEG * 3
    c = np.bincount(np.minimum((np.clip(s["time_offset"].values.astype(float), 0, None) // 60).astype(int), nb - 1),
                    minlength=nb)
    return np.log1p(c.astype(float))


def equal_k3(x):
    n = len(x)
    return [i * n // 3 for i in (1, 2)]


def single_cp(x):
    n = len(x)
    best, bp = None, None
    for k in range(MIN_SEG, n - MIN_SEG + 1):
        sse = ((x[:k] - x[:k].mean()) ** 2).sum() + ((x[k:] - x[k:].mean()) ** 2).sum()
        if best is None or sse < best:
            best, bp = sse, k
    return [bp]


def _best_split(x):
    n = len(x)
    best, bp = None, None
    for k in range(MIN_SEG, n - MIN_SEG + 1):
        sse = ((x[:k] - x[:k].mean()) ** 2).sum() + ((x[k:] - x[k:].mean()) ** 2).sum()
        if best is None or sse < best:
            best, bp = sse, k
    return bp, best, ((x - x.mean()) ** 2).sum()


def binseg_bic(x, out):
    n = len(x)
    stack = [(0, n)]
    while stack:
        lo, hi = stack.pop()
        xseg = x[lo:hi]
        if len(xseg) < 2 * MIN_SEG:
            continue
        bp_rel, sse_split, sse0 = _best_split(xseg)
        if bp_rel is None:
            continue
        k = lo + bp_rel
        gain = sse0 - sse_split
        sigma2 = x.var() if x.var() > 0 else 1e-9
        if gain > sigma2 * np.log(len(x)):
            out.append(k)
            stack.append((lo, k))
            stack.append((k, hi))
    return sorted(out)


def greedy_match(pred, ref, tol=60.0):
    pairs, used = [], set()
    for p in sorted(pred):
        best, bj = None, None
        for j, q in enumerate(sorted(ref)):
            if j in used:
                continue
            d = abs(p - q)
            if d <= tol and (best is None or d < best):
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            pairs.append((p, sorted(ref)[bj], best))
    return pairs


def main():
    # 5 支探索视频的原始数据
    COLS_ALL = ["bvid", "tname", "duration", "pubdate", "time_offset", "send_time", "danmaku_got", "part_count", "status"]
    parts = []
    for f in sorted(glob.glob(f"{V2}/data/cli_collected/hotlist_top200_202610*.csv")):
        day = int(re.search(r"(2026\d{4})", os.path.basename(f)).group(1))
        if day > 20261026:
            continue
        d = pd.read_csv(f, usecols=COLS_ALL, encoding="utf-8-sig", low_memory=False)
        d["batch"] = day
        parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    for c in ["duration", "pubdate", "time_offset", "send_time", "danmaku_got", "part_count"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a["status"] == "ok") & a[["pubdate", "duration", "send_time"]].notna().all(axis=1) & (a["duration"] > 0)]

    A = {it["bvid"]: it for it in json.load(open(f"{E43}/annotator_A.json", encoding="utf-8"))["items"]}
    B = {it["bvid"]: it for it in json.load(open(f"{E43}/annotator_B.json", encoding="utf-8"))["items"]}

    def bounds(it):
        return [s["start_s"] for s in it.get("segments", [])[1:]]

    def consensus(bv):
        ba, bb = sorted(bounds(A[bv])), sorted(bounds(B[bv]))
        pairs, used = [], set()
        for x in ba:
            best, bj = None, None
            for j, y in enumerate(bb):
                if j in used:
                    continue
                d = abs(x - y)
                if d <= 60 and (best is None or d < best):
                    best, bj = d, j
            if bj is not None:
                used.add(bj)
                pairs.append((x + bb[bj]) / 2)
        return sorted(pairs), (ba, bb)

    rows, detail = [], []
    for bv in sorted(A):
        dur = float(a[(a.bvid == bv)]["duration"].iloc[0])
        x = series_of(a, bv, None, dur)
        cons, (ba, bb) = consensus(bv)
        algo = {"EqualK3": equal_k3(x), "SingleCP": single_cp(x), "BinsegBIC": binseg_bic(x, [])}
        # 边界秒：桶索引 k = 边界位于第 k 桶开始处 = k*60 秒
        for name, bks in algo.items():
            pred = [k * 60 for k in bks]
            pr = greedy_match(pred, cons)
            rec = len(pr) / max(1, len(cons))
            prec = len(pr) / max(1, len(pred))
            errs = [p[2] for p in pr]
            rows.append(dict(bvid=bv, algo=name, n_consensus=len(cons), n_pred=len(pred),
                             matched=len(pr), recall=round(rec, 3), precision=round(prec, 3),
                             med_err=round(float(np.median(errs)), 1) if errs else None,
                             nseg_diff=len(pred) + 1 - (len(cons) + 1)))
            detail.append(dict(bvid=bv, algo=name, consensus=[round(c, 1) for c in cons],
                               pred=[round(p, 1) for p in pred], matched_err=[round(e, 1) for e in errs]))
        # 图
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(np.arange(len(x)) * 60, x, lw=1, color="#88a7d8", label="log1p 强度（60s 桶）")
        colors = {"EqualK3": "#4a7ddf", "SingleCP": "#e8a33d", "BinsegBIC": "#c93d68"}
        for name, bks in algo.items():
            for k in bks:
                ax.axvline(k * 60, color=colors[name], lw=1.2, ls="--" if name != "BinsegBIC" else "-")
        for c in cons:
            ax.axvline(c, color="#10b981", lw=2, alpha=0.6)
        ax.plot([], [], color="#10b981", lw=2, label="人工共识边界")
        for name, c in colors.items():
            ax.plot([], [], color=c, ls="--" if name != "BinsegBIC" else "-", lw=1.2, label=name)
        ax.set_title(f"{bv}（{int(dur)}s）"); ax.set_xlabel("视频时间（秒）"); ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(f"{OUT}/eval_{bv}.png", dpi=120); plt.close(fig)

    df = pd.DataFrame(rows)
    summ = {}
    for name, s in df.groupby("algo"):
        summ[name] = dict(recall_med=round(float(s.recall.median()), 3),
                          med_err_overall=round(float(np.nanmedian(
                              [e for d2 in detail if d2["algo"] == name for e in d2["matched_err"]])), 1),
                          recall_per_video={r.bvid: r.recall for _, r in s.iterrows()},
                          nseg_diff_med=round(float(s.nseg_diff.median()), 1))
    gate = {name: bool(summ[name]["recall_med"] >= 0.60 and summ[name]["med_err_overall"] <= 15) for name in summ}
    out = dict(algo_version="e44-v1", params=dict(unit="60s 桶 log1p", min_seg=MIN_SEG, penalty="BIC c=1",
                                                  seed=SEED, note="一次性运行，无调参"),
               reference="exp43 A/B 共识边界（60s 内匹配对取均值；不一致边界不进参考）",
               per_video=rows, per_video_detail=detail, summary=summ, gate=gate,
               gate_note="门槛=任务40 预注册：召回≥0.60 且 匹配误差中位≤15s；n=5 为初步验证")
    json.dump(out, open(f"{OUT}/eval_results.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"summary": summ, "gate": gate}, ensure_ascii=False, indent=1))
    print("EVAL-DONE")


if __name__ == "__main__":
    main()
