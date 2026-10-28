# -*- coding: utf-8 -*-
"""exp45 · 事件型节奏检测（无新依赖；参数预注册一次运行）
算法 e45-v1：60s 桶 log1p 强度 x；局部基线 base_i = 中位数(x[i±6])（12 桶窗）；
  突起判据 x[i] >= base_i + 2*sqrt(base_i+1)（泊松 2σ，k=2 固定）；
  事件=连续超阈桶游程；过滤：事件总计数>=5 且 总超阈量>=3；输出 [start,end,peak]。
对照 PeakBase：移动平均(3 桶)平滑 + find_peaks(prominence=0.5, distance=2)（scipy，环境已有）。
评估（exp44 口径）：事件起/止端点 vs A/B 共识边界 60s 容差贪心匹配（召回/精度/误差）；
  事件级：预测事件 vs 人工 resonance_burst 段 IoU>0.3。n=5 探索性。
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
from scipy.signal import find_peaks

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

V2 = "/home/molly/projects/danmaku-v2"
E43 = f"{V2}/experiments/exp43_timeline_verify"
OUT = f"{V2}/experiments/exp45_event_detection"
os.makedirs(OUT, exist_ok=True)
PARAMS = dict(bucket_s=60, local_window_half=6, thr_k=2, thr_form="base + 2*sqrt(base+1)",
              min_total_count=5, min_excess=3,
              peak=dict(smooth=3, prominence=0.5, distance=2))


def events_of(x):
    """x = 原始计数（非 log1p）——泊松 2σ 阈值语义在计数空间（v2 修正：v1 误用 log1p 空间致零检出）"""
    n = len(x)
    base = np.array([np.median(x[max(0, i - 6):i + 7]) for i in range(n)])
    thr = base + PARAMS["thr_k"] * np.sqrt(base + 1.0)
    over = x >= thr
    evs = []
    i = 0
    while i < n:
        if over[i]:
            j = i
            while j + 1 < n and over[j + 1]:
                j += 1
            seg = x[i:j + 1]
            total = float(np.expm1(seg).sum())
            excess = float((seg - thr[i:j + 1]).clip(0).sum())
            if total >= PARAMS["min_total_count"] and excess >= PARAMS["min_excess"]:
                evs.append(dict(start=i, end=j + 1, peak=i + int(np.argmax(seg)),
                                start_s=i * 60, end_s=(j + 1) * 60,
                                peak_s=(i + int(np.argmax(seg))) * 60,
                                total=round(total, 1), base=round(float(base[i:j + 1].mean()), 2)))
            i = j + 1
        else:
            i += 1
    return evs, base, thr


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
    COLS = ["bvid", "duration", "pubdate", "time_offset", "send_time", "part_count", "status"]
    parts = []
    for f in sorted(glob.glob(f"{V2}/data/cli_collected/hotlist_top200_202610*.csv")):
        day = int(re.search(r"(2026\d{4})", os.path.basename(f)).group(1))
        if day > 20261026:
            continue
        d = pd.read_csv(f, usecols=COLS, encoding="utf-8-sig", low_memory=False)
        d["batch"] = day
        parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    for c in ["duration", "pubdate", "time_offset", "send_time", "part_count"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a[(a["status"] == "ok") & a[["pubdate", "duration", "send_time"]].notna().all(axis=1) & (a["duration"] > 0)]

    A = {it["bvid"]: it for it in json.load(open(f"{E43}/annotator_A.json", encoding="utf-8"))["items"]}
    B = {it["bvid"]: it for it in json.load(open(f"{E43}/annotator_B.json", encoding="utf-8"))["items"]}

    def consensus(bv):
        ba = sorted(s["start_s"] for s in A[bv]["segments"][1:])
        bb = sorted(s["start_s"] for s in B[bv]["segments"][1:])
        pairs, used = [], set()
        for x in ba:
            best, bj = None, None
            for j, y in enumerate(bb):
                if j in used:
                    continue
                dd = abs(x - y)
                if dd <= 60 and (best is None or dd < best):
                    best, bj = dd, j
            if bj is not None:
                used.add(bj)
                pairs.append((x + bb[bj]) / 2)
        rb = [s for s in A[bv]["segments"] if s["category"] == "resonance_burst"]
        return sorted(pairs), rb

    rows, ev_detail = [], []
    for bv in sorted(A):
        dur = float(a[(a.bvid == bv)]["duration"].iloc[0])
        s = a[(a.bvid == bv)]
        nb = max(int(dur // 60), 12)
        x = np.bincount(np.minimum((np.clip(s["time_offset"].values.astype(float), 0, None) // 60).astype(int), nb - 1),
                        minlength=nb).astype(float)
        evs, base, thr = events_of(x)
        cons, rb = consensus(bv)
        pb = []
        for e in evs:
            if e["start_s"] > 0:
                pb.append(e["start_s"])
            if e["end_s"] < dur:
                pb.append(e["end_s"])
        pr = greedy_match(pb, cons)
        rec = len(pr) / max(1, len(cons)); prec = len(pr) / max(1, len(pb)) if pb else 0.0
        errs = [p[2] for p in pr]
        # 事件级 vs 人工 rb 段
        hit = []
        for e in evs:
            for r in rb:
                lo, hi = max(e["start_s"], r["start_s"]), min(e["end_s"], r["end_s"])
                inter = max(0.0, hi - lo)
                union = max(e["end_s"], r["end_s"]) - min(e["start_s"], r["start_s"])
                if union > 0 and inter / union > 0.3:
                    hit.append((e, r, round(inter / union, 2)))
                    break
        rows.append(dict(bvid=bv, n_cons=len(cons), n_rb_manual=len(rb), n_events=len(evs),
                         pred_bounds=len(pb), matched=len(pr), recall=round(rec, 3),
                         precision=round(prec, 3), med_err=round(float(np.median(errs)), 1) if errs else None,
                         rb_hit=len(hit), rb_hit_iou=[h[2] for h in hit]))
        ev_detail.append(dict(bvid=bv, events=[{k: e[k] for k in ("start_s", "end_s", "peak_s", "total", "base")} for e in evs],
                              manual_rb=rb, rb_hits=[{"pred": [h[0]["start_s"], h[0]["end_s"]],
                                                      "manual": [h[1]["start_s"], h[1]["end_s"]], "iou": h[2]} for h in hit]))
        # 图
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.plot(np.arange(nb) * 60, x, lw=1, color="#88a7d8", label="强度（60s 桶原始计数）")
        ax.plot(np.arange(nb) * 60, thr, lw=1, color="#e8a33d", ls=":", label="局部基线+2σ 阈值")
        for e in evs:
            ax.axvspan(e["start_s"], e["end_s"], color="#c93d68", alpha=0.18)
            ax.plot(e["peak_s"], x[e["peak"]], "v", color="#c93d68", ms=6)
        for r in rb:
            ax.axvspan(r["start_s"], r["end_s"], color="#10b981", alpha=0.12)
        ax.plot([], [], color="#c93d68", alpha=0.3, lw=6, label="预测事件")
        ax.plot([], [], color="#10b981", alpha=0.3, lw=6, label="人工共振段")
        ax.set_title(f"{bv}（{int(dur)}s）事件检测 vs 人工共振段"); ax.set_xlabel("视频时间（秒）")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(f"{OUT}/ev_{bv}.png", dpi=120); plt.close(fig)

    # PeakBase 对照：峰位落入人工 rb 段的比例
    peak_rows = []
    for bv in sorted(A):
        dur = float(a[(a.bvid == bv)]["duration"].iloc[0])
        s = a[(a.bvid == bv)]
        nb = max(int(dur // 60), 12)
        raw = np.bincount(np.minimum((np.clip(s["time_offset"].values.astype(float), 0, None) // 60).astype(int), nb - 1),
                          minlength=nb).astype(float)
        xs = np.log1p(raw)
        sm = np.convolve(xs, np.ones(PARAMS["peak"]["smooth"]) / PARAMS["peak"]["smooth"], mode="same")
        pk, _ = find_peaks(sm, prominence=PARAMS["peak"]["prominence"], distance=PARAMS["peak"]["distance"])
        rb = [sg for sg in A[bv]["segments"] if sg["category"] == "resonance_burst"]
        inr = sum(1 for p in pk if any(r["start_s"] <= p * 60 < r["end_s"] for r in rb))
        peak_rows.append(dict(bvid=bv, n_peaks=int(len(pk)), peaks_in_manual_rb=inr,
                              n_manual_rb=len(rb)))
    E44 = json.load(open(f"{V2}/experiments/exp44_c1_algorithm/eval_results.json"))
    out = dict(algo_version="e45-v1", params=PARAMS, reference="exp44 共识边界 + 人工 resonance_burst 段（IoU>0.3）",
               per_video=rows, event_detail=ev_detail,
               comparison_exp44=E44["summary"], peak_baseline=peak_rows,
               gate_note="门槛沿用任务40：召回≥0.60 且 匹配误差中位≤15s；n=5 探索性")
    summ = {}
    df = pd.DataFrame(rows)
    summ["event_v1"] = dict(recall_med=round(float(df.recall.median()), 3),
                            med_err=round(float(np.nanmedian([e for r2 in rows for e in ([r2["med_err"]] if r2["med_err"] is not None else [])])), 1) if any(r2["med_err"] is not None for r2 in rows) else None,
                            rb_hit_total=int(df.rb_hit.sum()), rb_manual_total=int(df.n_rb_manual.sum()),
                            events_total=int(df.n_events.sum()))
    out["summary"] = summ
    json.dump(out, open(f"{OUT}/eval_results.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"rows": rows, "summary": summ, "peak_baseline": peak_rows}, ensure_ascii=False, indent=1))
    print("EXP45-DONE")


if __name__ == "__main__":
    main()
