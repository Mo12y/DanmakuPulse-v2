#!/usr/bin/env python3
"""exp35 λ 分层条件分析（仅 D-train）。

复用 repeated_cv.py 的 5×5 OOF（base=−log1p_lam, ET_l10, MLP），按协议三层 λ 切分：
  λ=0 / 0<λ<1 / λ≥1
每层：视频数、点数、正负例、爆发率、层内视频级 AUC（宏平均）、pooled PR-AUC、
模型相对固定基线的配对差（bvid 聚类 bootstrap）、逐视频差值分布。

运行：~/danmaku-venv/bin/python experiments/exp35_models/lambda_strata.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "jobs"))
sys.path.insert(0, HERE)

import m2_l6_burst_risk as M  # noqa: E402
from run_models import load_dtrain  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

OUT = os.path.join(REPO, "data-out", "exp35_models")
SEEDS = (20261026, 20261027, 20261028, 20261029, 20261030)
MODELS = {"base": None, "ET_l10": None, "MLP": None}


def strata_of(lam):
    """协议三层；返回 0/1/2 与标签。"""
    lam = np.asarray(lam, dtype=float)
    idx = np.where(lam <= 0, 0, np.where(lam < 1.0, 1, 2))
    return idx


def reconstruct_points():
    """重建行表（与 load_dtrain 同序），并补出未来窗最大计数。"""
    d = load_dtrain()
    man = os.path.join(REPO, "docs", "exp35_508", "file_manifest.csv")
    seqs, _ = M.load_sequences(man, "/home/molly/projects/danmaku",
                               batches=M.D_TRAIN_BATCHES, layer="ods")
    rows = M.filter_multip(M.build_rows(seqs), "keep")
    elig = set(M.eligible_video_ids(rows))
    rows = [r for r in rows if r["bvid"] in elig]
    by_seq = {(s.bvid, s.batch): s for s in seqs}
    fut = np.empty(len(rows), dtype=np.int64)
    for i, r in enumerate(rows):
        s = by_seq[(r["bvid"], r["batch"])]
        fut[i] = int(s.c[r["t"]:r["t"] + M.H].max())
    lam = np.array([r["lam"] for r in rows], dtype=float)
    thr = np.array([r["thr"] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=int)
    vid = np.array([r["bvid"] for r in rows], dtype=str)
    # 与 OOF 顺序对齐校验
    assert np.array_equal(y, d["y"]), "y 与 OOF 顺序不一致"
    assert np.allclose(np.log1p(lam), d["X6"][:, 0]), "lam 与特征列不一致"
    assert np.array_equal(vid, d["vid"]), "vid 顺序不一致"
    return d, lam, thr, y, vid, fut


def video_auc_subset(y, p, vid, mask):
    """层内逐视频 AUC；单类视频排除并计数。"""
    per, excluded = {}, 0
    for v in np.unique(vid[mask]):
        m = mask & (vid == v)
        yy = y[m]
        if yy.min() == yy.max():
            excluded += 1
            continue
        per[v] = M.auc(yy, p[m])
    return per, excluded


def boot(deltas, B=2000, seed=20261026):
    vals = np.array([deltas[k] for k in sorted(deltas)], dtype=float)
    rng = np.random.default_rng(seed)
    reps = np.array([vals[rng.integers(0, vals.size, vals.size)].mean() for _ in range(B)])
    return float(vals.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def main():
    d, lam, thr, y, vid, fut = reconstruct_points()
    sid = strata_of(lam)
    names = ["λ=0", "0<λ<1", "λ≥1"]
    print(f"[data] points={y.size} videos={np.unique(vid).size} seeds={SEEDS}")

    # ── A. 分层概况 ────────────────────────────────────────────────
    print("\n[A] 分层概况")
    prof = []
    for k, nm in enumerate(names):
        m = sid == k
        vs = np.unique(vid[m])
        # 集中度：前 5 视频占该层点数比例
        cnt = np.array([int((m & (vid == v)).sum()) for v in vs]) if vs.size else np.array([0])
        top5 = float(np.sort(cnt)[::-1][:5].sum() / max(1, m.sum()))
        per_cls = {}
        for v in vs:
            mm = m & (vid == v)
            per_cls[v] = (int(y[mm].sum()), int(mm.sum() - y[mm].sum()))
        both = sum(1 for a, b in per_cls.values() if a >= 1 and b >= 1)
        rec = dict(stratum=nm, points=int(m.sum()), videos=int(vs.size),
                   pos=int(y[m].sum()), neg=int(m.sum() - y[m].sum()),
                   burst_rate=float(y[m].mean()) if m.sum() else None,
                   videos_with_both_classes=int(both),
                   top5_video_point_share=top5,
                   median_points_per_video=float(np.median(cnt)) if vs.size else 0.0)
        prof.append(rec)
        print(f"  {nm:7s} videos={rec['videos']:3d} points={rec['points']:5d} "
              f"pos={rec['pos']:5d} neg={rec['neg']:5d} rate={rec['burst_rate']:.3f} "
              f"both={both:3d} top5share={top5:.2f} med_pts={rec['median_points_per_video']:.0f}")

    # ── B. 分层模型比较 ────────────────────────────────────────────
    print("\n[B] 分层模型比较（层内视频宏平均 AUC；CI 为 bvid 聚类 bootstrap）")
    results = {nm: {} for nm in names}
    for k, nm in enumerate(names):
        mask = sid == k
        oof = {}
        for s in SEEDS:
            z = np.load(os.path.join(OUT, f"rcv_oof_seed{s}.npz"))
            oof.setdefault("base", []).append(-d["X6"][:, 0])
            oof.setdefault("ET_l10", []).append(z["ET_l10"])
            oof.setdefault("MLP", []).append(z["MLP"])
        per_model = {}
        for name, arrs in oof.items():
            per_video = {}
            aps, excl = [], 0
            for arr in arrs:
                per, ex = video_auc_subset(y, arr, vid, mask)
                excl = ex
                for v, a in per.items():
                    per_video.setdefault(v, []).append(a)
                if mask.any():
                    aps.append(average_precision_score(y[mask], arr[mask]))
            vm = {v: float(np.mean(a)) for v, a in per_video.items()}
            per_model[name] = dict(macro=float(np.mean(list(vm.values()))) if vm else None,
                                   per_video=vm, n_videos=len(vm), excluded=excl,
                                   pr=float(np.mean(aps)) if aps else None)
        base = per_model["base"]
        row = dict(base=dict(macro=base["macro"], pr=base["pr"],
                             n_videos=base["n_videos"], excluded=base["excluded"]))
        for name in ("ET_l10", "MLP"):
            pm = per_model[name]
            common = sorted(set(pm["per_video"]) & set(base["per_video"]))
            dl = {v: pm["per_video"][v] - base["per_video"][v] for v in common}
            delta, lo, hi = boot(dl)
            vals = np.array(list(dl.values()))
            row[name] = dict(macro=pm["macro"], pr=pm["pr"], n_videos=pm["n_videos"],
                             excluded=pm["excluded"], delta=delta, lo=lo, hi=hi,
                             median=float(np.median(vals)),
                             p10=float(np.percentile(vals, 10)),
                             p90=float(np.percentile(vals, 90)),
                             win=float((vals > 0).mean()),
                             no_difference=bool(abs(delta) < 1e-9))
        results[nm] = row
        print(f"  {nm:7s} base={base['macro']:.4f}(PR {base['pr']:.3f}, n={base['n_videos']}) "
              f"ET={row['ET_l10']['macro']:.4f}(Δ{row['ET_l10']['delta']:+.4f} "
              f"[{row['ET_l10']['lo']:+.3f},{row['ET_l10']['hi']:+.3f}]) "
              f"MLP={row['MLP']['macro']:.4f}(Δ{row['MLP']['delta']:+.4f} "
              f"[{row['MLP']['lo']:+.3f},{row['MLP']['hi']:+.3f}])")

    # ── C. 标签机制 ────────────────────────────────────────────────
    print("\n[C] 标签机制")
    mech = []
    for k, nm in enumerate(names):
        m = sid == k
        rec = dict(stratum=nm,
                   lam_min=float(lam[m].min()), lam_p50=float(np.median(lam[m])),
                   lam_max=float(lam[m].max()),
                   fut_p50=float(np.median(fut[m])), fut_p90=float(np.percentile(fut[m], 90)),
                   fut_max=int(fut[m].max()),
                   thr_min=float(thr[m].min()), thr_p50=float(np.median(thr[m])),
                   thr_max=float(thr[m].max()),
                   y_rate=float(y[m].mean()),
                   frac_fut_ge_thr=float((fut[m] >= thr[m]).mean()),
                   frac_fut_eq_thr=float((fut[m] == thr[m]).mean()),
                   frac_lambda_zero=float((lam[m] == 0).mean()) if m.any() else None)
        mech.append(rec)
        print(f"  {nm:7s} λ[{rec['lam_min']:.2f},{rec['lam_max']:.2f}] "
              f"fut_p50={rec['fut_p50']:.0f} fut_p90={rec['fut_p90']:.0f} "
              f"thr_p50={rec['thr_p50']:.2f} y={rec['y_rate']:.3f} "
              f"y=fut>=thr:{rec['frac_fut_ge_thr']:.4f} fut==thr:{rec['frac_fut_eq_thr']:.4f}")
    exact = bool(np.array_equal(y.astype(bool), (fut >= thr)))
    print(f"  标签恒等式 y ≡ 1[fut >= thr]：{exact}")

    payload = dict(seeds=list(SEEDS), folds=M.CV_FOLDS, data="D-train only",
                   definition=dict(base="score = -log1p_lam (deterministic)",
                                   strata="lambda_t: =0 / (0,1) / >=1",
                                   auc="video-macro within-stratum AUC (mean over 5 seeds)",
                                   pr="pooled point-level AP within stratum (mean over 5 seeds)",
                                   ci="bvid-clustered bootstrap over per-video mean-over-seeds"),
                   profile=prof, results=results, mechanism=mech,
                   identity_y_equals_fut_ge_thr=exact)
    with open(os.path.join(OUT, "lambda_strata.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("\nsaved:", os.path.join(OUT, "lambda_strata.json"))


if __name__ == "__main__":
    main()
