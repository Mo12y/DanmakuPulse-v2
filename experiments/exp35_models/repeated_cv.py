#!/usr/bin/env python3
"""exp35 重复分组交叉验证（5 个固定种子 × 5 折）：排名稳定性 + M1 折依赖诊断。

约束：
* 只用 D-train；不读 D-dev / D-conf。
* 每个种子的折划分固定（M.group_folds），所有模型共用同一样本与折。
* ENS5 在**每个外层训练折内**训练五个成员，再对验证折概率取均值；不使用既有 OOF、
  验证折标签或验证折表现选择成员/权重。
* 不确定性：bvid 为聚类单位；同一视频的 5 次重复预测先按视频取均值，再对视频重采样。

运行：PYTHONPATH=$HOME/exp-libs ~/danmaku-venv/bin/python experiments/exp35_models/repeated_cv.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "jobs"))
sys.path.insert(0, HERE)

import m2_l6_burst_risk as M  # noqa: E402
from run_models import _xgb, _cat, _scaled, load_dtrain  # noqa: E402

from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

OUT = os.path.join(REPO, "data-out", "exp35_models")
SEEDS = (20261026, 20261027, 20261028, 20261029, 20261030)   # 预先固定
FOLDS = M.CV_FOLDS
MODEL_SEED = 42

MEMBERS = {
    "ET_l10": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10,
                                           n_jobs=-1, random_state=MODEL_SEED),
    "SVM_C3": lambda: _scaled(SVC(C=3.0, gamma="scale", probability=True,
                                  random_state=MODEL_SEED)),
    "MLP": lambda: _scaled(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                         early_stopping=True, validation_fraction=0.1,
                                         random_state=MODEL_SEED)),
    "CAT_d4": lambda: _cat(depth=4),
    "XGB_d3": lambda: _xgb(max_depth=3),
}
CANDIDATES = ["ENS5", "MLP", "ET_l10", "SVM_C3", "CAT_d4", "M1"]
REFERENCE = ["M2"]          # 冻结协议主模型，仅作参照，不参与排名
ALL = CANDIDATES + REFERENCE


def m1_factory():
    return _scaled(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))


def m2_factory():
    return _scaled(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))


def run_seed(d, seed):
    y, vid = d["y"], d["vid"]
    X6, X1 = d["X6"], d["X1"]
    fmap = M.group_folds(vid, folds=FOLDS, seed=seed)
    fold_id = np.array([fmap[v] for v in vid])
    oof = {k: np.full(y.size, np.nan) for k in ALL}
    times = {k: 0.0 for k in ALL}
    coef_m1 = []
    for k in range(FOLDS):
        te = fold_id == k
        tr = ~te
        t = time.time()
        est1 = m1_factory().fit(X1[tr], y[tr])
        oof["M1"][te] = est1.predict_proba(X1[te])[:, 1]
        coef_m1.append(float(est1.named_steps["clf"].coef_[0, 0]))
        times["M1"] += time.time() - t

        t = time.time()
        est2 = m2_factory().fit(X6[tr], y[tr])
        oof["M2"][te] = est2.predict_proba(X6[te])[:, 1]
        times["M2"] += time.time() - t

        probs = {}
        for name, fac in MEMBERS.items():
            t = time.time()
            est = fac().fit(X6[tr], y[tr])
            probs[name] = est.predict_proba(X6[te])[:, 1]
            times[name] = times.get(name, 0.0) + (time.time() - t)
            if name in oof:
                oof[name][te] = probs[name]
        oof["ENS5"][te] = np.mean([probs[m] for m in MEMBERS], axis=0)
    times["ENS5"] = sum(times[m] for m in MEMBERS)
    return oof, times, coef_m1


def metrics(y, p, vid):
    per = M.macro_within_video_auc(y, p, vid)
    arr = np.array(list(per["per_video"].values()), dtype=float)
    return dict(macro=float(per["macro"]), per_video=per["per_video"],
                pr=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)),
                auc_p10=float(np.percentile(arr, 10)),
                auc_p50=float(np.percentile(arr, 50)),
                auc_p90=float(np.percentile(arr, 90)),
                share_below_half=float((arr < 0.5).mean()))


def cluster_bootstrap(deltas, B=2000, seed=20261026):
    vals = np.array([deltas[k] for k in sorted(deltas)], dtype=float)
    rng = np.random.default_rng(seed)
    reps = np.array([vals[rng.integers(0, vals.size, vals.size)].mean() for _ in range(B)])
    return dict(delta=float(vals.mean()), lo=float(np.percentile(reps, 2.5)),
                hi=float(np.percentile(reps, 97.5)), n_videos=int(vals.size))


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load_dtrain()
    y, vid = d["y"], d["vid"]
    print(f"[data] points={d['n']} videos={d['n_vid']} pos={d['pos']} "
          f"seeds={SEEDS} folds={FOLDS}")

    per_seed, coefs = {}, {}
    for s in SEEDS:
        t0 = time.time()
        oof, times, coef_m1 = run_seed(d, s)
        res = {k: metrics(y, oof[k], vid) for k in ALL}
        per_seed[s] = dict(results=res, times=times)
        coefs[s] = coef_m1
        np.savez(os.path.join(OUT, f"rcv_oof_seed{s}.npz"), **{k: oof[k] for k in ALL})
        print(f"  seed {s}: " + " ".join(f"{k}={res[k]['macro']:.4f}" for k in ALL)
              + f"  ({time.time()-t0:.1f}s)  M1coef_signs="
              + "".join("+" if c > 0 else "-" for c in coef_m1))

    # 跨种子：先按视频平均，再对视频 bootstrap
    agg = {}
    for k in ALL:
        vid_auc = {}
        for v in np.unique(vid):
            vals = [per_seed[s]["results"][k]["per_video"][v] for s in SEEDS
                    if v in per_seed[s]["results"][k]["per_video"]]
            if vals:
                vid_auc[str(v)] = float(np.mean(vals))
        agg[k] = dict(macro=float(np.mean(list(vid_auc.values()))), per_video=vid_auc,
                      by_seed=[per_seed[s]["results"][k]["macro"] for s in SEEDS],
                      pr_by_seed=[per_seed[s]["results"][k]["pr"] for s in SEEDS],
                      brier_by_seed=[per_seed[s]["results"][k]["brier"] for s in SEEDS],
                      time_by_seed=[per_seed[s]["times"][k] for s in SEEDS])

    deltas = {k: {v: agg[k]["per_video"][v] - agg["M1"]["per_video"][v]
                  for v in agg[k]["per_video"]}
              for k in ALL if k != "M1"}
    deltas_ens = {k: {v: agg[k]["per_video"][v] - agg["ENS5"]["per_video"][v]
                      for v in agg[k]["per_video"]}
                  for k in CANDIDATES if k not in ("ENS5", "M1")}
    boots = {k: cluster_bootstrap(v) for k, v in deltas.items()}
    boots_ens = {k: cluster_bootstrap(v) for k, v in deltas_ens.items()}

    ranks = {s: {k: int(r) for r, k in enumerate(
        sorted(CANDIDATES, key=lambda x: -per_seed[s]["results"][x]["macro"]), 1)} for s in SEEDS}

    sig = {k: dict(median=float(np.median(list(v.values()))),
                   q10=float(np.percentile(list(v.values()), 10)),
                   q90=float(np.percentile(list(v.values()), 90)),
                   win=float(np.mean([x > 0 for x in v.values()])),
                   big=float(np.mean([abs(x) > 0.1 for x in v.values()])))
           for k, v in deltas.items()}

    payload = dict(seeds=list(SEEDS), folds=FOLDS, n_points=d["n"], n_videos=d["n_vid"],
                   data="D-train only", candidates=CANDIDATES, reference=REFERENCE,
                   m1_coef_by_seed={str(k): v for k, v in coefs.items()},
                   per_seed={str(s): dict(
                       macro={k: per_seed[s]["results"][k]["macro"] for k in ALL},
                       pr={k: per_seed[s]["results"][k]["pr"] for k in ALL},
                       brier={k: per_seed[s]["results"][k]["brier"] for k in ALL},
                       share_below_half={k: per_seed[s]["results"][k]["share_below_half"]
                                         for k in ALL},
                       times=per_seed[s]["times"]) for s in SEEDS},
                   aggregate={k: dict(macro=agg[k]["macro"], by_seed=agg[k]["by_seed"],
                                      pr_by_seed=agg[k]["pr_by_seed"],
                                      brier_by_seed=agg[k]["brier_by_seed"],
                                      time_by_seed=agg[k]["time_by_seed"]) for k in ALL},
                   deltas_vs_M1=boots, deltas_vs_ENS5=boots_ens,
                   ranks_by_seed={str(k): v for k, v in ranks.items()},
                   per_video_delta_vs_M1=sig)
    with open(os.path.join(OUT, "repeated_cv.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print("\n[macro by seed]  (M2 为参照，不参与排名)")
    for k in ALL:
        vals = agg[k]["by_seed"]
        rk = [ranks[s].get(k, "-") for s in SEEDS]
        print(f"  {k:7s} mean={agg[k]['macro']:.4f}  " + " ".join(f"{v:.4f}" for v in vals)
              + (f"  ranks={rk}" if k in CANDIDATES else ""))
    print("\n[ranks by seed]")
    for s in SEEDS:
        print("  ", s, {k: ranks[s][k] for k in sorted(CANDIDATES, key=lambda x: ranks[s][x])})
    print("\n[delta vs M1, bvid-clustered bootstrap]")
    for k in ALL:
        if k == "M1":
            continue
        b = boots[k]
        print(f"  {k:7s} Δ={b['delta']:+.4f} [{b['lo']:+.3f},{b['hi']:+.3f}]")
    print("\n[delta vs ENS5]")
    for k in CANDIDATES:
        if k in ("ENS5", "M1"):
            continue
        b = boots_ens[k]
        print(f"  {k:7s}-ENS5 Δ={b['delta']:+.4f} [{b['lo']:+.3f},{b['hi']:+.3f}]")
    print("\n[per-video delta vs M1]")
    for k, v in sig.items():
        print(f"  {k:7s} median={v['median']:+.4f} q10={v['q10']:+.3f} q90={v['q90']:+.3f} "
              f"win={v['win']:.0%} |Δ|>0.1={v['big']:.0%}")
    print("\nsaved:", os.path.join(OUT, "repeated_cv.json"))


if __name__ == "__main__":
    main()
