#!/usr/bin/env python3
"""检验 A：M1 参照异常（符号翻转）核查；检验 B：视频内中心化对照。

同一 5 个种子 × 5 折；只用 D-train；不改冻结资产。
三种特征方案：
  raw      —— 原始 6 特征
  causal   —— 4 个动态特征做**因果**视频内中心化（只用该视频此前已出现的决策点，
              首点无历史 → 偏差记为 0）；prog / duration 保持原值
  oracle   —— 用整段视频均值中心化（**含未来点，不可部署**），仅作上限诊断
固定基线：score = -log1p_lam（不受拟合系数符号影响）

运行：PYTHONPATH=$HOME/exp-libs ~/danmaku-venv/bin/python experiments/exp35_models/diag_baseline_center.py
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "jobs"))
sys.path.insert(0, HERE)

import m2_l6_burst_risk as M  # noqa: E402
from run_models import _scaled, load_dtrain  # noqa: E402

from sklearn.ensemble import ExtraTreesClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

OUT = os.path.join(REPO, "data-out", "exp35_models")
SEEDS = (20261026, 20261027, 20261028, 20261029, 20261030)
FOLDS = M.CV_FOLDS
MODEL_SEED = 42
DYN = [0, 1, 2, 3]        # log1p_lam, slope, since, censored


def builders():
    return {
        "LR": lambda: _scaled(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
        "ET_l10": lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=10,
                                               n_jobs=-1, random_state=MODEL_SEED),
        "MLP": lambda: _scaled(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                             early_stopping=True, validation_fraction=0.1,
                                             random_state=MODEL_SEED)),
    }


def center_features(X, vid, causal=True):
    """按视频中心化 4 个动态列。causal=True 只用此前点；False 用整段均值（不可部署）。"""
    Z = X.copy()
    for v in np.unique(vid):
        idx = np.flatnonzero(vid == v)          # 行内已按 t 升序
        if idx.size == 0:
            continue
        block = X[idx][:, DYN]
        if causal:
            csum = np.cumsum(block, axis=0)
            prev_sum = np.vstack([np.zeros((1, block.shape[1])), csum[:-1]])
            cnt = np.arange(idx.size)[:, None]
            mean_prev = np.divide(prev_sum, cnt, out=np.zeros_like(prev_sum), where=cnt > 0)
            Z[np.ix_(idx, DYN)] = block - mean_prev
        else:
            Z[np.ix_(idx, DYN)] = block - block.mean(axis=0, keepdims=True)
    return Z


def macro_auc(y, p, vid):
    m = M.macro_within_video_auc(y, p, vid)
    return m["macro"], m["per_video"]


def cluster_boot(deltas, B=2000, seed=20261026):
    vals = np.array([deltas[k] for k in sorted(deltas)], dtype=float)
    rng = np.random.default_rng(seed)
    reps = np.array([vals[rng.integers(0, vals.size, vals.size)].mean() for _ in range(B)])
    return dict(delta=float(vals.mean()), lo=float(np.percentile(reps, 2.5)),
                hi=float(np.percentile(reps, 97.5)))


def main():
    d = load_dtrain()
    y, vid, X = d["y"], d["vid"], d["X6"]
    Xc = center_features(X, vid, causal=True)
    Xo = center_features(X, vid, causal=False)
    schemes = {"raw": X, "causal": Xc, "oracle": Xo}
    print(f"[data] points={d['n']} videos={d['n_vid']} cause_first_point_zero="
          f"{int(sum(1 for v in np.unique(vid) if (vid == v).sum() > 0))} videos")

    # ── 检验 A：逐折 M1 系数 / 排序一致性 / 折内宏平均 ──────────────
    print("\n[A] M1 逐折：系数、与 -log1p_lam 的 Spearman、折内宏平均 AUC")
    fold_rows = []
    for s in SEEDS:
        fmap = M.group_folds(vid, folds=FOLDS, seed=s)
        fold_id = np.array([fmap[v] for v in vid])
        for f in range(FOLDS):
            te = fold_id == f
            tr = ~te
            est = _scaled(LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")).fit(
                X[tr][:, [0]], y[tr])
            coef = float(est.named_steps["clf"].coef_[0, 0])
            pred = est.predict_proba(X[te][:, [0]])[:, 1]
            rho = stats.spearmanr(pred, -X[te][:, 0]).statistic
            vids = np.unique(vid[te])
            m, _ = macro_auc(y[te], pred, vid[te])
            fold_rows.append(dict(seed=s, fold=f, coef=coef, sign="+" if coef > 0 else "-",
                                  spearman=float(rho), n_videos=int(vids.size),
                                  fold_macro=float(m)))
        sub = [r for r in fold_rows if r["seed"] == s]
        print(f"  seed {s}: signs={''.join(r['sign'] for r in sub)} "
              f"coefs={[round(r['coef'],4) for r in sub]} "
              f"fold_macro={[round(r['fold_macro'],3) for r in sub]}")

    # ── 固定基线：-log1p_lam ─────────────────────────────────────────
    base_macro, base_per = macro_auc(y, -X[:, 0], vid)
    print(f"\n[base] score=-log1p_lam 的宏平均 AUC = {base_macro:.6f}")

    # ── 检验 B：3 方案 × 3 模型 ─────────────────────────────────────
    print("\n[B] 方案 × 模型（同 5×5）")
    res = {sc: {m: [] for m in builders()} for sc in schemes}
    per_video = {sc: {m: {} for m in builders()} for sc in schemes}
    for s in SEEDS:
        fmap = M.group_folds(vid, folds=FOLDS, seed=s)
        fold_id = np.array([fmap[v] for v in vid])
        for sc, XX in schemes.items():
            oof = {m: np.full(y.size, np.nan) for m in builders()}
            for f in range(FOLDS):
                te = fold_id == f
                tr = ~te
                for name, fac in builders().items():
                    est = fac().fit(XX[tr], y[tr])
                    oof[name][te] = est.predict_proba(XX[te])[:, 1]
            for name in builders():
                m, pv = macro_auc(y, oof[name], vid)
                res[sc][name].append(float(m))
                for v, a in pv.items():
                    per_video[sc][name].setdefault(v, []).append(a)
        print(f"  seed {s}: " + "  ".join(
            f"{sc}:{name}={res[sc][name][-1]:.4f}" for sc in schemes for name in res[sc]))

    summary = {}
    for sc in schemes:
        for name in res[sc]:
            summary[f"{sc}|{name}"] = dict(mean=float(np.mean(res[sc][name])),
                                           by_seed=res[sc][name])
    for k, v in summary.items():
        print(f"  {k:14s} mean={v['mean']:.4f}  " + " ".join(f"{x:.4f}" for x in v["by_seed"]))

    # 相对固定基线与相对 raw 的配对差
    deltas = {}
    for sc in schemes:
        for name in res[sc]:
            v_auc = {v: float(np.mean(a)) for v, a in per_video[sc][name].items()}
            deltas[f"{sc}|{name}_vs_base"] = {v: v_auc[v] - base_per[v] for v in v_auc}
            if sc != "raw":
                raw = {v: float(np.mean(a)) for v, a in per_video["raw"][name].items()}
                deltas[f"{sc}|{name}_vs_raw"] = {v: v_auc[v] - raw[v] for v in v_auc}
    boots = {k: cluster_boot(v) for k, v in deltas.items()}

    print("\n[delta] vs 固定基线 -log1p_lam")
    for k, b in boots.items():
        if k.endswith("vs_base"):
            print(f"  {k:28s} Δ={b['delta']:+.4f} [{b['lo']:+.3f},{b['hi']:+.3f}]")
    print("\n[delta] causal/oracle vs raw")
    for k, b in boots.items():
        if k.endswith("vs_raw"):
            print(f"  {k:28s} Δ={b['delta']:+.4f} [{b['lo']:+.3f},{b['hi']:+.3f}]")

    payload = dict(seeds=list(SEEDS), folds=FOLDS, data="D-train only",
                   base="-log1p_lam", base_macro=float(base_macro),
                   folds_m1=fold_rows, summary=summary,
                   deltas=boots)
    with open(os.path.join(OUT, "diag_baseline_center.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print("\nsaved:", os.path.join(OUT, "diag_baseline_center.json"))


if __name__ == "__main__":
    main()
