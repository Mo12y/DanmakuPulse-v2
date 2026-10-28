"""稳定性与负对照：逐视频差值、符号检验、标签置换对照。"""
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "jobs"))
import m2_l6_burst_risk as M  # noqa: E402
from run_models import load_dtrain  # noqa: E402

OUT = os.path.join(REPO, "data-out", "exp35_models")


def per_video_auc(p, y, vid):
    return M.macro_within_video_auc(y, p, vid)["per_video"]


def main():
    d = load_dtrain()
    y, vid = d["y"], d["vid"]
    m1 = np.load(os.path.join(OUT, "oof_round1_M1_LR_1feat.npy"))
    a1 = per_video_auc(m1, y, vid)

    print("=== 逐视频差值 vs M1（视频为单位）===")
    for tag, f in (("MLP", "round1_MLP"), ("ET_l10", "round2_ET_leaf10"),
                   ("SVM", "round1_SVM_RBF"), ("CAT_d4", "round2_CAT_d4"),
                   ("XGB_d3", "round2_XGB_d3"), ("RF", "round1_RF")):
        p = np.load(os.path.join(OUT, f"oof_{f}.npy"))
        a2 = per_video_auc(p, y, vid)
        keys = sorted(set(a1) & set(a2))
        diff = np.array([a2[k] - a1[k] for k in keys])
        pos = int((diff > 0).sum())
        n = diff.size
        pval = stats.binomtest(pos, n, 0.5, alternative="greater").pvalue
        print(f"  {tag:8s} n={n} meanΔ={diff.mean():+.4f} medianΔ={np.median(diff):+.4f} "
              f"赢={pos}/{n} ({pos/n:.0%}) 符号检验 p={pval:.2g} "
              f"|Δ|>0.1 的视频={int((np.abs(diff) > 0.1).sum())}")

    print("\n=== 负对照：视频内标签置换后重跑 MLP（应≈0.5）===")
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X = d["X6"]
    rng = np.random.default_rng(7)
    yp = y.copy()
    for v in np.unique(vid):
        m = vid == v
        yp[m] = rng.permutation(y[m])
    oof = np.full(yp.size, np.nan)
    for f in range(M.CV_FOLDS):
        te = d["fold_id"] == f
        tr = ~te
        est = Pipeline([("sc", StandardScaler()),
                        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                              early_stopping=True, validation_fraction=0.1,
                                              random_state=42))])
        est.fit(X[tr], yp[tr])
        oof[te] = est.predict_proba(X[te])[:, 1]
    r = M.macro_within_video_auc(y, oof, vid)
    print(f"  置换标签训练 → 真标签评估：macro={r['macro']:.4f}（期望≈0.5）")


if __name__ == "__main__":
    main()
