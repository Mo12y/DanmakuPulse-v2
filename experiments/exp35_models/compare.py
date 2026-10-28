"""两两比较：视频级配对 bootstrap + 简单集成（探索，非正式）。"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "jobs"))
import m2_l6_burst_risk as M  # noqa: E402
from run_models import load_dtrain  # noqa: E402

OUT = os.path.join(REPO, "data-out", "exp35_models")
PICK = [("M1", "round1_M1_LR_1feat"), ("LR6", "round1_LR_6feat"),
        ("RF", "round1_RF"), ("ET", "round1_ExtraTrees"),
        ("HGB", "round1_HistGB"), ("SVM", "round1_SVM_RBF"), ("MLP", "round1_MLP"),
        ("XGB", "round1_XGBoost"), ("LGBM", "round1_LightGBM"), ("CAT", "round1_CatBoost"),
        ("ET_l10", "round2_ET_leaf10"), ("SVM_C3", "round2_SVM_C3.0"),
        ("XGB_d3", "round2_XGB_d3"), ("CAT_d4", "round2_CAT_d4")]


def main():
    d = load_dtrain()
    y, vid = d["y"], d["vid"]
    oof = {}
    for tag, fname in PICK:
        p = os.path.join(OUT, f"oof_{fname}.npy")
        if os.path.exists(p):
            oof[tag] = np.load(p)
    print("loaded:", list(oof))

    members = [t for t in ("ET_l10", "SVM_C3", "MLP", "CAT_d4", "XGB_d3") if t in oof]
    ens = np.mean([oof[t] for t in members], axis=0)
    print("ensemble members:", members)
    oof["ENS3"] = ens

    rows = []
    base = oof["M1"]
    for tag, p in oof.items():
        m = M.macro_within_video_auc(y, p, vid)
        rec = dict(model=tag, macro=m["macro"])
        if tag != "M1":
            b = M.paired_bootstrap_delta(y, p, base, vid, B=2000, seed=M.BOOT_SEED)
            rec.update(delta=b["delta"], lo=b["ci_lo"], hi=b["ci_hi"])
        rows.append(rec)
    print("\nvs M1:")
    for r in sorted(rows, key=lambda x: -x["macro"]):
        if "delta" in r:
            print(f"  {r['model']:9s} macro={r['macro']:.4f} Δ={r['delta']:+.4f} "
                  f"[{r['lo']:+.3f},{r['hi']:+.3f}]")
        else:
            print(f"  {r['model']:9s} macro={r['macro']:.4f}  (baseline)")

    pairs = [("MLP", "ET_l10"), ("ET_l10", "CAT_d4"), ("CAT_d4", "XGB_d3"),
             ("SVM_C3", "CAT_d4"), ("ENS3", "MLP"), ("ENS3", "ET_l10"),
             ("XGB", "XGB_d3"), ("RF", "MLP")]
    print("\npairwise:")
    out = {}
    for a, b in pairs:
        if a not in oof or b not in oof:
            continue
        r = M.paired_bootstrap_delta(y, oof[a], oof[b], vid, B=2000, seed=M.BOOT_SEED)
        out[f"{a}-{b}"] = r
        print(f"  {a:6s} - {b:6s} Δ={r['delta']:+.4f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
              f"{'排除0' if r['ci_lo'] > 0 or r['ci_hi'] < 0 else '含0'}")

    with open(os.path.join(OUT, "pairwise.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(vs_m1=rows, pairwise=out), fh, ensure_ascii=False, indent=2)
    print("\nsaved:", os.path.join(OUT, "pairwise.json"))


if __name__ == "__main__":
    main()
