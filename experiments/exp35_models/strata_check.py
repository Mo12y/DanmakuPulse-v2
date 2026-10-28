#!/usr/bin/env python3
"""核查：各 λ 层内 AUC 有效视频集合与配对 bootstrap 样本是否一致；输出排除原因。

只读现有 OOF（rcv_oof_seed*.npz）与标签；不重训、不改冻结资产。
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "jobs"))
sys.path.insert(0, HERE)

import m2_l6_burst_risk as M  # noqa: E402
from lambda_strata import reconstruct_points, strata_of  # noqa: E402

OUT = os.path.join(REPO, "data-out", "exp35_models")
SEEDS = (20261026, 20261027, 20261028, 20261029, 20261030)
MODELS = ("base", "ET_l10", "MLP")


def main():
    d, lam, thr, y, vid, fut = reconstruct_points()
    sid = strata_of(lam)
    names = ["λ=0", "0<λ<1", "λ≥1"]
    report = {"strata": {}, "sets_identical_across_models": True,
              "sets_identical_across_seeds": True, "nan_scores": {}}

    for k, nm in enumerate(names):
        mask = sid == k
        vids = np.unique(vid[mask])
        # 每个视频在该层的正负例（只依赖标签，与模型无关）
        cls = {}
        for v in vids:
            m = mask & (vid == v)
            cls[str(v)] = (int(y[m].sum()), int(m.sum() - y[m].sum()))
        elig = sorted([v for v, (p, n) in cls.items() if p >= 1 and n >= 1])
        excl = {v: dict(pos=p, neg=n, reason="no_pos" if p == 0 else "no_neg")
                for v, (p, n) in cls.items() if p == 0 or n == 0}

        # 逐 seed、逐模型的实际参与集合 + 分数有限性
        per_model_sets = {mm: [] for mm in MODELS}
        nan_hits = Counter()
        for s in SEEDS:
            z = np.load(os.path.join(OUT, f"rcv_oof_seed{s}.npz"))
            scores = {"base": -d["X6"][:, 0], "ET_l10": z["ET_l10"], "MLP": z["MLP"]}
            for mm in MODELS:
                p = scores[mm]
                nan_hits[mm] += int((~np.isfinite(p[mask])).sum())
                got = []
                for v in vids:
                    m = mask & (vid == v)
                    yy, pp = y[m], p[m]
                    if yy.min() == yy.max():
                        continue
                    got.append(str(v))
                per_model_sets[mm].append(sorted(got))

        same_models = all(sorted(set(per_model_sets[mm][0])) == sorted(set(elig))
                          for mm in MODELS)
        same_seeds = all(all(sorted(s) == sorted(per_model_sets[mm][0])
                             for s in per_model_sets[mm]) for mm in MODELS)
        report["sets_identical_across_models"] &= same_models
        report["sets_identical_across_seeds"] &= same_seeds
        for mm in MODELS:
            report["nan_scores"][f"{nm}|{mm}"] = nan_hits[mm]

        report["strata"][nm] = dict(
            stratum_videos=len(vids), eligible=len(elig), excluded=len(excl),
            excluded_detail=excl,
            set_equal_all_models=same_models, set_equal_all_seeds=same_seeds,
            counts={mm: len(per_model_sets[mm][0]) for mm in MODELS})

        print(f"\n[{nm}] 层内视频={len(vids)} 可算AUC={len(elig)} 剔除={len(excl)} "
              f"| 集合跨模型一致={same_models} 跨seed一致={same_seeds} "
              f"| 非有限分数={dict(nan_hits)}")
        for v, why in sorted(excl.items()):
            print(f"    剔除 {v}: pos={why['pos']} neg={why['neg']} ({why['reason']})")

    # 配对 bootstrap 实际用的交集
    print("\n[配对 bootstrap 的样本集]")
    for k, nm in enumerate(names):
        mask = sid == k
        # 与 lambda_strata.py 相同的聚合：视频均值 -> 交集
        base_pv, et_pv, mlp_pv = {}, {}, {}
        for s in SEEDS:
            z = np.load(os.path.join(OUT, f"rcv_oof_seed{s}.npz"))
            for tag, pv, p in (("b", base_pv, -d["X6"][:, 0]),
                               ("e", et_pv, z["ET_l10"]), ("m", mlp_pv, z["MLP"])):
                for v in np.unique(vid[mask]):
                    m = mask & (vid == v)
                    if y[m].min() == y[m].max():
                        continue
                    pv.setdefault(str(v), []).append(M.auc(y[m], p[m]))
        cb, ce, cm = set(base_pv), set(et_pv), set(mlp_pv)
        print(f"  {nm:7s} base={len(cb)} ET={len(ce)} MLP={len(cm)} "
              f"| base∩ET={len(cb & ce)} base∩MLP={len(cb & cm)} "
              f"| 三者相等={(cb == ce == cm)}")

    with open(os.path.join(OUT, "strata_sample_check.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("\nsaved:", os.path.join(OUT, "strata_sample_check.json"))


if __name__ == "__main__":
    main()
