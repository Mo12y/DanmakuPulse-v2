# -*- coding: utf-8 -*-
"""exp67 R7 audit: independent recomputation from raw annotator CSVs + design/label verification."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon, spearmanr, friedmanchisquare

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp67_evidence_audit"
os.makedirs(OUT, exist_ok=True)
R = {}


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in sorted(glob.glob(f"{d}/*.csv"))],
                     ignore_index=True)


lab = pd.read_csv(f"{V2}/data-out/exp65_r7/r7_stratified_pairs.csv", encoding="utf-8-sig")
A = pd.read_csv("/tmp/r7audit/A.csv", encoding="utf-8-sig")
B = pd.read_csv("/tmp/r7audit/B.csv", encoding="utf-8-sig")
for n, d in (("master", lab), ("A", A), ("B", B)):
    R[f"rows_{n}"] = int(len(d))
R["same_pair_set"] = bool(set(A.pair_id) == set(B.pair_id) == set(lab.pair_id))
R["dup_pair_id"] = dict(A=int(A.pair_id.duplicated().sum()), B=int(B.pair_id.duplicated().sum()))
R["missing_scores"] = dict(A=int(pd.to_numeric(A.similar_1to5, errors="coerce").isna().sum()),
                           B=int(pd.to_numeric(B.similar_1to5, errors="coerce").isna().sum()))
R["score_domain"] = dict(A=sorted(pd.to_numeric(A.similar_1to5).unique().tolist()),
                         B=sorted(pd.to_numeric(B.similar_1to5).unique().tolist()))
R["note_counts"] = dict(A=int(A.note.fillna("").astype(str).str.strip().ne("").sum()),
                        B=int(B.note.fillna("").astype(str).str.strip().ne("").sum()))

m = lab.merge(A[["pair_id", "similar_1to5"]].rename(columns={"similar_1to5": "A"}), on="pair_id") \
       .merge(B[["pair_id", "similar_1to5"]].rename(columns={"similar_1to5": "B"}), on="pair_id")
m["A"] = pd.to_numeric(m.A); m["B"] = pd.to_numeric(m.B); m["mean"] = (m.A + m.B) / 2
R["n_pairs"] = int(len(m))
R["n_queries"] = int(m.bvid_a.nunique())
R["strata_per_query"] = m.groupby("bvid_a").stratum.nunique().value_counts().to_dict()
R["pairs_per_query"] = m.groupby("bvid_a").size().value_counts().to_dict()
R["candidate_reuse"] = dict(distinct_bvid_b=int(m.bvid_b.nunique()),
                            max_times_one_candidate=int(m.bvid_b.value_counts().max()),
                            n_candidates_used_more_than_once=int((m.bvid_b.value_counts() > 1).sum()))
R["query_equals_candidate"] = int((m.bvid_a == m.bvid_b).sum())
R["bvid_b_distinct_within_query"] = int((m.groupby("bvid_a").bvid_b.nunique() != 3).sum())

# ordering / hidden stratification
R["order"] = dict(same_order_A_master=bool(list(A.pair_id) == list(lab.pair_id)),
                  same_order_B_master=bool(list(B.pair_id) == list(lab.pair_id)),
                  same_order_A_B=bool(list(A.pair_id) == list(B.pair_id)),
                  spearman_order_A_B=round(float(spearmanr(range(len(A)), [list(B.pair_id).index(p) for p in A.pair_id]).statistic), 3))
R["adjacent_same_query_A"] = int((A.bvid_a.values[1:] == A.bvid_a.values[:-1]).sum())

# stratum means
R["stratum_means"] = {k: round(float(v), 3) for k, v in m.groupby("stratum")["mean"].mean().items()}
R["stratum_means_A"] = {k: round(float(v), 3) for k, v in m.groupby("stratum")["A"].mean().items()}
R["stratum_means_B"] = {k: round(float(v), 3) for k, v in m.groupby("stratum")["B"].mean().items()}
R["share_very_ge4"] = round(float((m[m.stratum == "very_similar"]["mean"] >= 4).mean()), 3)
R["share_dissim_le2"] = round(float((m[m.stratum == "dissimilar"]["mean"] <= 2).mean()), 3)

# reliability
R["spearman_A_B"] = round(float(spearmanr(m.A, m.B).statistic), 4)
R["spearman_A_B_p"] = float(spearmanr(m.A, m.B).pvalue)
R["exact_agreement"] = round(float((m.A == m.B).mean()), 3)
R["within1_agreement"] = round(float((np.abs(m.A - m.B) <= 1).mean()), 3)

# per-query paired test (cluster-corrected)
qs = []
for q, g in m.groupby("bvid_a"):
    v = g.loc[g.stratum == "very_similar", "mean"]
    d = g.loc[g.stratum == "dissimilar", "mean"]
    mid = g.loc[g.stratum == "middle", "mean"]
    qs.append(dict(query=q, very=float(v.mean()), middle=float(mid.mean()), dissim=float(d.mean()),
                   diff=float(v.mean() - d.mean())))
C = pd.DataFrame(qs)
R["per_query"] = C.round(3).to_dict("records")
R["n_queries_very_gt_dissim"] = int((C["diff"] > 0).sum())
R["sign_test_p"] = round(float(binomtest(int((C["diff"] > 0).sum()), len(C), 0.5, alternative="two-sided").pvalue), 5)
R["wilcoxon_p"] = round(float(wilcoxon(C.very, C.dissim, alternative="two-sided").pvalue), 5)
R["friedman_3level_p"] = round(float(friedmanchisquare(C.very, C.middle, C.dissim).pvalue), 6)
R["mean_diff"] = round(float(C["diff"].mean()), 3)
R["median_diff"] = round(float(C["diff"].median()), 3)
# cluster bootstrap CI on mean diff
rng = np.random.default_rng(20261028)
bs = [float(C["diff"].values[rng.integers(0, len(C), len(C))].mean()) for _ in range(5000)]
R["mean_diff_ci"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]

# monotonicity at query level
R["n_queries_monotone"] = int(((C.very >= C.middle) & (C.middle >= C.dissim)).sum())
R["per_annotator_queries_positive"] = {}
for w in ("A", "B"):
    k = 0
    for q, g in m.groupby("bvid_a"):
        if g.loc[g.stratum == "very_similar", w].mean() > g.loc[g.stratum == "dissimilar", w].mean():
            k += 1
    R["per_annotator_queries_positive"][w] = f"{k}/{m.bvid_a.nunique()}"

# verify stratum labels against recomputed distances on the C1 pool
units = read_dir(f"{V2}/data-out/exp56/units_csv")
units["c1"] = units.c1_eligible.astype(str).str.lower().eq("true")
cur = read_dir(f"{V2}/data-out/exp56/curves1pct_csv")
cur["bucket1pct"] = pd.to_numeric(cur.bucket1pct, errors="coerce").astype(int)
cur["cnt"] = pd.to_numeric(cur.cnt, errors="coerce")
dense = cur.pivot_table(index="bvid", columns="bucket1pct", values="cnt", aggfunc="sum").fillna(0.0)
pool = sorted(set(units[units.c1].bvid) & set(dense.index))
X = np.array([dense.loc[b].values / max(dense.loc[b].values.sum(), 1e-9) for b in pool])
idx = {b: i for i, b in enumerate(pool)}
R["pool_size"] = len(pool)

rank_recs = []
for r in m.drop_duplicates("pair_id").itertuples():
    if r.bvid_a not in idx or r.bvid_b not in idx:
        continue
    dist = np.linalg.norm(X - X[idx[r.bvid_a]], axis=1)
    order = np.argsort(dist)
    # rank of candidate among all others (0 = closest)
    rank_recs.append(dict(stratum=r.stratum, rank=int(np.where(order == idx[r.bvid_b])[0][0]), n=len(pool)))
rr = pd.DataFrame(rank_recs)
R["stratum_rank_check"] = {k: dict(n=int(len(g)), median_rank=float(g["rank"].median()),
                                   p10=float(g["rank"].quantile(.1)), p90=float(g["rank"].quantile(.9)))
                           for k, g in rr.groupby("stratum")}
R["expected_ranks"] = dict(very_similar=int(len(pool) // 5), middle=int(len(pool) // 2), dissimilar=int(len(pool) - 2))

json.dump(R, open(f"{OUT}/r7_audit.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
print("R7-DONE")
