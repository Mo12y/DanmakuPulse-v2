#!/usr/bin/env python3
"""
kshape_validate.py — 形态聚类器②（08 §3.4：K-shape，driver 侧）+ 两法 ARI + 独立复算

运行：~/danmaku-venv/bin/python tools/kshape_validate.py   （宿主 venv——tslearn 在此，镜像无）
输入（由 m2_l1_cluster.py 导出于 data-out/）：
  cluster_curves_dense.csv   509×100 稠密 1% 曲线（缺失桶 ⇔ 0）
  cluster_kmeans_labels.csv  MLlib K-means++ 标签（主口径）
  cluster_kmeans_vectors.csv 16 维标准化向量（sklearn 复算轮廓用）
  m2_l1_cluster_report.json  MLlib 侧轮廓值（复算对拍基准）
输出：
  cluster_kshape_labels.csv  bvid,batch_date,cluster_id_kshape,k_kshape
  cluster_kshape_result.json k 网格/K 选中/ARI/sklearn 复算一致性

口径：
  - 曲线归一化 = 每视频 z-score（shape 聚类标准做法；形状可比、幅度无关）
  - K-shape k∈[3,8]（08 §3.4），n_init=3、random_state=42 → 重跑确定
  - 轮廓用 euclidean（曲线空间）；两法一致性 = adjusted_rand_score（sklearn）
  - 复算门②：sklearn silhouette（sqeuclidean，与 MLlib ClusteringEvaluator 同口径）重算
    K-means 选中 k 的轮廓值 → 与 MLlib 值对拍（容差见 RESIL_TOL，超出即报失败）
"""
import csv
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'data-out')
K_GRID = list(range(3, 9))
RESIL_TOL = 1e-4          # sklearn vs MLlib 轮廓复算容差（两实现同口径，仅浮点序差）
N_INIT = 3
SEED = 42


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    curves_rows = load_csv(os.path.join(OUT, 'cluster_curves_dense.csv'))
    label_rows = load_csv(os.path.join(OUT, 'cluster_kmeans_labels.csv'))
    vec_rows = load_csv(os.path.join(OUT, 'cluster_kmeans_vectors.csv'))
    report = json.load(open(os.path.join(OUT, 'm2_l1_cluster_report.json'), encoding='utf-8'))

    key = lambda r: (r['batch_date'], r['bvid'])
    curves_rows.sort(key=key)
    label_rows.sort(key=key)
    vec_rows.sort(key=key)
    assert all(key(a) == key(b) for a, b in zip(curves_rows, vec_rows)), '曲线/向量行对齐失败'
    n = len(curves_rows)
    print(f'输入 {n} 条曲线（{curves_rows[0]["batch_date"]}–{curves_rows[-1]["batch_date"]}）')

    # ── 曲线（稠密 → z-score 归一化）──
    b_cols = [c for c in curves_rows[0].keys() if c.startswith('b') and c[1:].isdigit()]
    b_cols.sort(key=lambda c: int(c[1:]))
    X_raw = np.array([[float(r[c]) for c in b_cols] for r in curves_rows], dtype=float)
    mu = X_raw.mean(axis=1, keepdims=True)
    sd = X_raw.std(axis=1, keepdims=True)
    sd[sd <= 0] = 1.0
    X = (X_raw - mu) / sd

    # ── sklearn 复算门②：K-means 轮廓（sqeuclidean ≡ MLlib squaredEuclidean）──
    km_lbl = np.array([int(r['cluster_id']) for r in label_rows], dtype=int)
    v_cols = [c for c in vec_rows[0].keys() if c.startswith('v_')]
    V = np.array([[float(r[c]) for c in v_cols] for r in vec_rows], dtype=float)
    sil_sklearn = float(silhouette_score(V, km_lbl, metric='sqeuclidean'))
    sil_mllib = float(report['kmeans_silhouette_chosen'])
    delta = abs(sil_sklearn - sil_mllib)
    recompute_ok = delta <= RESIL_TOL
    print(f'复算门②：sklearn={sil_sklearn:.6f} vs MLlib={sil_mllib:.6f}，Δ={delta:.2e} '
          f'→ {"一致 ✓" if recompute_ok else "不一致 ✗"}')

    # ── K-shape（tslearn）──
    from tslearn.clustering import KShape
    sils = {}
    best = None
    t0 = time.time()
    for k in K_GRID:
        t1 = time.time()
        ks = KShape(n_clusters=k, n_init=N_INIT, random_state=SEED, verbose=False)
        y = ks.fit_predict(X)
        sil = float(silhouette_score(X, y, metric='euclidean'))
        sils[k] = sil
        if best is None or sil > best[1]:
            best = (k, sil, y)
        print(f'  K-shape K={k}: silhouette={sil:.4f}（{time.time()-t1:.1f}s）')
    k_k, sil_k, y_k = best
    print(f'K-shape 选中 k={k_k}（{time.time()-t0:.1f}s 总）：silhouette={sil_k:.4f}')

    ari = float(adjusted_rand_score(km_lbl, y_k))
    print(f'两法 ARI（K-means k={report["kmeans_k_chosen"]} vs K-shape k={k_k}）= {ari:.4f}')

    # ── 落盘 ──
    with open(os.path.join(OUT, 'cluster_kshape_labels.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['bvid', 'batch_date', 'cluster_id_kshape', 'k_kshape'])
        for r, lab in zip(curves_rows, y_k):
            w.writerow([r['bvid'], r['batch_date'], int(lab), k_k])
    sizes = {int(v): int(c) for v, c in zip(*np.unique(y_k, return_counts=True))}
    result = {
        'tool': 'kshape_validate', 'n': n,
        'kshape_silhouette': {str(k): round(v, 6) for k, v in sils.items()},
        'k_kshape': k_k, 'sil_kshape': round(sil_k, 6),
        'kmeans_k': report['kmeans_k_chosen'], 'sil_kmeans_mllib': round(sil_mllib, 6),
        'ari_kmeans_kshape': round(ari, 6),
        'kshape_cluster_sizes': sizes,
        'kmeans_recompute_sklearn': round(sil_sklearn, 6),
        'kmeans_recompute_delta': delta, 'kmeans_recompute_ok': recompute_ok,
        'normalization': 'per-video z-score',
        'run_at': time.strftime('%F %T'),
    }
    with open(os.path.join(OUT, 'cluster_kshape_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not recompute_ok:
        print('❌ 复算门②未过——请人工检查后再合并入表')
        sys.exit(1)
    print('✅ kshape_validate 完成：重跑 make m2-l1-cluster 合并入表')


if __name__ == '__main__':
    main()
