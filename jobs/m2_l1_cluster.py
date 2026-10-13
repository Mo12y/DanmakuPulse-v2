#!/usr/bin/env python3
"""
m2_l1_cluster.py — L1 形态聚类器①（08 §3.4：K-means++ 先行；K-shape 在宿主侧，见下）

输入：dws_video_rhythm_features（16 维指纹，全量 509）+ dws_density_1pct（1% 桶，曲线导出用）
输出：
  ① {wh}/dws_curve_clusters —— 每 (bvid, batch_date) 一行：
     cluster_id / k / silhouette（K-means++，主口径，MLlib）
     cluster_id_kshape / k_kshape / ari_kmeans_kshape（K-shape 对拍列；由宿主工具回填后重跑本作业合并）
  ② /opt/work/data-out/ 导出（宿主 K-shape 与复算用，driver 侧 open() 写——集群 fs 默认 HDFS）：
     cluster_curves_dense.csv（509×100 稠密 1% 曲线：缺失桶 ⇔ 0，与 m2_l1_density 契约一致）
     cluster_kmeans_labels.csv（bvid,batch_date,cluster_id,k,silhouette）
     cluster_kmeans_vectors.csv（bvid,batch_date + 16 维标准化向量）
     m2_l1_cluster_report.json（k 网格轮廓、选中 k、簇大小、最近质心复算一致率、合并状态）

流程（两法分工，08 §3.4）：
  ① 本作业（容器）：MLlib K-means++，k∈[3,8] 按轮廓系数选优 → 表（kshape 列为空）→ 导出
  ② make cluster-kshape（宿主 venv，tslearn「driver 侧」）：K-shape + 两法 ARI → data-out 结果文件
  ③ 重跑本作业：检测到宿主结果文件 → 合并 kshape 列 + ARI 入表（幂等）
  ⚠️ K-shape 走宿主 = 08「driver 侧」的落位（镜像无 tslearn；与 probe 类脚本同模式，记 CHANGELOG）

口径：
  - K-means 在 MLlib 内部为 k-means||（分布式 k-means++ 变体；initMode 默认），头注记明
  - 标准化：StandardScaler(withMean/withStd)；轮廓用 ClusteringEvaluator（squaredEuclidean，MLlib 默认）
  - 复算门（B2.1「抽样复算一致」）：最近质心（NumPy）复算全量标签 vs MLlib 标签；宿主侧再用
    sklearn（sqeuclidean）复算选中 k 的轮廓值（对拍 MLlib 值）——见 tools/kshape_validate.py
  - 幂等（08 §0.1）：全量覆盖写
"""
import argparse
import csv
import json
import os

import numpy as np
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

ENGINE_VERSION = 'b2-v1'
FEAT_COLS = ['dm_per_min', 'wdm_per_min', 'rhythm_entropy', 'burst_cv', 'w_burst_cv',
             'early_lag_ratio', 'color_ratio', 'top_mode_ratio',
             'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio', 'avg_text_len',
             'opening_5pct', 'ending_5pct', 'mid_ratio']
N_BUCKETS = 100          # 1% 桶域 [0,99]（density 契约：缺失行 ⇔ 0）
K_GRID = list(range(3, 9))   # 08 §3.4：k∈[3,8]


def read_host_csv(path):
    """driver 侧读宿主导出文件（集群 executor 无 data-out 挂载——不能 spark.read.csv 本地路径）"""
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_host_csv(path, header, rows):
    """driver 侧 open() 写（spark-master 挂 ../data-out；集群 df.write 无 scheme 路径会进 HDFS）"""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('m2_l1_cluster')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    fp = spark.read.parquet(f'{wh}/dws_video_rhythm_features')
    n_all = fp.count()
    null_cond = F.lit(False)
    for c in FEAT_COLS:
        null_cond = null_cond | F.col(c).isNull()
    fp_ok = fp.filter(~null_cond).select('bvid', 'batch_date', *FEAT_COLS)
    n_ok = fp_ok.count()
    n_null = n_all - n_ok

    # ── 标准化（向量组装 + StandardScaler）──
    # ⚠️ 容器无 pandas（刻意，不引）：不可用 pyspark.ml.functions.vector_to_array；
    #    collect 后取 DenseVector.toArray() 转 NumPy（2026-10-13 实测踩坑）
    asm = VectorAssembler(inputCols=FEAT_COLS, outputCol='features')
    assembled = asm.transform(fp_ok).cache()
    assembled.count()
    sc = StandardScaler(inputCol='features', outputCol='features_scaled',
                        withMean=True, withStd=True)
    scaled = sc.fit(assembled).transform(assembled).cache()
    scaled.count()

    # ── k 网格：K-means++（MLlib k-means||）按轮廓选优 ──
    evaluator = ClusteringEvaluator(featuresCol='features_scaled', predictionCol='cluster_id',
                                    metricName='silhouette')  # 默认 squaredEuclidean（头注）
    sils, preds = {}, {}
    for k in K_GRID:
        km = KMeans(featuresCol='features_scaled', predictionCol='cluster_id', k=k, seed=42)
        model = km.fit(scaled)
        pred = model.transform(scaled).cache()
        pred.count()
        sils[k] = float(evaluator.evaluate(pred))
        preds[k] = (model, pred)
    best_k = max(sils, key=lambda k: sils[k])
    best_model, best_pred = preds[best_k]
    sil_best = sils[best_k]
    centroids = np.array(best_model.clusterCenters())

    # ── 复算门①：最近质心（NumPy）复算标签 vs MLlib 标签 ──
    rows = best_pred.select('bvid', 'batch_date', 'cluster_id', 'features_scaled').collect()
    X = np.array([r['features_scaled'].toArray() for r in rows], dtype=float)
    lbl = np.array([r['cluster_id'] for r in rows], dtype=int)
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    recompute = d2.argmin(axis=1)
    recompute_match = float((recompute == lbl).mean())

    # ── 导出（宿主 K-shape / sklearn 复算用；driver 侧写）──
    os.makedirs(args.out_dir, exist_ok=True)
    rows_sorted = sorted(rows, key=lambda r: (r['batch_date'], r['bvid']))
    write_host_csv(os.path.join(args.out_dir, 'cluster_kmeans_labels.csv'),
                   ['bvid', 'batch_date', 'cluster_id', 'k', 'silhouette'],
                   [[r['bvid'], r['batch_date'], r['cluster_id'], best_k, f'{sil_best:.10f}']
                    for r in rows_sorted])
    write_host_csv(os.path.join(args.out_dir, 'cluster_kmeans_vectors.csv'),
                   ['bvid', 'batch_date'] + [f'v_{c}' for c in FEAT_COLS],
                   [[r['bvid'], r['batch_date']] + [f'{v:.10f}' for v in r['features_scaled'].toArray()]
                    for r in rows_sorted])

    # 稠密 1% 曲线（缺失桶 ⇔ 0）——driver collect 45k 行（稀疏）后按域填充；
    # 行集合 = 全量 509 键（与 labels/vectors 严格一一对齐；density 契约保证非空桶全覆盖）
    d1 = (spark.read.parquet(f'{wh}/dws_density_1pct').select('bvid', 'batch_date', 'bucket1pct', 'cnt')
          .collect())
    dense = {}
    for r in d1:
        dense.setdefault((r['batch_date'], r['bvid']), [0.0] * N_BUCKETS)[r['bucket1pct']] = float(r['cnt'])
    curve_keys = [(r['batch_date'], r['bvid']) for r in rows_sorted]
    n_curve_missing = sum(1 for k in curve_keys if k not in dense)
    curve_rows = [[bv, bd] + [f'{v:.6f}' for v in dense.get((bd, bv), [0.0] * N_BUCKETS)]
                  for bd, bv in curve_keys]
    write_host_csv(os.path.join(args.out_dir, 'cluster_curves_dense.csv'),
                   ['bvid', 'batch_date'] + [f'b{i}' for i in range(N_BUCKETS)], curve_rows)

    # ── K-shape 结果合并（宿主工具产出 → 存在则并入）──
    ks_labels_path = os.path.join(args.out_dir, 'cluster_kshape_labels.csv')
    ks_result_path = os.path.join(args.out_dir, 'cluster_kshape_result.json')
    merged = {'kshape_merged': False}
    base = best_pred.select('bvid', 'batch_date', 'cluster_id').withColumn('k', F.lit(best_k)) \
        .withColumn('silhouette', F.lit(sil_best))
    if os.path.isfile(ks_labels_path) and os.path.isfile(ks_result_path):
        ks_rows = read_host_csv(ks_labels_path)
        ks_res = json.load(open(ks_result_path, encoding='utf-8'))
        ks_df = spark.createDataFrame(
            [(r['bvid'], r['batch_date'], int(r['cluster_id_kshape']), int(r['k_kshape']))
             for r in ks_rows], ['bvid', 'batch_date', 'cluster_id_kshape', 'k_kshape'])
        out = (base.join(ks_df, ['bvid', 'batch_date'], 'left')
               .withColumn('ari_kmeans_kshape', F.lit(float(ks_res['ari_kmeans_kshape'])))
               .withColumn('engine_version', F.lit(ENGINE_VERSION)))
        merged = {'kshape_merged': True, 'k_kshape': ks_res['k_kshape'],
                  'sil_kshape': ks_res['sil_kshape'], 'ari': ks_res['ari_kmeans_kshape']}
    else:
        out = (base.withColumn('cluster_id_kshape', F.lit(None).cast('int'))
               .withColumn('k_kshape', F.lit(None).cast('int'))
               .withColumn('ari_kmeans_kshape', F.lit(None).cast('double'))
               .withColumn('engine_version', F.lit(ENGINE_VERSION)))

    out = out.select('bvid', 'batch_date', 'cluster_id', 'k', 'silhouette',
                     'cluster_id_kshape', 'k_kshape', 'ari_kmeans_kshape', 'engine_version')
    out.write.mode('overwrite').parquet(f'{wh}/dws_curve_clusters')
    n_out = out.count()

    # 簇大小
    sizes = {int(r['cluster_id']): r['n'] for r in
             best_pred.groupBy('cluster_id').count().withColumnRenamed('count', 'n').collect()}
    report = {
        'job': 'm2_l1_cluster', 'engine_version': ENGINE_VERSION,
        'n_rows_in': n_all, 'n_used': n_ok, 'n_null_dropped': n_null,
        'k_grid': K_GRID, 'kmeans_silhouette': {str(k): round(v, 6) for k, v in sils.items()},
        'kmeans_k_chosen': best_k, 'kmeans_silhouette_chosen': round(sil_best, 6),
        'kmeans_cluster_sizes': sizes,
        'numpy_recompute_match': round(recompute_match, 10),
        'n_curve_rows_exported': len(curve_rows), 'n_curve_missing': n_curve_missing,
        'note': 'K-shape/ARI 由 tools/kshape_validate.py（宿主 venv，tslearn）产出后重跑本作业合并',
        **merged,
    }
    with open(os.path.join(args.out_dir, 'm2_l1_cluster_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    tag = f"（kshape 已合并，ARI={merged['ari']:.4f}）" if merged['kshape_merged'] else "（kshape 列待宿主工具回填）"
    print(f'✅ m2_l1_cluster：{n_out} 行；k={best_k}，silhouette={sil_best:.4f}，'
          f'复算一致率={recompute_match:.4f} {tag}')
    spark.stop()


if __name__ == '__main__':
    main()
