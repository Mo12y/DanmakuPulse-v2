#!/usr/bin/env python3
"""
m6_panel_train.py — M6 截断面板实验（B4.3；07 §7 v1.4 统计纪律）

输入：{wh}/ml_panel（m6_panel 产；1148 行 / 101 bvid）
输出：{out}/ml_panel_metrics.csv + {out}/m6_panel_report.json

任务：① y_next = log1p(c_{d+1})（主，日级更新预测）② log_ret = log1p(c_{d+1})−log1p(c_d)（T5 留存）
切分（防泄漏双保险，报告一致性）：
  (a) GroupKFold(5) by bvid —— 同视频行不跨折（严格）
  (b) 日期滚动 —— train day≤T / test day∈{T+1,T+2}，T∈{3,5,7,9,11}（窗口测试集互斥）
      ⚠️ 日期滚动允许同视频跨日泛化（弱于 (a)）——两法并报、以一致性为结论依据
统计纪律（07 §7 v1.4）：
  - **双 N 强制**（行数/唯一 bvid）；**有效样本以 bvid 计**
  - **bvid 聚类稳健 SE / 视频级 bootstrap**（B=500）报 skill ± SE 与 95% CI；不报"×7~14"
  - DM 检验用 bvid 聚类稳健方差（CR0）；T5 控制协议：log1p(c_d) 强制入模 + 前置描述性相关
运行：spark-submit --master local[2]
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import spearmanr, t as t_dist
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b2-v1'
FEAT_COLS = ['n_danmaku', 'dm_per_min', 'rhythm_entropy', 'burst_cv', 'opening_5pct',
             'ending_5pct', 'mid_ratio', 'avg_text_len', 'color_ratio', 'top_mode_ratio',
             'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio',
             'n_events', 'peak_gap_cv', 'sync_score_med', 'herd_mean']
B_COLS = FEAT_COLS[:14]
C_COLS = FEAT_COLS[14:]
N_BOOT = 500
ROLL_T = [3, 5, 7, 9, 11]


def to_df(spark, X, y):
    return spark.createDataFrame([(Vectors.dense(X[i].tolist()), float(y[i])) for i in range(len(y))],
                                 ['features', 'label'])


def fit_predict(spark, model, Xtr, ytr, Xte):
    m = model.fit(to_df(spark, Xtr, ytr))
    return np.array([r['prediction'] for r in
                     m.transform(to_df(spark, Xte, np.zeros(len(Xte)))).select('prediction').collect()])


def model_factory(name):
    if name == 'ridge':
        return LinearRegression(featuresCol='features', labelCol='label',
                                regParam=0.1, elasticNetParam=0.0)
    if name == 'rf':
        return RandomForestRegressor(featuresCol='features', labelCol='label',
                                     numTrees=100, maxDepth=5, seed=42)
    return GBTRegressor(featuresCol='features', labelCol='label',
                        maxDepth=5, maxIter=50, seed=42)


def metrics(y, p):
    e = y - p
    sse, sst = float(np.sum(e ** 2)), float(np.sum((y - y.mean()) ** 2))
    return float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))), \
        (1 - sse / sst if sst > 0 else float('nan'))


def cluster_dm(d, groups):
    """CR0 聚类稳健 DM：Var(d̄) ≈ Σ_g (Σ_{i∈g} d_i)² / n²"""
    n = len(d)
    gs = {}
    for i, g in enumerate(groups):
        gs[g] = gs.get(g, 0.0) + d[i]
    var = sum(v * v for v in gs.values()) / (n * n)
    if var <= 0:
        return 0.0, 1.0
    dm = d.mean() / np.sqrt(var)
    p = 2 * t_dist.sf(abs(dm), df=len(gs) - 1)
    return float(dm), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()

    b = SparkSession.builder.appName('m6_panel_train').config('spark.sql.shuffle.partitions', '8')
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    rows = spark.read.parquet(f'{args.wh_root}/ml_panel').collect()
    bvids = sorted({r['bvid'] for r in rows})
    vid_ix = {v: i for i, v in enumerate(bvids)}
    n_rows, n_vid = len(rows), len(bvids)
    print(f'面板：{n_rows} 行 / {n_vid} bvid（有效样本以 bvid 计）')

    # 特征矩阵
    from collections import Counter
    tn = Counter(r['tname'] for r in rows)
    top_tn = [t for t, _ in tn.most_common(10)]
    def mk(r):
        tname = r['tname'] or 'NA'
        a = [float(r['duration'] or 0), 1.0 if r['pub_hour_bin'] == 0 else 0.0,
             1.0 if r['pub_hour_bin'] == 1 else 0.0, 1.0 if r['pub_hour_bin'] == 2 else 0.0,
             float(r['up_prior_n'] or 1)] + \
            [1.0 if tname == t else 0.0 for t in top_tn] + [1.0 if tname not in top_tn else 0.0]
        return a + [float(r[c] or 0.0) for c in FEAT_COLS]
    X = np.array([mk(r) for r in rows])
    nA = 5 + len(top_tn) + 1
    idx_A = list(range(nA))
    idx_B = list(range(nA, nA + len(B_COLS)))
    idx_C = list(range(nA + len(B_COLS), nA + len(B_COLS) + len(C_COLS)))
    Xabc = np.concatenate([X[:, idx_A], X[:, idx_B], X[:, idx_C]], axis=1)
    day = np.array([int(r['day_idx']) for r in rows])
    vid = np.array([vid_ix[r['bvid']] for r in rows])
    cd = np.array([float(r['c_d']) for r in rows])
    logcd = np.log1p(cd)
    TARGETS = {'y_next': np.array([float(r['y_next']) for r in rows]),
               'log_ret': np.array([float(r['log_ret']) for r in rows])}

    # 切分
    def group_folds(seed=42, k=5):
        rng = np.random.RandomState(seed)
        vfold = rng.permutation(n_vid) % k
        return np.array([vfold[v] for v in vid])
    gf = group_folds()

    def rolling_windows():
        out = []
        for T in ROLL_T:
            te = (day == T + 1) | (day == T + 2)
            tr = day <= T
            if te.sum() and tr.sum():
                out.append((T, tr, te))
        return out
    rw = rolling_windows()
    print(f'切分：GroupKFold(5) + 日期滚动 {[T for T, _, _ in rw]}')

    def naive_pred(tr, te, y):
        med = {}
        for d0 in np.unique(day[tr]):
            m = tr & (day == d0)
            if m.sum():
                med[d0] = float(np.median(y[m]))
        g_med = float(np.median(y[tr]))
        return np.array([med.get(d0, g_med) for d0 in day[te]])

    results = {}
    rows_csv = []
    for tname, y in TARGETS.items():
        for split in ('group', 'rolling'):
            oof_pred = {m: np.full(n_rows, np.nan) for m in ('naive', 'ridge', 'rf', 'gbt')}
            folds = ([(gf != f, gf == f) for f in range(5)] if split == 'group'
                     else [(tr, te) for _, tr, te in rw])
            for tr, te in folds:
                mu, sd = Xabc[tr].mean(0), Xabc[tr].std(0)
                sd[sd == 0] = 1.0
                Xtr, Xte = (Xabc[tr] - mu) / sd, (Xabc[te] - mu) / sd
                p = naive_pred(tr, te, y)
                oof_pred['naive'][te] = p
                for m in ('ridge', 'rf', 'gbt'):
                    oof_pred[m][te] = fit_predict(spark, model_factory(m), Xtr, y[tr], Xte)
            # pooled 指标（全体行；未覆盖行（若有）剔除）
            mask = ~np.isnan(oof_pred['gbt'])
            yv = y[mask]
            e_naive = yv - oof_pred['naive'][mask]
            rmse_naive = float(np.sqrt(np.mean(e_naive ** 2)))
            for m in ('naive', 'ridge', 'rf', 'gbt'):
                pm = oof_pred[m][mask]
                em = yv - pm
                rmse, mae, r2 = metrics(yv, pm)
                skill = 1 - rmse / rmse_naive
                dm, dmp = cluster_dm(e_naive ** 2 - em ** 2, vid[mask])
                q90 = float(np.quantile(np.abs(em), 0.90))
                cov = float(np.mean(np.abs(em) <= q90))
                # 视频级 bootstrap（skill 与 RMSE 的 SE/CI）
                vids_m = np.unique(vid[mask])
                idx_by_v = {v: np.where(vid[mask] == v)[0] for v in vids_m}
                rng = np.random.RandomState(7)
                sk_b, rm_b = [], []
                for _ in range(N_BOOT):
                    pick = rng.choice(vids_m, size=len(vids_m), replace=True)
                    ii = np.concatenate([idx_by_v[v] for v in pick])
                    rn = np.sqrt(np.mean(e_naive[ii] ** 2))
                    rr = np.sqrt(np.mean(em[ii] ** 2))
                    rm_b.append(rr)
                    sk_b.append(1 - rr / rn if rn > 0 else 0.0)
                sk_se, sk_lo, sk_hi = float(np.std(sk_b)), float(np.percentile(sk_b, 2.5)), \
                    float(np.percentile(sk_b, 97.5))
                results[(tname, split, m)] = {
                    'rmse': rmse, 'mae': mae, 'r2': r2, 'skill': skill,
                    'skill_boot_se': sk_se, 'skill_ci95': [sk_lo, sk_hi],
                    'rmse_boot_se': float(np.std(rm_b)),
                    'dm_cluster': dm, 'dm_p': dmp, 'coverage': cov, 'q90': q90}
                rows_csv.append([tname, split, m, n_rows, n_vid, rmse, mae, r2, skill,
                                 sk_se, sk_lo, sk_hi, dm, dmp, cov])
        # 两切分一致性（模型 RMSE 排序 Spearman）
        ms = ['naive', 'ridge', 'rf', 'gbt']
        rho, pv = spearmanr([results[(tname, 'group', m)]['rmse'] for m in ms],
                            [results[(tname, 'rolling', m)]['rmse'] for m in ms])
        results[(tname, 'consistency')] = {'rmse_rank_spearman': float(rho),
                                           'p': float(pv), 'models': ms}

    # T5 控制协议（log_ret；log c_d 强制入模；GroupKFold 折上 TEST 增量 R²）
    y = TARGETS['log_ret']
    inc = {'B': [], 'C': [], 'B+C': []}
    for f in range(5):
        tr, te = gf != f, gf == f
        def r2t(cols):
            Xtr = np.column_stack([np.ones(tr.sum())] + [cols[tr][:, j] for j in range(cols.shape[1] - 0)])
            Xte = np.column_stack([np.ones(te.sum())] + [cols[te][:, j] for j in range(cols.shape[1] - 0)])
            beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
            pred = Xte @ beta
            sse, sst = np.sum((y[te] - pred) ** 2), np.sum((y[te] - y[te].mean()) ** 2)
            return 1 - sse / sst if sst > 0 else np.nan
        base = r2t(logcd[:, None])
        inc['B'].append(r2t(np.column_stack([logcd, X[:, idx_B]])) - base)
        inc['C'].append(r2t(np.column_stack([logcd, X[:, idx_C]])) - base)
        inc['B+C'].append(r2t(np.column_stack([logcd, X[:, idx_B], X[:, idx_C]])) - base)
    ctrl = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in inc.items()}
    desc = {'corr_logret_logcd': float(np.corrcoef(y, logcd)[0, 1]),
            'corr_ynext_logcd': float(np.corrcoef(TARGETS['y_next'], logcd)[0, 1])}

    report = {'job': 'm6_panel_train', 'engine_version': ENGINE_VERSION,
              'N_rows': n_rows, 'N_bvid': n_vid,
              'statistics_discipline': '双 N 强制；有效样本以 bvid 计；skill±视频级 bootstrap SE',
              'splits': {'group_kfold': 5, 'rolling_windows': [T for T, _, _ in rw]},
              'results': {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in results.items() if k[1] != 'consistency'},
              'consistency': {f'{t}|rmse_rank_spearman': results[(t, 'consistency')]
                              for t in TARGETS},
              'control_protocol_log_ret': {'incremental_r2_test': ctrl, 'descriptive': desc}}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm6_panel_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'ml_panel_metrics.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['target', 'split', 'model', 'N_rows', 'N_bvid', 'rmse', 'mae', 'r2',
                    'skill', 'skill_boot_se', 'skill_ci_lo', 'skill_ci_hi', 'dm_cluster',
                    'dm_p', 'coverage'])
        w.writerows(rows_csv)

    for t in TARGETS:
        print(f'── {t} ──')
        for split in ('group', 'rolling'):
            for m in ('naive', 'ridge', 'rf', 'gbt'):
                v = results[(t, split, m)]
                print('  %-8s %-6s RMSE %.4f  R2 %+.3f  skill %+.3f±%.3f [%+.3f,%+.3f]  DM %+.2f p=%.3f'
                      % (split, m, v['rmse'], v['r2'], v['skill'], v['skill_boot_se'],
                         v['skill_ci95'][0], v['skill_ci95'][1], v['dm_cluster'], v['dm_p']))
        c = results[(t, 'consistency')]
        print('  两切分 RMSE 排序一致性：spearman %.3f (p=%.3f)' % (c['rmse_rank_spearman'], c['p']))
    print('控制协议 [log_ret]：', json.dumps(ctrl, ensure_ascii=False))
    print('描述性相关：', json.dumps(desc, ensure_ascii=False))
    print(f'✅ m6_panel_train：{n_rows} 行 / {n_vid} bvid；报告与 ml_panel_metrics.csv 已落盘')
    spark.stop()


if __name__ == '__main__':
    main()
