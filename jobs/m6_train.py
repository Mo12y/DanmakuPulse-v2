#!/usr/bin/env python3
"""
m6_train.py — M6 T1 消融训练与验证（预注册 14 号 §5/§6）

输入：{wh}/ml_features（m6_features 产；provisional_n101）
输出：{out}/ml_metrics.csv（长表：目标×层×模型×重复×折）+ {out}/m6_train_report.json

协议（预注册 §5，逐条）：
  - 切分：5-fold × 5 重复（按 c03 十分位分层；seed=42+rep）——单批不足时对滚动切分的临时适配
  - 模型：naive（c03 十分位中位增长）→ 岭回归（MLlib LinearRegression, elasticNet=0, regParam=0.1）
          → 随机森林（100 树, depth5）→ GBT（主；maxDepth=5, maxIter=50）
  - 层：A / A+B / A+B+C；目标：y_growth_v2（主）/ log1p_c07（对照）
  - 指标：RMSE/MAE/R²（折均值±std + pooled OOF）；skill=1−RMSE/RMSE_naive；
          DM（pooled OOF；HLN 修正 DM×√(n/(n−1))，p 取 t(n−1) 双尾）
  - conformal（|OOF 残差| q90 → 实测覆盖率）、学习曲线（20–100%）、分位误差（c03 十分位）
  - 控制协议演示：OLS y ~ [1, log(c03), 块] → 各块 TEST 集增量 R²（半偏语义）
运行：spark-submit --master local[2]（101 样本；MLlib 数学等价，本地模式免集群调度开销）
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import t as t_dist
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

ENGINE_VERSION = 'b2-v1'
A_NUM = ['duration', 'pub_hour_bin_0', 'pub_hour_bin_1', 'pub_hour_bin_2', 'up_prior_n']
B_COLS = ['n_danmaku', 'dm_per_min', 'rhythm_entropy', 'burst_cv', 'opening_5pct', 'ending_5pct',
          'mid_ratio', 'avg_text_len', 'color_ratio', 'top_mode_ratio',
          'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio']
C_COLS = ['n_events', 'peak_gap_cv', 'sync_score_med', 'herd_mean']
N_REP, N_FOLD = 5, 5


def to_df(spark, X, y):
    rows = [(Vectors.dense(X[i].tolist()), float(y[i])) for i in range(len(y))]
    return spark.createDataFrame(rows, ['features', 'label'])


def ml_predict(spark, model, Xtr, ytr, Xte_list):
    """fit on train, predict for a list of test matrices（复用同一模型对象）"""
    m = model.fit(to_df(spark, Xtr, ytr))
    return [np.array([r['prediction'] for r in
                      m.transform(to_df(spark, Xte, np.zeros(len(Xte)))).select('prediction').collect()])
            for Xte in Xte_list]


def stratified_folds(c03, seed):
    order = np.argsort(c03)
    fold = np.empty(len(c03), dtype=int)
    rng = np.random.RandomState(seed)
    # 每 10 个一组（十分位近似）内轮转分配
    for i in range(0, len(order), 10):
        grp = order[i:i + 10]
        perm = rng.permutation(len(grp))
        for j, k in enumerate(perm):
            fold[grp[k]] = j % N_FOLD
    return fold


def dm_test(e_naive, e_model):
    d = e_naive ** 2 - e_model ** 2
    n = len(d)
    dbar = d.mean()
    g0 = d.var(ddof=1)
    if g0 <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(g0 / n)
    dm_hln = dm * np.sqrt(n / (n - 1))          # HLN 修正（h=1）
    p = 2 * t_dist.sf(abs(dm_hln), df=n - 1)
    return float(dm_hln), float(p)


def metrics(y, p):
    e = y - p
    sse = float(np.sum(e ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {'rmse': float(np.sqrt(np.mean(e ** 2))), 'mae': float(np.mean(np.abs(e))),
            'r2': float(1 - sse / sst) if sst > 0 else float('nan')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()

    b = (SparkSession.builder.appName('m6_train')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    df = spark.read.parquet(f'{args.wh_root}/ml_features')
    rows = df.collect()
    n = len(rows)
    print(f'样本 {n}（{df.select("batch_date").distinct().count()} 批）')

    # ── 特征矩阵（driver 侧；tname top10 + other one-hot）──
    from collections import Counter
    tn = Counter(r['tname'] for r in rows)
    top_tn = [t for t, _ in tn.most_common(10)]
    def mk(d):
        tname = d['tname'] or 'NA'
        oh = [1.0 if tname == t else 0.0 for t in top_tn] + [1.0 if tname not in top_tn else 0.0]
        a = [float(d['duration'] or 0), 1.0 if d['pub_hour_bin'] == 0 else 0.0,
             1.0 if d['pub_hour_bin'] == 1 else 0.0, 1.0 if d['pub_hour_bin'] == 2 else 0.0,
             float(d['up_prior_n'] or 1)] + oh
        bcols = [float(d[c] or 0.0) for c in B_COLS]
        ccols = [float(d[c] or 0.0) for c in C_COLS]
        return np.array(a + bcols + ccols)
    X = np.array([mk(r) for r in rows])
    nA, nB, nC = len(A_NUM) + len(top_tn) + 1, len(B_COLS), len(C_COLS)
    idx_A = slice(0, nA); idx_B = slice(nA, nA + nB); idx_C = slice(nA + nB, nA + nB + nC)
    y_main = np.array([float(r['y_growth_v2']) for r in rows])
    y_abs = np.array([float(r['log1p_c07']) for r in rows])
    c03 = np.array([float(r['c03']) for r in rows])
    logc03 = np.log1p(c03)

    LAYERS = {'A': [idx_A], 'A+B': [idx_A, idx_B], 'A+B+C': [idx_A, idx_B, idx_C]}
    TARGETS = {'y_growth_v2': y_main, 'log1p_c07': y_abs}
    rows_csv = []

    # naive：按训练集 c03 十分位取中位增长（绝对值目标 = 中位 delta + logc03）
    def naive_pred2(c03tr, ytr, logtr, c03te, logte, is_abs):
        qs = np.quantile(c03tr, np.linspace(0, 1, 11))
        qs[0], qs[-1] = -np.inf, np.inf
        bins_tr = np.digitize(c03tr, qs[1:-1])
        med = {}
        for bbi in range(10):
            m = bins_tr == bbi
            if is_abs:
                med[bbi] = np.median(ytr[m] - logtr[m]) if m.sum() else float(np.median(ytr - logtr))
            else:
                med[bbi] = np.median(ytr[m]) if m.sum() else float(np.median(ytr))
        bins_te = np.digitize(c03te, qs[1:-1])
        out = np.array([med[bbi] for bbi in bins_te])
        return out + logte if is_abs else out

    oof = {}   # (target, layers, model) -> (preds_per_rep list of arrays, per-fold rows)
    tr_global = {}

    for tname, y in TARGETS.items():
        is_abs = (tname == 'log1p_c07')
        for lname, slices in LAYERS.items():
            cols = np.concatenate([X[:, s] for s in slices], axis=1)
            for model in ['naive', 'ridge', 'rf', 'gbt']:
                rep_preds, per_fold = [], []
                for rep in range(N_REP):
                    fold = stratified_folds(c03, seed=42 + rep)
                    pr = np.full(n, np.nan)
                    for f in range(N_FOLD):
                        te = fold == f; tr = ~te
                        tr_global['t'] = tr
                        mu, sd = cols[tr].mean(0), cols[tr].std(0)
                        sd[sd == 0] = 1.0
                        Xtr, Xte = (cols[tr] - mu) / sd, (cols[te] - mu) / sd
                        if model == 'naive':
                            p = naive_pred2(c03[tr], y[tr], logc03[tr], c03[te], logc03[te], is_abs)
                        elif model == 'ridge':
                            [p] = ml_predict(spark, LinearRegression(featuresCol='features',
                                             labelCol='label', regParam=0.1, elasticNetParam=0.0),
                                             Xtr, y[tr], [Xte])
                        elif model == 'rf':
                            [p] = ml_predict(spark, RandomForestRegressor(featuresCol='features',
                                             labelCol='label', numTrees=100, maxDepth=5, seed=42),
                                             Xtr, y[tr], [Xte])
                        else:
                            [p] = ml_predict(spark, GBTRegressor(featuresCol='features',
                                             labelCol='label', maxDepth=5, maxIter=50, seed=42),
                                             Xtr, y[tr], [Xte])
                        pr[te] = p
                        mf = metrics(y[te], p)
                        per_fold.append({**mf, 'rep': rep, 'fold': f})
                    rep_preds.append(pr)
                oof[(tname, lname, model)] = (rep_preds, per_fold)
                fm = {k: [d[k] for d in per_fold] for k in ('rmse', 'mae', 'r2')}
                print(f'  [{tname}|{lname}|{model}] RMSE {np.mean(fm["rmse"]):.4f}±{np.std(fm["rmse"]):.4f} '
                      f'R² {np.mean(fm["r2"]):.3f}')

    # ── pooled 汇总 + skill + DM + conformal ──
    summary = {}
    for tname, y in TARGETS.items():
        naive_pool = np.nanmean([oof[(tname, 'A', 'naive')][0][r] for r in range(N_REP)], axis=0)
        rmse_naive = float(np.sqrt(np.mean((y - naive_pool) ** 2)))
        for lname in LAYERS:
            for model in ['naive', 'ridge', 'rf', 'gbt']:
                reps, per_fold = oof[(tname, lname, model)]
                pool = np.nanmean(reps, axis=0)
                mm = metrics(y, pool)
                skill = 1 - mm['rmse'] / rmse_naive if rmse_naive > 0 else 0.0
                dmv, dmp = dm_test(y - naive_pool, y - pool)
                q90 = float(np.quantile(np.abs(y - pool), 0.90))
                cov = float(np.mean(np.abs(y - pool) <= q90))
                key = f'{tname}|{lname}|{model}'
                summary[key] = {**mm, 'skill_vs_naive': round(skill, 4),
                                'dm_hln': round(dmv, 3), 'dm_p': round(dmp, 6),
                                'conformal_q90': round(q90, 4), 'coverage': round(cov, 4)}
                rows_csv.append([tname, lname, model, -1, -1, mm['rmse'], mm['mae'], mm['r2'],
                                 skill, dmv, dmp, q90, cov, n])
                for d in per_fold:
                    rows_csv.append([tname, lname, model, d['rep'], d['fold'], d['rmse'],
                                     d['mae'], d['r2'], '', '', '', '', '', len(y)])

    # ── 学习曲线（GBT × A+B+C × 主目标）──
    y = y_main
    cols = np.concatenate([X[:, s] for s in LAYERS['A+B+C']], axis=1)
    curve = []
    for size in (0.2, 0.4, 0.6, 0.8, 1.0):
        rmses = []
        for rep in range(N_REP):
            fold = stratified_folds(c03, seed=42 + rep)
            for f in range(N_FOLD):
                te = fold == f; tr = np.where(~te)[0]
                k = max(int(len(tr) * size), 8)
                trs = tr[np.argsort(c03[tr])]                    # 按 c03 跨度均匀子采样
                sub = trs[np.linspace(0, len(trs) - 1, k).astype(int)]
                mu, sd = cols[sub].mean(0), cols[sub].std(0)
                sd[sd == 0] = 1.0
                [p] = ml_predict(spark, GBTRegressor(featuresCol='features', labelCol='label',
                                 maxDepth=5, maxIter=50, seed=42),
                                 (cols[sub] - mu) / sd, y[sub], [(cols[te] - mu) / sd])
                rmses.append(metrics(y[te], p)['rmse'])
        curve.append({'frac': size, 'rmse_mean': float(np.mean(rmses)),
                      'rmse_std': float(np.std(rmses))})
        print(f'  [学习曲线] {int(size*100)}% → RMSE {curve[-1]["rmse_mean"]:.4f}')

    # ── 分位误差（GBT × A+B+C × 主目标；按 c03 十分位）──
    pool = np.nanmean([oof[('y_growth_v2', 'A+B+C', 'gbt')][0][r] for r in range(N_REP)], axis=0)
    qe = []
    qs = np.quantile(c03, np.linspace(0, 1, 11)); qs[0], qs[-1] = -np.inf, np.inf
    b10 = np.digitize(c03, qs[1:-1])
    for bbi in range(10):
        m = b10 == bbi
        if m.sum():
            qe.append({'decile': bbi, 'n': int(m.sum()),
                       'rmse': float(np.sqrt(np.mean((y[m] - pool[m]) ** 2)))})

    # ── 控制协议（log c03 强制入模；块增量 TEST R²）──
    ctrl = {}
    for tname, yv in TARGETS.items():
        inc = {'B': [], 'C': [], 'B+C': []}
        for rep in range(N_REP):
            fold = stratified_folds(c03, seed=42 + rep)
            for f in range(N_FOLD):
                te = fold == f; tr = ~te
                def r2test(Xtr, Xte):
                    Xtr1 = np.column_stack([np.ones(tr.sum()), Xtr])
                    Xte1 = np.column_stack([np.ones(te.sum()), Xte])
                    beta, *_ = np.linalg.lstsq(Xtr1, yv[tr], rcond=None)
                    pred = Xte1 @ beta
                    sse = np.sum((yv[te] - pred) ** 2)
                    sst = np.sum((yv[te] - yv[te].mean()) ** 2)
                    return 1 - sse / sst if sst > 0 else np.nan
                base = r2test(logc03[tr][:, None], logc03[te][:, None])
                bB = np.concatenate([logc03[:, None], X[:, idx_B]], axis=1)
                bC = np.concatenate([logc03[:, None], X[:, idx_C]], axis=1)
                bBC = np.concatenate([logc03[:, None], X[:, idx_B], X[:, idx_C]], axis=1)
                inc['B'].append(r2test(bB[tr], bB[te]) - base)
                inc['C'].append(r2test(bC[tr], bC[te]) - base)
                inc['B+C'].append(r2test(bBC[tr], bBC[te]) - base)
        ctrl[tname] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in inc.items()}

    report = {'job': 'm6_train', 'engine_version': ENGINE_VERSION, 'n_samples': n,
              'protocol': {'folds': N_FOLD, 'repeats': N_REP, 'seed_base': 42},
              'summary': summary, 'learning_curve': curve, 'quantile_error': qe,
              'control_protocol_incremental_r2_test': ctrl}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm6_train_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out_dir, 'ml_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['target', 'layers', 'model', 'rep', 'fold', 'rmse', 'mae', 'r2', 'skill',
                    'dm_hln', 'dm_p', 'conformal_q90', 'coverage', 'n'])
        w.writerows(rows_csv)
    print(json.dumps({'summary_main': {k: v for k, v in summary.items()
                                       if k.startswith('y_growth_v2')}}, ensure_ascii=False, indent=2))
    print(f'✅ m6_train：{n} 样本；{len(summary)} 个 组合；报告与 ml_metrics.csv 已落 {args.out_dir}')
    spark.stop()


if __name__ == '__main__':
    main()
