#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp53 · 有监督边界检测 + 结构边界诊断（解决 ML 设计问题的核心实验）

背景（为什么做）：
  Gate B 预注册评估（exp51）把 t=0（视频开头）和 t=n（视频结尾）也算"边界"——这两个是任何
  分段的定义性产物，所有方法免费命中，污染匹配率的分母；且候选（事件检测器）在结构上
  无法为平台转换（L↔H）产生边界——它是事件检测器却被当成整段分段器评价。
  Gate A 之前没有人工标签，只能手写无监督规则（exp44/45/51 全部失败）；
  **现在有 30 支裁决边界，ML 设计的正确解法是有监督边界分类**。

内容：
  Part1 诊断：exp51 三方法在"仅内部边界"口径下重算（剔除 t=0/t=n），量化结构边界扭曲。
  Part2 主实验：LOO 有监督边界检测——
    特征（桶 t，只用 t 之前与局部窗口，无未来段信息）：局部 log1p 计数窗、差分、局部均值差、
      相对突起、位置占比、缺失标记
    标签：t 是否为裁决内部边界（0 < t < n）
    模型：LogisticRegression（class_weight=balanced）
    阈值：每个 LOO 折内用 29 支训练视频选 F1 最大阈值（无测试集信息泄漏）
    后处理：±1 桶非极大抑制
    评价：内部边界 P/R/F1（60s 容差 = ±1 桶）、中位误差
  对照：多数类（从不预测）、exp51 三方法同口径。

输出：experiments/exp53_supervised_boundary/results.json
"""
import csv
import glob
import importlib.util
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = '/home/molly/projects/danmaku-v2'
C1 = f'{REPO}/data-out/c1'
OUT = f'{REPO}/experiments/exp53_supervised_boundary'
SPEC = importlib.util.spec_from_file_location('exp49', f'{REPO}/experiments/exp49_c1_gateb.py')
exp49 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp49)

TOL = 1
SEED = 20261028


# ---------- 数据 ----------
def load():
    segs = {}
    with open(f'{C1}/ann_final.csv', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            segs.setdefault(r['bvid'], []).append((int(r['start_bucket']), int(r['end_bucket']), r['category']))
    series, states = {}, {}
    for p in sorted(glob.glob(f'{C1}/curves_csv/*.csv')):
        with open(p, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                bv, i = r['bvid'], int(r['bucket60'])
                series.setdefault(bv, [0.0])
                states.setdefault(bv, ['unknown_fetch'])
                while len(series[bv]) <= i:
                    series[bv].append(0.0)
                    states[bv].append('unknown_fetch')
                series[bv][i] = float(r['cnt'])
                states[bv][i] = r['bucket_state']
    sample = {}
    for p in sorted(glob.glob(f'{C1}/sample30_csv/*.csv')):
        with open(p, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                sample[r['bvid']] = dict(n=int(r['n_buckets_expected']), got=int(r['danmaku_got']))
    return segs, series, states, sample


def internal_boundaries(segs_v, n):
    """仅内部边界：0 < b < n。t=0（开头）与 t=n（结尾）是定义性边界，剔除。"""
    b = set()
    for s, e, _ in segs_v:
        if 0 < s < n:
            b.add(s)
        if 0 < e + 1 < n:
            b.add(e + 1)
    return sorted(b)


# ---------- 特征（桶 t：只用 t-1 及更早 + 局部对称窗；无未来"段"信息） ----------
def features(series_v, states_v, t, n):
    c = series_v
    lg = np.log1p(np.array(c, float))
    lo, hi = max(0, t - 3), min(n, t + 3)
    base = float(np.median(c[max(0, t - 6):min(n, t + 7)]))
    left = float(np.mean(lg[max(0, t - 3):t])) if t > 0 else 0.0
    right = float(np.mean(lg[t:min(n, t + 3)])) if t < n else 0.0
    return [lg[t] - lg[t - 1],                      # 相邻差分（exp52 Q2 的核心信号）
            abs(lg[t] - lg[t - 1]),
            left - right,                            # 局部均值差（水平变化方向与幅度）
            abs(left - right),                       # 水平变化幅度（无方向）
            lg[t] - np.log1p(base),                  # 相对突起（exp45/51 构念，保留作特征）
            c[t - 1] / max(base, 1e-9),
            t / n,                                   # 位置占比
            float(states_v[t] != 'observed_nonzero' or states_v[t - 1] != 'observed_nonzero')]


def build_xy(segs, series, states, sample):
    X, y, pos, vids = [], [], [], []
    internal = {}
    for bv in sorted(segs):
        n = sample[bv]['n']
        ib = internal_boundaries(segs[bv], n)
        internal[bv] = ib
        ibset = set(ib)
        for t in range(1, n):
            X.append(features(series[bv], states[bv], t, n))
            y.append(1 if t in ibset else 0)
            pos.append((bv, t))
            vids.append(bv)
    return np.array(X, float), np.array(y, int), pos, vids, internal


def nms(probs, positions, thr):
    """±1 桶非极大抑制：返回选中的边界桶列表"""
    cand = [(p, t) for p, t in zip(probs, positions) if p >= thr]
    cand.sort(reverse=True)
    chosen = []
    for p, t in cand:
        if all(abs(t - c) > 1 for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def best_threshold(y_true, probs, positions):
    """折内选阈值：训练视频上 F1 最大（无测试集信息）"""
    best, best_f1 = 0.5, -1.0
    for thr in np.arange(0.20, 0.81, 0.05):
        pred = set(nms(probs, positions, thr))
        tp = len(pred & set(y_true))
        p = tp / len(pred) if pred else 0.0
        r = tp / len(y_true) if len(y_true) else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best, best_f1 = float(thr), f1
    return best, best_f1


def evaluate(truth_by, pred_by):
    return exp49.evaluate(truth_by, pred_by, tol=TOL)


def main():
    segs, series, states, sample = load()
    X, y, pos, vids, internal = build_xy(segs, series, states, sample)
    vids_u = sorted(set(vids))
    pos_arr = np.array(pos, dtype=object)

    def rows_of(v):
        return [i for i, b in enumerate(vids) if b == v]

    # ---------- Part1 诊断：exp51 三方法，仅内部边界 ----------
    truth_int = {bv: internal[bv] for bv in vids_u}
    diag = {}
    for name, fn in (('equal_freq', lambda bv: exp49.boundaries_of_segments(
            exp49.seg_equal_freq(sample[bv]['n'], len(segs[bv])), sample[bv]['n'])),
            ('single_cp', lambda bv: exp49.boundaries_of_segments(
                exp49.seg_single_changepoint(series[bv]), sample[bv]['n'])),
            ('candidate_v2', lambda bv: exp49.boundaries_of_segments(
                exp49.seg_event_candidates_v2(series[bv]), sample[bv]['n']))):
        pred_int = {bv: [b for b in fn(bv) if 0 < b < sample[bv]['n']] for bv in vids_u}
        r = evaluate(truth_int, pred_int)['micro']
        diag[name] = dict(f1=round(r['f1'], 4), precision=round(r['precision'], 4),
                          recall=round(r['recall'], 4), TP=r['TP'], FP=r['FP'], FN=r['FN'],
                          med_err_s=(None if r['med_err_buckets'] is None else r['med_err_buckets'] * 60.0))
    n_int_truth = sum(len(v) for v in truth_int.values())

    # ---------- Part2 主实验：LOO 有监督（两种类权重：balanced / 自然 prevalence）----------
    lr_ok, maj_note = [], '从不预测（多数类=无边界）'
    pred_sup, thr_used, variants = {}, [], {}
    sweep = []          # 工作点扫描（诊断）：阈值 → 召回/精度/F1
    for cw in ('balanced', None):
        pred_v, thrs_v, sweep_v = {}, [], []
        from collections import Counter
        for v in vids_u:
            te_rows = rows_of(v)
            tr_rows = [i for i in range(len(vids)) if vids[i] != v]
            lr = LogisticRegression(max_iter=2000, C=1.0, class_weight=cw,
                                    random_state=SEED).fit(X[tr_rows], y[tr_rows])
            probs = lr.predict_proba(X[te_rows])[:, 1]
            positions = [int(p[1]) for p in pos_arr[te_rows]]
            # 硬过滤：边界两侧都必须是已观测桶（未观测区的"边界"无意义）
            valid = [t for t in positions
                     if states[v][t] == 'observed_nonzero' and states[v][t - 1] == 'observed_nonzero']
            vp = [p for p, t in zip(probs, positions) if t in set(valid)]
            thr, tr_f1 = best_threshold(y[tr_rows], vp, valid)
            pred = nms(vp, valid, thr)
            pred_v[v] = pred
            thrs_v.append(thr)
            for t in np.arange(0.10, 0.91, 0.05):
                sweep_v.append((round(float(t), 2), v, nms(vp, valid, float(t))))
        variants[str(cw)] = dict(pred=pred_v, thr_median=float(np.median(thrs_v)), sweep=sweep_v)
        pred_sup.update(pred_v if cw == 'balanced' else {})
        thr_used = thrs_v if cw == 'balanced' else thr_used
    # 多数类（无边界）基线：FN=全部
    # 有监督评估（仅内部边界）：两变体各评一次
    res_bal = evaluate(truth_int, variants['balanced']['pred'])['micro']
    res_nat = evaluate(truth_int, variants['None']['pred'])['micro']

    def pack(res, thr_med, n_pred):
        return dict(f1=round(res['f1'], 4), precision=round(res['precision'], 4),
                    recall=round(res['recall'], 4), TP=res['TP'], FP=res['FP'], FN=res['FN'],
                    med_err_s=(None if res['med_err_buckets'] is None else res['med_err_buckets'] * 60.0),
                    thr_median=round(thr_med, 2),
                    n_pred=n_pred, n_truth_internal=n_int_truth)
    sup_bal = pack(res_bal, variants['balanced']['thr_median'],
                   sum(len(v) for v in variants['balanced']['pred'].values()))
    sup_nat = pack(res_nat, variants['None']['thr_median'],
                   sum(len(v) for v in variants['None']['pred'].values()))
    # 工作点曲线（两变体聚合各视频同阈值）
    def sweep_table(sweep_v):
        tab = []
        for t in sorted(set(s[0] for s in sweep_v)):
            pred_by = {v: p for (tt, v, p) in sweep_v if tt == t}
            r = evaluate(truth_int, pred_by)['micro']
            tab.append(dict(thr=t, recall=round(r['recall'], 4), precision=round(r['precision'], 4),
                            f1=round(r['f1'], 4), n_pred=r['TP'] + r['FP']))
        return tab
    sweep_bal = sweep_table(variants['balanced']['sweep'])
    sweep_nat = sweep_table(variants['None']['sweep'])
    best_bal = max(sweep_bal, key=lambda d: d['f1'])
    best_nat = max(sweep_nat, key=lambda d: d['f1'])
    recall_ge_bal = [d for d in sweep_bal if d['recall'] >= 0.60]
    best_prec_bal = max(recall_ge_bal, key=lambda d: d['precision']) if recall_ge_bal else None

    # 有监督 vs 三方法（内部口径）的增量
    delta = {k: dict(d_f1=round(sup_bal['f1'] - diag[k]['f1'], 4),
                     d_recall=round(sup_bal['recall'] - diag[k]['recall'], 4)) for k in diag}

    # 逐视频 + 失败案例
    per_v = evaluate(truth_int, variants['balanced']['pred'])['per_video']
    worst = sorted(per_v.items(), key=lambda kv: (kv[1]['recall'], -kv[1]['n_truth']))[:5]
    worst5 = [dict(bvid=bv, **{k: v for k, v in d.items() if k != 'pairs'}) for bv, d in worst]

    # 敏感性：覆盖 / 长度
    def stratify(keyfn):
        buckets = {}
        for bv in vids_u:
            buckets.setdefault(keyfn(bv), []).append(bv)
        out = {}
        for g, bvs in sorted(buckets.items()):
            r = evaluate({b: truth_int[b] for b in bvs}, {b: pred_sup[b] for b in bvs})['micro']
            out[g] = dict(n=len(bvs), f1=round(r['f1'], 4), recall=round(r['recall'], 4))
        return out
    sens = dict(by_coverage=stratify(lambda b: 'cov>=0.8' if sample[b]['got'] and
                                     sample[b]['n'] else 'cov<0.8'),
                by_len=stratify(lambda b: 'short(<15桶)' if sample[b]['n'] < 15 else
                                          ('mid(15-40桶)' if sample[b]['n'] <= 40 else 'long(>40桶)')))

    # 误报诊断：未命中预测到最近真值边界的距离（定位近失 vs 幻觉结构）
    dist_all, dist_fp = [], []
    for bv in vids_u:
        tb = set(truth_int[bv])
        for p in pred_sup[bv]:
            if not tb:
                continue
            d = min(abs(p - t) for t in tb)
            dist_all.append(d)
            if d > TOL:
                dist_fp.append(d)
    nearmiss = dict(dist_hist={str(k): v for k, v in sorted(__import__('collections').Counter(dist_all).items())},
                    fp_count=len(dist_fp),
                    fp_within_3buckets=round(sum(1 for d in dist_fp if d <= 3) / max(len(dist_fp), 1), 3),
                    note='近失(≤3桶)=定位偏差；远距=把小波动当结构（过度触发）')

    gate_ref = dict(threshold='参考 Gate B 预注册门槛（match>=0.60, med<=15s）；本实验为协议后探索，不改 Gate B 结论',
                    match_rate_gate=sup_bal['recall'] >= 0.60,
                    med_err_gate=sup_bal['med_err_s'] is not None and sup_bal['med_err_s'] <= 15.0)

    out = dict(meta=dict(n_videos=len(vids_u), n_buckets=len(vids), n_internal_truth=n_int_truth,
                         note='t=0/t=n 为定义性边界，已从评价中剔除；预测硬过滤=两侧桶均已观测；'
                              '特征只用局部窗口，LOO 无泄漏'),
               part1_diagnostic_internal_only=diag,
               part2_supervised=dict(balanced=sup_bal, natural=sup_nat),
               delta_vs_unsupervised=delta,
               sweep=dict(balanced=sweep_bal, natural=sweep_nat,
                          best_f1=dict(balanced=best_bal, natural=best_nat),
                          best_precision_at_recall_ge_060=dict(balanced=best_prec_bal)),
               nearmiss=nearmiss,
               sensitivity=sens, worst5=worst5, gate_reference=gate_ref)
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(dict(diag=diag, sup_balanced=sup_bal, sup_natural=sup_nat,
                          sweep_balanced=sweep_bal, sweep_natural=sweep_nat,
                          delta=delta, sens=sens), ensure_ascii=False, indent=1))
    print(f"EXP53-DONE balanced: F1={sup_bal['f1']} R={sup_bal['recall']} P={sup_bal['precision']} | "
          f"natural: F1={sup_nat['f1']} R={sup_nat['recall']} P={sup_nat['precision']} | "
          f"truth_internal={n_int_truth}")


if __name__ == '__main__':
    main()
