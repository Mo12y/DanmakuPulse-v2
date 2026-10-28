#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp52 · C1 负结果的机理分析 + C2 可预测性探针（只用既有资产，不建新管道）

问题（Gate B 失败后真正开放的）：
  Q1 为什么透明规则输给均匀切分？——人工阶段是"真实结构"还是"标注任务的均匀切分倾向"？
     → 段长分布统计 + 边界处强度变化幅度 vs 非边界位置的分离度
  Q2 人工标签里还有没有可学习信号？——决定 C2 是否值得追加 30 支纯测试集标注
     → 留一视频（LOO）下一阶段类别预测：探针 vs 多数类 vs 一阶转移

输入（全部既有）：data-out/c1/{ann_final.csv, curves_csv/(Spark dws_rhythm_60s), sample30_csv/}
输出：experiments/exp52_c1_structure_probe/results.json + stdout 摘要
纪律：Gate B 已按协议关闭（一次调整权用尽），本实验**不产生新的分段算法**、不改 Gate B 结论；
     Q2 是 C2 决策门的证据，不是 C2 实验（30 支全部用于探针，无独立测试集）。
"""
import csv
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

REPO = '/home/molly/projects/danmaku-v2'
C1 = f'{REPO}/data-out/c1'
OUT = f'{REPO}/experiments/exp52_c1_structure_probe'
CATS = ['L', 'R', 'H', 'B', 'D', 'X']


def load_all():
    segs, order = {}, []
    with open(f'{C1}/ann_final.csv', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            segs.setdefault(r['bvid'], []).append(
                (int(r['start_bucket']), int(r['end_bucket']), r['category']))
    cur = {}
    for p in sorted(glob.glob(f'{C1}/curves_csv/*.csv')):
        with open(p, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                cur.setdefault(r['bvid'], []).append(
                    (int(r['bucket60']), int(float(r['cnt'])), r['bucket_state']))
    sample = {}
    for p in sorted(glob.glob(f'{C1}/sample30_csv/*.csv')):
        with open(p, encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                sample[r['bvid']] = dict(n=int(r['n_buckets_expected']), got=int(r['danmaku_got']))
    for bv in segs:
        cur[bv].sort()
        order.append(bv)
    return segs, cur, sample, sorted(order)


def q1_segment_lengths(segs, sample):
    """段长分布：总体/分类别；每视频实测 CV vs 均匀切分 CV（同 k、同 n 的期望）。"""
    lens, by_cat = [], {c: [] for c in CATS}
    cvs = []
    for bv, ss in segs.items():
        n = sample[bv]['n']
        ls = [e - s + 1 for s, e, _ in ss]
        lens.extend(ls)
        for (s, e, c), l in zip(ss, ls):
            by_cat[c].append(l)
        k = len(ss)
        if k >= 2 and n >= k:
            mu = n / k
            cv_unif = float(np.sqrt(mu * (1 - 1 / k)) / mu)   # 均匀切分下段长的期望 CV（方差=n/k*(1-1/k)... 见注）
            obs = float(np.std(ls, ddof=0))
            cvs.append(dict(bvid=bv, k=k, n=n, mean_len=round(float(np.mean(ls)), 2),
                            cv_obs=round(obs / np.mean(ls), 3), cv_uniform_ref=round(cv_unif, 3)))
    arr = np.array(lens, float)
    return dict(n_segments=len(lens),
                len_mean=round(float(arr.mean()), 2), len_median=round(float(np.median(arr)), 2),
                len_p10=round(float(np.percentile(arr, 10)), 1), len_p90=round(float(np.percentile(arr, 90)), 1),
                cv_overall=round(float(arr.std() / arr.mean()), 3),
                by_cat_median={c: (round(float(np.median(v)), 1) if v else None) for c, v in by_cat.items()},
                per_video_cv=cvs,
                note='cv_uniform_ref：把 n 个桶均匀切成 k 段时段长的期望变异系数'
                     '（二项近似 sd=sqrt(mu*(1-1/k))，mu=n/k；是"边界完全随机等距"的参照，非精确解析式）')


def q2_boundary_vs_strength_change(segs, cur, sample):
    """裁决边界处的强度变化幅度，是否显著大于非边界位置？
    delta = |log1p(c[t-1]) - log1p(c[t])|（横跨边界的相邻两桶）；
    边界邻域 ±1 桶内有 X（采集缺失）的边界剔除；分母=全部可判相邻对。"""
    b_delta, b_states = [], []
    o_delta = []
    for bv, ss in segs.items():
        cs = sorted(cur[bv])
        c = [x[1] for x in cs]
        st = [x[2] for x in cs]
        n = len(c)
        bset = set()
        for s, e, cat in ss:
            if 1 <= s <= n - 1:
                bset.add(s)
        if ss[-1][1] + 1 <= n - 1:
            bset.add(ss[-1][1] + 1)
        for t in range(1, n):
            if st[t - 1] != 'observed_nonzero' or st[t] != 'observed_nonzero':
                continue                                   # 缺失/采集后桶不可判
            d = abs(np.log1p(c[t - 1]) - np.log1p(c[t]))
            near_x = any(x[2] != 'observed_nonzero' for x in cs[max(0, t - 2):t + 2])
            if t in bset and not near_x:
                b_delta.append(d)
                b_states.append((bv, t))
            elif t not in bset and not near_x:
                o_delta.append(d)
    b, o = np.array(b_delta), np.array(o_delta)
    # 分离度：用 |Δlog1p| 作"边界 vs 非边界"的判别 AUC（秩检验实现）
    from scipy.stats import mannwhitneyu
    auc = float(mannwhitneyu(b, o, alternative='greater').statistic / (len(b) * len(o))) \
        if len(b) and len(o) else None
    return dict(n_boundaries=len(b), n_nonboundary=len(o),
                boundary_delta_med=round(float(np.median(b)), 3) if len(b) else None,
                nonboundary_delta_med=round(float(np.median(o)), 3) if len(o) else None,
                boundary_delta_p90=round(float(np.percentile(b, 90)), 3) if len(b) else None,
                auc_delta_pred_boundary=round(auc, 4) if auc else None,
                note='AUC=0.5 ⇒ 边界处强度变化与非边界无异（构念错配的定量证据）；'
                     'AUC>0.7 ⇒ 边界确实落在强度跳变处')


def q2_next_category_probe(segs, sample):
    """C2 探针：LOO 视频预测"下一段类别"。
    样本 = 相邻段对 (cur_cat, cur_len, pos_frac) → next_cat；next 或 cur 为 X 的对剔除。"""
    rows = []
    for bv, ss in segs.items():
        n = sample[bv]['n']
        ss = sorted(ss)
        for i in range(len(ss) - 1):
            s0, e0, c0 = ss[i]
            s1, e1, c1 = ss[i + 1]
            if 'X' in (c0, c1):
                continue
            rows.append(dict(bvid=bv, cur=c0, ln=e0 - s0 + 1, pos=round((e0 + 1) / n, 3), nxt=c1))
    vids = sorted(set(r['bvid'] for r in rows))
    y_all = [r['nxt'] for r in rows]
    maj = max(set(y_all), key=y_all.count)

    # 基线1：全局多数类
    pred_maj = [maj] * len(rows)
    acc_maj = float(np.mean([p == t for p, t in zip(pred_maj, y_all)]))
    # 基线2：一阶转移 P(next|cur)（LOO：训练集排除该视频）
    acc_tr, acc_tr_f1 = [], []
    # 探针：LOO 逻辑回归（one-hot cur + log1p(len) + pos）
    def enc(r):
        v = [1.0 if r['cur'] == c else 0.0 for c in CATS if c != 'X']
        return v + [np.log1p(r['ln']), r['pos']]
    X = np.array([enc(r) for r in rows], float)
    acc_lr = []
    for v in vids:
        tr = [i for i, r in enumerate(rows) if r['bvid'] != v]
        te = [i for i, r in enumerate(rows) if r['bvid'] == v]
        # 基线2 LOO
        cnt = {}
        for i in tr:
            cnt.setdefault(rows[i]['cur'], {}).setdefault(rows[i]['nxt'], 0)
            cnt[rows[i]['cur']][rows[i]['nxt']] += 1
        for i in te:
            cands = cnt.get(rows[i]['cur'], {})
            nxt = max(cands, key=cands.get) if cands else maj
            acc_tr.append(nxt == rows[i]['nxt'])
            acc_tr_f1.append((nxt, rows[i]['nxt']))
        # 探针 LOO
        lr = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], [rows[i]['nxt'] for i in tr])
        pr = lr.predict(X[te])
        acc_lr.extend((pr == np.array([rows[i]['nxt'] for i in te])).tolist())

    def acc(a):
        return round(float(np.mean(a)), 4) if len(a) else None

    lr_ok = np.array(acc_lr, bool)
    tr_ok = np.array(acc_tr, bool)
    bv_arr = np.array([r['bvid'] for r in rows])
    lr_pairs = []          # (pred, true) 按 LOO 顺序，供 macro-F1
    _i = 0
    for v in vids:
        te = [i for i, r in enumerate(rows) if r['bvid'] == v]
        for i in te:
            lr_pairs.append((None, None))
    # 重建 lr 的 (pred,true) 对：与 acc_lr 同序
    lr_pairs = []
    for v in vids:
        tr = [i for i, r in enumerate(rows) if r['bvid'] != v]
        te = [i for i, r in enumerate(rows) if r['bvid'] == v]
        lr = LogisticRegression(max_iter=1000, C=1.0).fit(X[tr], [rows[i]['nxt'] for i in tr])
        pr = lr.predict(X[te])
        for p, i in zip(pr, te):
            lr_pairs.append((p, rows[i]['nxt']))

    # 探针 vs 转移基线的准确率差异：视频级成对 bootstrap（B=2000，seed=20261028）
    rng = np.random.default_rng(20261028)
    diffs = []
    for _ in range(2000):
        pick = rng.choice(vids, len(vids), True)
        m = np.isin(bv_arr, pick)
        diffs.append(float(lr_ok[m].mean() - tr_ok[m].mean()))
    ci = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]

    # macro-F1（多类，简单实现）
    def macro_f1(pairs):
        from collections import Counter
        ys = set(t for _, t in pairs) | set(p for p, _ in pairs)
        f1s = []
        for c in ys:
            tp = sum(1 for p, t in pairs if p == c and t == c)
            fp = sum(1 for p, t in pairs if p == c and t != c)
            fn = sum(1 for p, t in pairs if p != c and t == c)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return round(float(np.mean(f1s)), 4) if f1s else None

    return dict(n_pairs=len(rows), n_videos=len(vids), next_class_dist={
        c: y_all.count(c) for c in CATS if y_all.count(c)},
        majority_class=maj,
        acc_majority=acc(acc_maj and [p == t for p, t in zip(pred_maj, y_all)]),
        acc_transition_loo=acc(acc_tr),
        acc_lr_probe_loo=acc(acc_lr),
        macro_f1_majority=macro_f1([(maj, t) for t in y_all]),
        macro_f1_transition_loo=macro_f1(acc_tr_f1),
        macro_f1_lr_probe_loo=macro_f1(lr_pairs),
        lr_vs_transition=dict(
            n=len(acc_tr),
            diff_acc=round(acc(acc_lr) - acc(acc_tr), 4),
            note='差异置信区间用成对 bootstrap（视频级，B=2000，seed=20261028）',
            ci=ci),
        note='LOO=留一视频；next/cur 为 X 的对已剔除；探针特征=当前类别 one-hot+段长+位置占比，'
             '仅 LOO 逻辑回归（无未来信息；特征不含被测段的任何后缀信息）')


def main():
    segs, cur, sample, order = load_all()
    out = dict(meta=dict(n_videos=len(order), n_segments=sum(len(s) for s in segs.values()),
                         generated_by='experiments/exp52_c1_structure_probe.py'),
               q1_segment_lengths=q1_segment_lengths(segs, sample),
               q2_boundary_vs_strength_change=q2_boundary_vs_strength_change(segs, cur, sample),
               q3_next_category_probe=q2_next_category_probe(segs, sample))
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ('q1_segment_lengths', 'q2_boundary_vs_strength_change',
                                          'q3_next_category_probe')}, ensure_ascii=False, indent=1))
    print('EXP52-DONE')


if __name__ == '__main__':
    main()
