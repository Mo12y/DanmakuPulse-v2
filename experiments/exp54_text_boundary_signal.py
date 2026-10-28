#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp54 · 边界的文本信号：内容维度 × 节奏维度（C1 遗产的内容侧挖掘）

问题（从未做过、且手里有独特资产）：
  人工裁决的阶段边界，除了计数形态变化（exp52 AUC=0.834），是否还伴随**文本构成**的突变？
  如果有，文本特征能否为 exp53 的有监督边界检测器提供**计数之外的增量价值**？
  ——这正是"内容×时间"维度，且直接决定 ML 设计里检测器的特征方案。

设计：
  同 exp53 的有监督 LOO 框架（30 支、内部边界 79、±1 桶容差、折内选阈值、NMS、未观测区硬过滤），
  三组特征配对比较（同一 LOO 折、同一种子）：
    S1 count-only   exp53 的 8 个计数特征（复现基线）
    S2 text-only    边界两侧的文本构成特征（无计数特征）
    S3 count+text   两者拼接（增量检验）
  配对差异 = S3 − S1 的 recall/F1，视频级 bootstrap 95%CI（B=2000，seed=20261028）。
  全部特征组在 LOO 折内做 StandardScaler（计数与文本率尺度不同，不标准化会互相淹没）。
  另附：各类别（裁决段）的文本构成画像（描述性）。

文本特征（每分钟桶，由桶内弹幕计算；空桶=0）：
  n, uniq_rate=1-重复率, mean_len, exclam(！!~), question(？?), laugh(233/哈哈/hhh/草/笑死),
  emotive(泪/破防/awsl/名场面/高能/ ahead), second(你), first(我), mark(高能|名场面|前方)
边界特征 = 每个文本特征在 t 与 t-1 的水平 + 差分 + |差分|。

输出：experiments/exp54_text_boundary_signal/results.json
"""
import csv
import glob
import importlib.util
import json
import os
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = '/home/molly/projects/danmaku-v2'
C1 = f'{REPO}/data-out/c1'
OUT = f'{REPO}/experiments/exp54_text_boundary_signal'
SPEC = importlib.util.spec_from_file_location('exp53', f'{REPO}/experiments/exp53_supervised_boundary.py')
e53 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e53)

SEED = 20261028
CATS = ['L', 'R', 'H', 'B', 'D', 'X']

RE_LAUGH = re.compile(r'233|哈哈|hhh|hhh|草|笑死|笑不活|绷不住', re.I)
RE_EMOTE = re.compile(r'泪|破防|awsl|名场面|高能|封神|爷青|太好哭|太好笑', re.I)
RE_EXCL = re.compile(r'[！!~]{1,}')
RE_QUES = re.compile(r'[？?]')
RE_SECOND = re.compile(r'你|妳|up|UP|楼主|up主')
RE_FIRST = re.compile(r'我|咱')


def bucket_text_features(texts):
    """texts: 该桶内弹幕文本 list[str] → dict 特征（空桶=全 0）"""
    n = len(texts)
    if n == 0:
        keys = ['n', 'uniq', 'mean_len', 'exclam', 'ques', 'laugh', 'emotive', 'second', 'first']
        return {k: 0.0 for k in keys}
    users = [u for u in texts]
    uniq = len(set(texts)) / n
    lens = [len(t) for t in texts]
    joined = ' '.join(texts)

    def rate(rx):
        c = sum(1 for t in texts if rx.search(t))
        return c / n
    return dict(n=float(n), uniq=round(uniq, 4), mean_len=round(float(np.mean(lens)), 2),
                exclam=round(rate(RE_EXCL), 4), ques=round(rate(RE_QUES), 4),
                laugh=round(rate(RE_LAUGH), 4), emotive=round(rate(RE_EMOTE), 4),
                second=round(rate(RE_SECOND), 4), first=round(rate(RE_FIRST), 4))


def load_text():
    txt = {}
    p = sorted(glob.glob(f'{C1}/text30/*.csv'))
    if not p:
        raise SystemExit('❌ text30 未导出（先跑 jobs/c1_export_text.py）')
    with open(p[0], encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            txt.setdefault((r['bvid'], int(r['batch_date'])), {}).setdefault(
                int(r['bucket60']), []).append(r['txt'])
    return txt


def main():
    segs, series, states, sample = e53.load()
    txt_all = load_text()
    # text30 的键是 (bvid,batch_date)；sample30 的 batch 也一致
    txt_by_bv = {}
    for (bv, bt), buckets in txt_all.items():
        txt_by_bv[bv] = buckets          # 30 支已按 bvid 去重（每支 1 批）

    vids_u = sorted(segs)
    missing = [bv for bv in vids_u if bv not in txt_by_bv]
    if missing:
        raise SystemExit(f'❌ 缺文本的视频：{missing[:3]}')

    # ---------- 每桶文本特征 ----------
    tf = {}
    for bv in vids_u:
        n = sample[bv]['n']
        buckets = txt_by_bv[bv]
        tf[bv] = {t: bucket_text_features(buckets.get(t, [])) for t in range(n)}

    # ---------- 三组特征矩阵 ----------
    TFEATS = ['n', 'uniq', 'mean_len', 'exclam', 'ques', 'laugh', 'emotive', 'second', 'first']

    def count_feat(series_v, states_v, t, n):
        return e53.features(series_v, states_v, t, n) if hasattr(e53, 'features') else None

    # exp53 的 features 是模块级函数：直接复用
    X_count, y_all, pos_all, vids_all, internal = e53.build_xy(segs, series, states, sample)
    pos_arr = np.array(pos_all, dtype=object)

    def text_feat_vec(bv, t, n):
        ft, fprev = tf[bv][t], tf[bv][t - 1] if t >= 1 else {k: 0.0 for k in TFEATS}
        v = []
        for k in TFEATS:
            v += [ft[k], fprev[k], ft[k] - fprev[k], abs(ft[k] - fprev[k])]
        return v

    def both_feat(bv, t, n, i):
        ci = list(X_count[i])
        return ci + text_feat_vec(bv, t, n)

    X_text, X_both = [], []
    for i, (bv, t) in enumerate(pos_all):
        X_text.append(text_feat_vec(bv, t, sample[bv]['n']))
        X_both.append(both_feat(bv, t, sample[bv]['n'], i))
    X_text, X_both = np.array(X_text, float), np.array(X_both, float)

    truth_int = {bv: e53.internal_boundaries(segs[bv], sample[bv]['n']) for bv in vids_u}

    # ---------- LOO（同 exp53 管线） ----------
    vids = vids_all                      # 与 X 行同序的 bvid 列表（exp53.build_xy 产出）
    def run_loo(Xm, cw='balanced'):
        pred, sweep = {}, []
        for v in vids_u:
            te = [i for i, b in enumerate(vids) if b == v]
            tr = [i for i in range(len(vids)) if vids[i] != v]
            lr = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, C=1.0, class_weight=cw,
                                                  random_state=SEED)).fit(Xm[tr], y_all[tr])
            probs = lr.predict_proba(Xm[te])[:, 1]
            positions = [int(p[1]) for p in pos_arr[te]]
            valid = [t for t in positions
                     if states[v][t] == 'observed_nonzero' and states[v][t - 1] == 'observed_nonzero']
            vp = [p for p, t in zip(probs, positions) if t in set(valid)]
            thr, _ = e53.best_threshold(y_all[tr], vp, valid)
            pred[v] = e53.nms(vp, valid, thr)
            for t in np.arange(0.10, 0.91, 0.05):
                sweep.append((round(float(t), 2), v, e53.nms(vp, valid, float(t))))
        return pred, sweep

    def pack(pred):
        r = e53.evaluate(truth_int, pred)['micro']
        return dict(f1=round(r['f1'], 4), precision=round(r['precision'], 4), recall=round(r['recall'], 4),
                    TP=r['TP'], FP=r['FP'], FN=r['FN'],
                    med_err_s=(None if r['med_err_buckets'] is None else r['med_err_buckets'] * 60.0),
                    n_pred=r['TP'] + r['FP'])

    pred_s1, sw1 = run_loo(X_count)
    pred_s2, sw2 = run_loo(X_text)
    pred_s3, sw3 = run_loo(X_both)
    S1, S2, S3 = pack(pred_s1), pack(pred_s2), pack(pred_s3)

    # 配对差异 bootstrap（视频级，S3−S1）
    rng = np.random.default_rng(SEED)
    d_rec, d_f1 = [], []
    for _ in range(2000):
        pick = rng.choice(vids_u, len(vids_u), True)
        r1 = e53.evaluate({b: truth_int[b] for b in pick}, {b: pred_s1[b] for b in pick})['micro']
        r3 = e53.evaluate({b: truth_int[b] for b in pick}, {b: pred_s3[b] for b in pick})['micro']
        d_rec.append(r3['recall'] - r1['recall'])
        d_f1.append(r3['f1'] - r1['f1'])
    ci = dict(d_recall=dict(diff=round(float(np.mean(d_rec)), 4),
                            ci=[round(float(np.percentile(d_rec, 2.5)), 4),
                                round(float(np.percentile(d_rec, 97.5)), 4)]),
              d_f1=dict(diff=round(float(np.mean(d_f1)), 4),
                        ci=[round(float(np.percentile(d_f1, 2.5)), 4),
                            round(float(np.percentile(d_f1, 97.5)), 4)]))

    # ---------- 描述性：各类别文本画像 ----------
    cat_feat = {c: {k: [] for k in TFEATS} for c in CATS}
    for bv, ss in segs.items():
        for s, e, c in ss:
            for t in range(s, e + 1):
                if t in tf[bv]:
                    for k in TFEATS:
                        cat_feat[c][k].append(tf[bv][t][k])
    cat_table = {c: {k: (round(float(np.median(v)), 3) if v else None) for k, v in d.items()}
                 for c, d in cat_feat.items()}
    cat_n = {c: len(cat_feat[c]['n']) for c in CATS}

    out = dict(meta=dict(n_videos=len(vids_u), n_internal_truth=sum(len(v) for v in truth_int.values()),
                         n_text_rows=11557, seed=SEED,
                         feature_sets=dict(S1='count-only(=exp53)', S2='text-only', S3='count+text')),
               results=dict(S1=S1, S2=S2, S3=S3), paired_bootstrap_S3_vs_S1=ci,
               category_text_profile=dict(median=cat_table, n_segments=cat_n))
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(dict(S1=S1, S2=S2, S3=S3, paired=ci), ensure_ascii=False, indent=1))
    print('类别画像（中位）:', json.dumps(cat_table, ensure_ascii=False))
    print('EXP54-DONE')


if __name__ == '__main__':
    main()
