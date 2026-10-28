#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp61_audit_checks.py — 三项判别性诊断（审查用）

E1 D1 特征 c0 是否被弹幕池截断（读 exp61_e1_censoring 的导出）
E2 Gate B 的等频基线是否因"知道真实段数 k"而占便宜（oracle k vs 匹配 k）
E3 D4 的"无增益"是否只是阈值口径问题（同一召回下比精度）
输出：experiments/exp61_audit/{e1,e2,e3}.json
"""
import csv
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd

REPO = '/home/molly/projects/danmaku-v2'
C1 = f'{REPO}/data-out/c1'
OUT = f'{REPO}/experiments/exp61_audit'
os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location('exp49', f'{REPO}/experiments/exp49_c1_gateb.py')
e49 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e49)


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{d}/*.csv'))],
                     ignore_index=True)


# ---------------- E1
c = read_dir(f'{REPO}/data-out/exp61/censor_csv')
for k in ('got', 'n_pool', 'n6', 'n24', 'max_rel_s', 'share6', 'share24'):
    c[k] = pd.to_numeric(c[k], errors='coerce')
c = c[c.n_pool >= 50]
c['q'] = pd.qcut(c.got, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
e1 = dict(n_units=int(len(c)),
          share6_by_got_quintile={str(q): dict(n=int(len(g)), got_med=float(g.got.median()),
                                              share6_med=round(float(g.share6.median()), 4),
                                              share24_med=round(float(g.share24.median()), 4))
                                  for q, g in c.groupby('q', observed=True)},
          spearman_got_share6=round(float(np.corrcoef(c.got.rank(), c.share6.rank())[0, 1]), 4),
          max_rel_hours_median=round(float(c.max_rel_s.median() / 3600), 1))
# 低池 vs 高池：share6 的绝对差
lo = c[c.got < 600]['share6']
hi = c[c.got >= 2000]['share6']
e1['share6_lowpool_median'] = round(float(lo.median()), 4)
e1['share6_highpool_median'] = round(float(hi.median()), 4)
e1['verdict'] = ('池截断存在：高弹幕量视频的 [0,6h) 占比显著更低' if
                 e1['share6_highpool_median'] < 0.6 * e1['share6_lowpool_median']
                 else '未见明显截断')
json.dump(e1, open(f'{OUT}/e1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[E1]', json.dumps({k: e1[k] for k in ('share6_lowpool_median', 'share6_highpool_median',
                                            'spearman_got_share6', 'verdict')}, ensure_ascii=False), flush=True)

# ---------------- E2
segs, series = {}, {}
with open(f'{C1}/ann_final.csv', encoding='utf-8-sig', newline='') as f:
    for r in csv.DictReader(f):
        segs.setdefault(r['bvid'], []).append((int(r['start_bucket']), int(r['end_bucket']), r['category']))
for p in sorted(glob.glob(f'{C1}/curves_csv/*.csv')):
    with open(p, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            series.setdefault(r['bvid'], {})[int(r['bucket60'])] = float(r['cnt'])
sample = {}
for p in sorted(glob.glob(f'{C1}/sample30_csv/*.csv')):
    with open(p, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            sample[r['bvid']] = int(r['n_buckets_expected'])


def arr(bv):
    s = series[bv]
    n = max(s) + 1
    return np.array([s.get(i, 0.0) for i in range(n)], float)


truth, eq_oracle, eq_matched, cand = {}, {}, {}, {}
for bv in sorted(segs):
    n = sample[bv]
    c_ = arr(bv)
    ts = segs[bv]
    truth[bv] = [b for b in e49.boundaries_of_segments([(s, e) for s, e, _ in ts], n) if 0 < b < n]
    true_k = len(ts)
    eq_oracle[bv] = e49.boundaries_of_segments(e49.seg_equal_freq(n, true_k), n)
    cs = e49.seg_event_candidates_v2(c_)
    k_c = max(len(cs), 1)
    eq_matched[bv] = e49.boundaries_of_segments(e49.seg_equal_freq(n, k_c), n)
    cand[bv] = e49.boundaries_of_segments(cs, n)
    cand[bv] = [b for b in cand[bv] if 0 < b < n]
    eq_oracle[bv] = [b for b in eq_oracle[bv] if 0 < b < n]
    eq_matched[bv] = [b for b in eq_matched[bv] if 0 < b < n]


def pack(pred):
    r = e49.evaluate(truth, pred, tol=1)['micro']
    return dict(f1=round(r['f1'], 4), precision=round(r['precision'], 4), recall=round(r['recall'], 4),
                TP=r['TP'], FP=r['FP'], FN=r['FN'], med_err_s=r['med_err_buckets'] and r['med_err_buckets'] * 60.0)


e2 = dict(n_videos=len(truth), n_truth_internal=sum(len(v) for v in truth.values()),
          equal_freq_oracle_k=pack(eq_oracle),
          equal_freq_matched_k=pack(eq_matched),
          candidate_v2=pack(cand),
          k_source='oracle_k=人工段数（算法不可得）；matched_k=候选方法自己检出的段数')
e2['verdict'] = ('等频基线的优势主要来自 oracle k ⇒ Gate B 的"输给等频"需重新表述'
                 if e2['equal_freq_oracle_k']['f1'] > e2['equal_freq_matched_k']['f1'] + 0.05
                 else '即使去掉 oracle k，等频基线仍不弱于候选')
json.dump(e2, open(f'{OUT}/e2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[E2]', json.dumps(e2, ensure_ascii=False), flush=True)
