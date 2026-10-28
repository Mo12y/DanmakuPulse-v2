#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp51 · Gate B 正式评估（协议 §7）：三个方法 vs 裁决后人工边界

方法（协议预设，不得增删）：
  1. equal_freq  等频分段（k = 该视频裁决段数）
  2. single_cp   单变点基线（累计计数最大分割增益，O(n)）
  3. candidate   C1 候选：2σ 突起事件 → 连续覆盖区间（透明、无学习参数）

评价参照：ann_final.csv（裁决后边界，2026-10-28 裁决，含 28 条逐条 resolution）
指标：中位边界误差（秒）、60s 内匹配率（=recall@1桶）、边界 precision/recall/F1、
      相对基线增量、逐视频分布、失败案例、按覆盖/弹幕量分层的敏感性。
门槛（预注册）：候选 med_err ≤ 15s 且 60s 匹配率 ≥ 60%。粒度说明：边界全在分钟刻度上，
误差 ∈ {0,60,120,...}s ⇒ "≤15s" 等价于"匹配对中位误差为 0 桶（同桶）"，从严执行，不做事后放宽。

输入：data-out/c1/{ann_final.csv, curves_csv/(Spark 产物), sample30_csv/}
输出：experiments/exp51_gateb_results/results.json + 本 stdout 摘要
"""
import csv
import glob
import importlib.util
import json
import os

import numpy as np

REPO = '/home/molly/projects/danmaku-v2'
C1 = f'{REPO}/data-out/c1'
OUT = f'{REPO}/experiments/exp51_gateb_results'
SPEC = importlib.util.spec_from_file_location('exp49', f'{REPO}/experiments/exp49_c1_gateb.py')
exp49 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp49)

GATE_MED_ERR_S = 15.0
GATE_MATCH_RATE = 0.60


def load_ann_final(path):
    segs = {}
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        for r in rd:
            segs.setdefault(r['bvid'], []).append((int(r['start_bucket']), int(r['end_bucket']), r['category']))
    return segs


def load_curves(path_glob):
    rows = []
    for p in sorted(glob.glob(f'{path_glob}/*.csv')):
        with open(p, encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f))
    series, n_exp = {}, {}
    for r in rows:
        bv = r['bvid']
        i = int(r['bucket60'])
        s = series.setdefault(bv, [0.0])
        while len(s) <= i:
            s.append(0.0)
        s[i] = float(r['cnt'])
    return series


def load_sample(path_glob):
    rows = []
    for p in sorted(glob.glob(f'{path_glob}/*.csv')):
        with open(p, encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f))
    return {r['bvid']: dict(n=int(r['n_buckets_expected']), got=int(r['danmaku_got']),
                            cov=float(r['n_buckets_nonzero']) / max(int(r['n_buckets_expected']), 1),
                            tname=r['tname']) for r in rows}


def summarise(truth, pred_by, sample, label):
    """对一种方法求总体指标 + 逐视频。返回 (overall, per_video)。"""
    res = exp49.evaluate(truth, pred_by, tol=1)
    micro = res['micro']
    med_err_s = None if micro['med_err_buckets'] is None else micro['med_err_buckets'] * 60.0
    overall = dict(method=label,
                   precision=round(micro['precision'], 4), recall=round(micro['recall'], 4),
                   f1=round(micro['f1'], 4), TP=micro['TP'], FP=micro['FP'], FN=micro['FN'],
                   med_err_s=med_err_s,
                   match_rate_60s=round(micro['recall'], 4))
    per = {}
    for bv, r in res['per_video'].items():
        per[bv] = dict(n_truth=r['n_truth'], n_pred=r['n_pred'], hits=r['hits'],
                       recall=round(r['recall'], 4),
                       med_err_s=(None if r is None or not r['errors'] else float(np.median(r['errors'])) * 60.0),
                       n_seg_expected=sample[bv]['n'] if bv in sample else None)
    return overall, per


def main():
    ann = load_ann_final(f'{C1}/ann_final.csv')
    series = load_curves(f'{C1}/curves_csv')
    sample = load_sample(f'{C1}/sample30_csv')

    truth, k_seg, missing_curve = {}, {}, []
    for bv, segs in ann.items():
        n = sample[bv]['n']
        truth[bv] = exp49.boundaries_of_segments([(s, e) for s, e, _ in segs], n)
        k_seg[bv] = len(segs)
        if bv not in series:
            missing_curve.append(bv)
    if missing_curve:
        raise SystemExit(f'❌ 裁决稿中的视频缺曲线（Spark 产物）：{missing_curve[:3]}')

    pred_equal, pred_cp, pred_cand, pred_cand_v1 = {}, {}, {}, {}
    for bv in ann:
        c, n = series[bv], sample[bv]['n']
        pred_equal[bv] = exp49.boundaries_of_segments(exp49.seg_equal_freq(n, k_seg[bv]), n)
        pred_cp[bv] = exp49.boundaries_of_segments(exp49.seg_single_changepoint(c), n)
        pred_cand[bv] = exp49.boundaries_of_segments(exp49.seg_event_candidates_v2(c), n)
        pred_cand_v1[bv] = exp49.boundaries_of_segments(exp49.seg_event_candidates(c), n)

    ov_eq, per_eq = summarise(truth, pred_equal, sample, 'equal_freq(等频)')
    ov_cp, per_cp = summarise(truth, pred_cp, sample, 'single_cp(单变点)')
    ov_cd, per_cd = summarise(truth, pred_cand, sample, 'candidate_v2(2σ突起,调整后)')
    ov_v1, _ = summarise(truth, pred_cand_v1, sample, 'candidate_v1(首跑,留档)')

    gate = dict(med_err_s=ov_cd['med_err_s'], med_err_gate=ov_cd['med_err_s'] is not None
               and ov_cd['med_err_s'] <= GATE_MED_ERR_S,
               match_rate_60s=ov_cd['match_rate_60s'],
               match_rate_gate=ov_cd['match_rate_60s'] >= GATE_MATCH_RATE)
    gate['verdict'] = 'PASS' if (gate['med_err_gate'] and gate['match_rate_gate']) else 'FAIL'

    # 增量（相对基线）
    delta = dict(vs_equal_freq=dict(d_f1=round(ov_cd['f1'] - ov_eq['f1'], 4),
                                    d_match=round(ov_cd['match_rate_60s'] - ov_eq['match_rate_60s'], 4)),
                 vs_single_cp=dict(d_f1=round(ov_cd['f1'] - ov_cp['f1'], 4),
                                   d_match=round(ov_cd['match_rate_60s'] - ov_cp['match_rate_60s'], 4)))

    # 失败案例：候选 recall 最差 5 支
    worst = sorted(per_cd.items(), key=lambda kv: (kv[1]['recall'], -kv[1]['n_truth']))[:5]
    worst5 = [dict(bvid=bv, **{k: v for k, v in d.items()}) for bv, d in worst]

    # 敏感性分层（K=4 主口径思路：覆盖 / 弹幕量）
    def stratify(keyfn):
        buckets = {}
        for bv in truth:
            g = keyfn(bv)
            buckets.setdefault(g, []).append(bv)
        out = {}
        for g, bvs in sorted(buckets.items()):
            r = exp49.evaluate({b: truth[b] for b in bvs}, {b: pred_cand[b] for b in bvs}, tol=1)['micro']
            out[g] = dict(n=len(bvs), f1=round(r['f1'], 4),
                          match_rate=round(r['recall'], 4),
                          med_err_s=(None if r['med_err_buckets'] is None else r['med_err_buckets'] * 60.0))
        return out
    sens = dict(by_coverage=stratify(lambda b: 'cov>=0.8' if sample[b]['cov'] >= 0.8 else 'cov<0.8'),
                by_got=stratify(lambda b: 'got<300' if sample[b]['got'] < 300 else
                                          ('got300-500' if sample[b]['got'] < 500 else 'got500-600')),
                by_len=stratify(lambda b: 'short(<15桶)' if sample[b]['n'] < 15 else
                                          ('mid(15-40桶)' if sample[b]['n'] <= 40 else 'long(>40桶)')))

    out = dict(gate=gate, overall=dict(equal_freq=ov_eq, single_cp=ov_cp, candidate=ov_cd,
                                       candidate_v1_rejected=ov_v1),
               adjustment_log=dict(allowed=1, used=1,
                                   reason='v1 两实现缺陷：3桶平滑稀释单桶尖峰(50→28.7<36.4)、'
                                          'mode=same 边缘零填充杀片头爆发(102→52.3<64.3)；'
                                          '构念不变（相对突起段起止为段界），检测改用原始计数',
                                   date='2026-10-28'),
               delta_vs_baselines=delta, sensitivity=sens,
               worst5_candidate=worst5,
               meta=dict(n_videos=len(truth), n_truth_bounds=sum(len(v) for v in truth.values()),
                         reference='ann_final.csv（2026-10-28 裁决，28 条逐条 resolution）',
                         curves='data-out/c1/curves_csv（Spark dws_rhythm_60s 产物）',
                         gate_predefined=dict(med_err_s=GATE_MED_ERR_S, match_rate=GATE_MATCH_RATE),
                         granularity_note='边界全在分钟刻度 ⇒ 误差∈{0,60,...}s；"≤15s"等价于匹配对中位同桶'))
    os.makedirs(OUT, exist_ok=True)
    with open(f'{OUT}/results.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(dict(gate=gate, overall=out['overall'], delta=delta, sens=sens,
                          worst5=worst5), ensure_ascii=False, indent=1))
    print(f"\nGATE-B-{gate['verdict']}  med_err={gate['med_err_s']}s  match_rate={gate['match_rate_60s']}")


if __name__ == '__main__':
    main()
