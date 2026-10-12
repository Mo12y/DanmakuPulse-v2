#!/usr/bin/env python3
"""probe_spam_calibration.py — 刷量阈值合成注入定标（10 §3；"合成放大"标注纪律）

方法：纯合成数据（seeded，可复现）模拟 m0_full 的检测规则：
    is_spam_user = (p_u > share 且 cv_u < cv) 或 burst_u > burst
扫描：spam 模式 {equal（等间隔——cv≈0）, burst（60s 内 25 条）} × spam 用户条数 {10,30,60,100}
      × 每视频 600 条正常弹幕（~200 用户长尾），每格 100 次重复取均值。
输出：TPR（注入 spam 被检出）/ FPR（正常用户误标）矩阵 + 结论（data-out/calibration_spam.json）

⚠️ 本实验数据为**合成放大**，不代表真实检出率；用途=阈值行为验证（边界条件与误报基线）。
"""
import json
import math
import os
import random

SHARE, CV_T, BURST_T = 0.05, 0.10, 20
N_TRIALS = 100
N_NORMAL = 220  # 正常用户数（条数 1-6 长尾，合计 ~600 条）


def gen_normal_ts(rng):
    n = rng.randint(1, 6)
    t = 0.0
    ts = []
    for _ in range(n):
        t += max(5.0, rng.lognormvariate(4.5, 1.0))  # 间隔 ~90s 中位
        ts.append(t)
    return ts


def gen_spam_ts(rng, mode, n):
    if mode == 'equal':
        return [30.0 * i for i in range(n)]  # 等间隔 30s → cv=0
    else:  # burst
        return [i * 0.02 for i in range(n)]  # 0.02s 间隔 → 1s 内全发


def user_stats(ts):
    cnt = len(ts)
    gaps = [ts[i + 1] - ts[i] for i in range(cnt - 1)]
    if len(gaps) >= 2:
        mean = sum(gaps) / len(gaps)
        var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        cv = math.sqrt(var) / mean if mean > 0 else None
    else:
        cv = None
    burst = 0
    j = 0
    for i in range(len(ts)):
        while ts[i] - ts[j] > 59:
            j += 1
        burst = max(burst, i - j + 1)
    return cnt, cv, burst


def detected(cnt, cv, burst, total):
    p_u = cnt / total
    return ((p_u > SHARE and cv is not None and cv < CV_T) or burst > BURST_T)


report = {'note': '合成放大（重放+扰动）——阈值行为验证，非真实检出率',
          'rule': f'p_u>{SHARE} & cv<{CV_T} | burst>{BURST_T}',
          'normal_danmaku_per_video': '~600（220 用户长尾）',
          'matrix': {}}
for mode in ['equal', 'burst']:
    for n in [10, 30, 60, 100]:
        tp = fp = 0
        for trial in range(N_TRIALS):
            rng = random.Random(7000 + trial)
            norm_ts = [gen_normal_ts(rng) for _ in range(N_NORMAL)]
            spam_ts = gen_spam_ts(rng, mode, n)
            total = sum(len(t) for t in norm_ts) + len(spam_ts)
            scnt, scv, sburst = user_stats(spam_ts)
            tp += int(detected(scnt, scv, sburst, total))
            for t in norm_ts:
                c, cv, b = user_stats(t)
                fp += int(detected(c, cv, b, total))
        n_fp_checks = N_TRIALS * N_NORMAL
        report['matrix'][f'{mode}_n{n}'] = {
            'tpr': round(tp / N_TRIALS, 4),
            'fpr': round(fp / n_fp_checks, 6),
        }

out = 'data-out/calibration_spam.json'
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print(json.dumps(report, ensure_ascii=False, indent=2))
