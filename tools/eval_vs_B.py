# -*- coding: utf-8 -*-
'''eval_vs_B.py — 检测器 vs 标注者 B（多模态看图）重跑落盘（2026-10-15 复核整改，F5）

口径（自定、可复现；原 25 号 B 行「D4 0.395」协议未留脚本、不可复现）：
  - B 数据源: tools/annotator_B_visual.csv（33 峰 / 6 视频；No.15/No.20 仅第 1 屏）
  - 真值区间: 峰秒 t → [t, t+5)（5s 桶约定）
  - 观看范围: No.15 ≤1200s / No.20 ≤1300s 之外的候选剔除（与标注范围一致）
  - 匹配: center（主）与 IoU 双报；thr=0.3（与 eval_l2_peaks 主流程一致）
  - batch 键: 从 annotator_A_poisson.csv 取（与 annotator_agreement.py 同源）
输出: data-out/eval_vs_B.json
'''
import csv
import json
import os
import sys
from collections import defaultdict

BASE = os.path.expanduser('~/projects/danmaku-v2')
sys.path.insert(0, os.path.join(BASE, 'tools'))
from eval_l2_peaks import load_cand, prf, ap  # noqa: E402

D = os.path.join(BASE, 'data-out')
VIEW = {'BV1XMYE6xEkM': 1200.0, 'BV1Cgpd6pEBR': 1300.0}

A_keys = {}
for r in csv.DictReader(open(os.path.join(BASE, 'tools/annotator_A_poisson.csv'), encoding='utf-8-sig')):
    A_keys[r['bvid']] = r['batch_date']

truth = defaultdict(list)
n_pk = 0
for r in csv.DictReader(open(os.path.join(BASE, 'tools/annotator_B_visual.csv'), encoding='utf-8-sig')):
    bv = r['bvid']
    t = float(r['peak_sec'])
    truth[(bv, A_keys[bv])].append((t, t + 5.0))
    n_pk += 1

cand = load_cand(os.path.join(D, 'l2_candidates.csv'))
candB = {}
for src in ['D1', 'D2', 'D3', 'D4', 'fused']:
    src_rows = {}
    for k, rows in cand.get(src, {}).items():
        if k[0] not in A_keys:
            continue
        lim = VIEW.get(k[0])
        rows2 = [x for x in rows if lim is None or x[0] < lim]
        if rows2:
            src_rows[k] = rows2
    candB[src] = src_rows

THRS = [round(i / 10, 2) for i in range(1, 10)]
out = {
    'protocol': 'B 峰秒 t->[t,t+5)；观看范围截断（No.15≤1200s / No.20≤1300s）；center 主/IoU 附；thr=0.3',
    'note': '原 25 号 B 行（最佳单法 D4=0.395, center）协议未留脚本、不可复现（复核 F5）；本文件为重跑唯一可复现口径',
    'date': '2026-10-15', 'n_truth_vb': len(truth), 'n_truth_peaks': n_pk,
    'sources': {},
}
print('真值 vb=%d 峰=%d | 候选限定后（fused 键数）=%d' % (len(truth), n_pk, len(candB.get('fused', {}))))
for src in ['D1', 'D2', 'D3', 'D4', 'fused']:
    c = candB.get(src, {})
    if not c:
        continue
    Pc, Rc, Fc, TPc, FPc, FNc = prf(c, truth, 0.3, 'center')
    Pi, Ri, Fi, TPi, FPi, FNi = prf(c, truth, 0.3, 'iou')
    Ac = ap(c, truth, THRS)
    out['sources'][src] = {
        'center': {'P@0.3': Pc, 'R@0.3': Rc, 'F1@0.3': Fc, 'TP': TPc, 'FP': FPc, 'FN': FNc, 'AP': Ac},
        'iou': {'P@0.3': Pi, 'R@0.3': Ri, 'F1@0.3': Fi, 'TP': TPi, 'FP': FPi, 'FN': FNi},
    }
    print('%-6s center: P=%.3f R=%.3f F1=%.3f AP=%.3f | iou: F1=%.3f' % (src, Pc, Rc, Fc, Ac, Fi))

with open(os.path.join(D, 'eval_vs_B.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('→ data-out/eval_vs_B.json 已落盘')
