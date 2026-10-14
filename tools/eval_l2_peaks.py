# -*- coding: utf-8 -*-
'''eval_l2_peaks.py — 21 号 §4 表2：逐方法匹配与 P/R/AP 对比
用法:
  python tools/eval_l2_peaks.py --cand data-out/l2_candidates.csv --truth <标注csv>
  truth 支持两种: mark_sheet.csv (peaks_sec 列, '起-止; 起-止') 或 csv(bvid,batch_date,start_sec,end_sec)
自检: --self-truth fused@0.5  用融合@0.5 当真值，验证代码可跑
'''
import argparse, ast, csv, sys
from collections import defaultdict
import numpy as np

def iou(a, b):
    s = max(a[0], b[0]); e = min(a[1], b[1])
    inter = max(0.0, e - s)
    uni = (a[1]-a[0]) + (b[1]-b[0]) - inter
    return inter/uni if uni > 0 else 0.0

def load_cand(p):
    d = defaultdict(lambda: defaultdict(list))
    with open(p, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            k = (r['bvid'], r['batch_date'])
            d[r['source'].strip()][k].append((float(r['start_sec']), float(r['end_sec']), float(r['score'])))
    return d

def parse_peaks(s):
    out = []
    for part in str(s).replace('；',';').split(';'):
        part = part.strip()
        if not part: continue
        for sep in ('-', '–', '~', '～'):
            if sep in part:
                a, b = part.split(sep, 1)
                try: out.append((float(a.strip()), float(b.strip())))
                except ValueError: pass
                break
    return out

def load_truth_sheet(p):
    t = defaultdict(list)
    with open(p, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            pk = parse_peaks(r.get('peaks_sec', ''))
            if pk: t[(r['bvid'], r['batch_date'])] = pk
    return t

def load_truth_csv(p):
    t = defaultdict(list)
    with open(p, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            t[(r['bvid'], r['batch_date'])].append((float(r['start_sec']), float(r['end_sec'])))
    return t

def prf(cand, truth, thr):
    TP = FP = FN = 0
    for k, peaks in truth.items():
        cs = sorted(cand.get(k, []), key=lambda x: -x[2])
        used = [False]*len(peaks)
        for s, e, sc in cs:
            best, bi = 0.0, -1
            for i, pk in enumerate(peaks):
                if used[i]: continue
                v = iou((s, e), pk)
                if v > best: best, bi = v, i
            if best >= thr and bi >= 0: used[bi] = True; TP += 1
            else: FP += 1
        FN += sum(1 for u in used if not u)
    P = TP/(TP+FP) if TP+FP else 0.0
    R = TP/(TP+FN) if TP+FN else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    return P, R, F, TP, FP, FN

def ap(cand, truth, thrs):
    pts = [(P, R) for P, R, F, *_ in (prf(cand, truth, t) for t in thrs)]
    pts = sorted(set(pts), key=lambda x: x[1])
    a = 0.0; pr = 0.0; rr = 0.0
    for P, R in pts:
        a += P * (R - rr); rr = R; pr = P
    return a

def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--cand', default='data-out/l2_candidates.csv')
    ap_.add_argument('--truth', required=True)
    ap_.add_argument('--out', default='data-out/l2_eval_report.json')
    a = ap_.parse_args()
    cand = load_cand(a.cand)
    truth = load_truth_sheet(a.truth) if a.truth.endswith('mark_sheet.csv') else load_truth_csv(a.truth)
    n_pk = sum(len(v) for v in truth.values())
    print('真值: %d 个视频, %d 个峰' % (len(truth), n_pk))
    THRS = [round(x, 2) for x in np.arange(0.1, 1.0, 0.1)]
    print('%-8s %6s %6s %6s %6s | ' % ('source', 'P', 'R', 'F1', 'AP') + ' '.join('P@%.1f' % t for t in (0.3, 0.5)), flush=True)
    res = {}
    for src in ['D1', 'D2', 'D3', 'D4', 'fused']:
        c = cand.get(src, {})
        if not c: continue
        P, R, F, TP, FP, FN = prf(c, truth, 0.3)
        A = ap(c, truth, THRS)
        p5 = prf(c, truth, 0.5)[0]
        res[src] = {'P@0.3': P, 'R@0.3': R, 'F1@0.3': F, 'AP': A, 'P@0.5': p5, 'TP': TP, 'FP': FP, 'FN': FN, 'n_cand': sum(len(v) for v in c.values())}
        print('%-8s %6.3f %6.3f %6.3f %6.3f | %6.3f %6.3f  (cand=%d TP=%d FP=%d FN=%d)' % (src, P, R, F, A, P, p5, res[src]['n_cand'], TP, FP, FN), flush=True)
    if 'fused' in res:
        best = max((s for s in res if s != 'fused'), key=lambda s: res[s]['F1@0.3'], default=None)
        if best:
            d = res['fused']['F1@0.3'] - res[best]['F1@0.3']
            print('DoD (08 4.2) 融合 F1 >= 最佳单法: %s (fused %.3f vs %s %.3f, diff %+.3f)' % ('PASS' if d >= 0 else 'FAIL', res['fused']['F1@0.3'], best, res[best]['F1@0.3'], d))
    import json
    json.dump(res, open(a.out, 'w'), ensure_ascii=False, indent=1)
    print('->', a.out)
main()