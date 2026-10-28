#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp49 · C1 Gate B 评估骨架（边界匹配 / 基线 / 候选方法）

状态（2026-10-28）：
  - **正式 Gate B 被 Gate A 阻塞**（30 支双人阶段标注尚未完成，见 docs/c1_annotation_protocol.md §0/§5）。
  - 本文件只交付：可测的纯函数（匹配/基线/候选）+ 冒烟运行（6 支 33 峰历史真值，峰位≠阶段边界，
    结果**不得**作为 Gate B 结论）。
  - 预设门槛（协议 §7）：中位边界误差 ≤15s 且 60s 匹配率 ≥60% —— 是门槛，不是结果。
  - 失败重试预算：一次。第二次调整 = 协议违规。

口径（与协议 v1.0 一致）：
  - 边界 = 阶段的 start_bucket；末段追加 end_bucket+1（右边界）
  - 匹配 = 双向贪心最近邻，|Δ| ≤ tol（桶）；一个真值边界至多配一个预测边界
  - precision = 命中 / #预测边界；recall = 命中 / #真值边界；F1 = 调和均值
"""
import glob
import json
import csv

import numpy as np

TOL_BUCKETS = 1          # 60s 容差 = 1 桶
SMOOTH_K = 3             # 候选方法：中心滑动均值窗
BASE_HALF = 6            # 候选方法：局部基线窗半宽（±6 桶）
THR_K = 2.0              # 候选方法：base + K*sqrt(base+1)
MIN_TOTAL = 5            # 事件最少总弹幕
MIN_EXCESS = 3           # 事件最少超出量


# ---------------------------------------------------------------- 匹配（纯函数，可测）
def match_boundaries(truth, pred, tol=TOL_BUCKETS):
    """双向贪心最近匹配。truth/pred: 桶序号 list[int]。
    返回 dict(hits, pairs, precision, recall, f1, errors)。"""
    truth = sorted(int(x) for x in truth)
    pred = sorted(int(x) for x in pred)
    if not truth or not pred:
        return dict(hits=0, pairs=[], precision=0.0, recall=0.0, f1=0.0,
                    errors=[], n_truth=len(truth), n_pred=len(pred))
    pairs, used = [], set()
    errors = []
    # 真值优先，按最近预测匹配（贪心）
    for t in truth:
        cand = sorted(((abs(p - t), p) for p in pred if p not in used))
        if cand and cand[0][0] <= tol:
            d, p = cand[0]
            used.add(p)
            pairs.append((t, p, d))
            errors.append(d)
    hits = len(pairs)
    precision = hits / len(pred)
    recall = hits / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(hits=hits, pairs=pairs, precision=precision, recall=recall, f1=f1,
                errors=errors, n_truth=len(truth), n_pred=len(pred))


def boundaries_of_segments(segs, n_buckets):
    """段列表 [(s,e)]（闭区间，桶）→ 边界桶列表 = 每个 s + 末段 e+1（视频右边界，协议 §6）"""
    if not segs:
        return []
    segs = sorted(segs)
    b = [s for s, _ in segs]
    right = min(segs[-1][1] + 1, n_buckets)
    if right > b[-1]:
        b.append(right)
    return sorted(set(b))


# ---------------------------------------------------------------- 基线 1：等频分段（纯函数）
def seg_equal_freq(n_buckets, k):
    """把 [0,n) 均分为 k 段（闭区间）。k<=0 或 n<=0 → []。"""
    if n_buckets <= 0 or k <= 0:
        return []
    k = min(k, n_buckets)
    edges = [round(i * n_buckets / k) for i in range(k + 1)]
    return [(edges[i], edges[i + 1] - 1) for i in range(k) if edges[i] < edges[i + 1]]


# ---------------------------------------------------------------- 基线 2：单变点（纯函数）
def seg_single_changepoint(cnts):
    """累计计数曲线的最大单变点（O(n) 扫描）：最大化左右两段与整体均值的平方误差和之差。
    返回 [(s,e)] 两段；n<4 时整段返回（不可分）。"""
    n = len(cnts)
    if n < 4:
        return [(0, n - 1)] if n else []
    c = np.asarray(cnts, float)
    total, pre = c.sum(), np.cumsum(c)
    best, best_k = -np.inf, None
    for k in range(2, n - 1):                       # 变点后第一段从 k 开始
        left, right = pre[k - 1], total - pre[k - 1]
        nl, nr = k, n - k
        sse_split = (left ** 2 / nl + right ** 2 / nr)
        sse_all = total ** 2 / n
        gain = sse_split - sse_all
        if gain > best:
            best, best_k = gain, k
    if best_k is None:
        return [(0, n - 1)]
    return [(0, best_k - 1), (best_k, n - 1)]


# ---------------------------------------------------------------- 候选：2σ 突起 → 阶段（纯函数）
def seg_event_candidates(cnts, smooth_k=SMOOTH_K, base_half=BASE_HALF, thr_k=THR_K,
                         min_total=MIN_TOTAL, min_excess=MIN_EXCESS):
    """透明候选方法：中心滑动均值平滑 → ±base_half 桶中位作局部基线 → base+K*sqrt(base+1)
    判突起 → 相邻突起合并为事件（total≥min_total 且 excess≥min_excess）→ 事件并集的
    连续覆盖区间即阶段。不含任何学习参数。"""
    c = np.asarray(cnts, float)
    n = len(c)
    if n == 0:
        return []
    sm = np.convolve(c, np.ones(smooth_k) / smooth_k, mode='same') if n >= smooth_k else c
    segs, i = [], 0
    while i < n:
        lo, hi = max(0, i - base_half), min(n, i + base_half + 1)
        base = float(np.median(c[lo:hi]))
        thr = base + thr_k * np.sqrt(base + 1.0)
        if sm[i] >= thr and c[i] >= thr:
            j = i
            while j + 1 < n:
                lo2, hi2 = max(0, j + 1 - base_half), min(n, j + 1 + base_half + 1)
                b2 = float(np.median(c[lo2:hi2]))
                t2 = b2 + thr_k * np.sqrt(b2 + 1.0)
                if sm[j + 1] >= t2 and c[j + 1] >= t2:
                    j += 1
                else:
                    break
            total = float(c[i:j + 1].sum())
            excess = float((c[i:j + 1] - np.array([np.median(c[max(0, k - base_half):min(n, k + base_half + 1)])
                                                   for k in range(i, j + 1)])).clip(0).sum())
            if total >= min_total and excess >= min_excess:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs


# ---------------------------------------------------------------- 汇总（纯函数）
def evaluate(truth_by_video, pred_by_video, tol=TOL_BUCKETS):
    """truth_by_video/pred_by_video: {bvid: 边界桶 list}。返回总体与逐视频。"""
    per, all_err = {}, []
    TP = FP = FN = 0
    for bv in sorted(set(truth_by_video) | set(pred_by_video)):
        r = match_boundaries(truth_by_video.get(bv, []), pred_by_video.get(bv, []), tol)
        per[bv] = r
        all_err.extend(r['errors'])
        TP += r['hits']
        FP += r['n_pred'] - r['hits']
        FN += r['n_truth'] - r['hits']
    P = TP / (TP + FP) if TP + FP else 0.0
    Rc = TP / (TP + FN) if TP + FN else 0.0
    F1 = 2 * P * Rc / (P + Rc) if (P + Rc) else 0.0
    return dict(micro=dict(precision=P, recall=Rc, f1=F1, TP=TP, FP=FP, FN=FN,
                           med_err_buckets=(float(np.median(all_err)) if all_err else None),
                           match_rate_in_tol=(len(all_err) / TP if TP else None)),
                per_video={k: {kk: vv for kk, vv in v.items() if kk != 'pairs'} for k, v in per.items()})


# ---------------------------------------------------------------- 冒烟：6 支 33 峰历史真值（非正式）
def smoke_peak_truth(snapshot_curve_fn=None, out_path=None):
    """用 data-out/annotator_B_visual.csv（6 支 33 峰）做冒烟。
    ⚠️ 峰秒 ≠ 阶段边界：本结果只证明链路能跑，**不得**写进任何 Gate B 结论。"""
    rows = list(csv.DictReader(open('/home/molly/projects/danmaku-v2/data-out/annotator_B_visual.csv',
                                    encoding='utf-8-sig')))
    truth = {}
    for r in rows:
        bv = (r.get('bvid') or '').strip()
        t = r.get('peak_sec') or r.get('peak_s') or r.get('t')
        if not bv or t in (None, ''):
            continue
        truth.setdefault(bv, []).append(int(float(t)) // 60)
    # 曲线来源：C1 池内这 6 支若不合格则如实缺数据；此处只报真值规模，不做算法对照
    res = {bv: dict(n_truth=len(peaks)) for bv, peaks in truth.items()}
    return dict(n_videos=len(truth), n_peaks=sum(len(v) for v in truth.values()),
                note='峰位≠阶段边界；仅链路冒烟', per_video=res)


if __name__ == '__main__':
    # ---- 纯函数自检（合成数据，不触真实数据）----
    r = match_boundaries([0, 10, 20], [0, 11, 40])
    assert r['hits'] == 2 and abs(r['recall'] - 2 / 3) < 1e-9 and abs(r['precision'] - 2 / 3) < 1e-9
    assert seg_equal_freq(10, 3) == [(0, 2), (3, 6), (7, 9)], seg_equal_freq(10, 3)
    cp = seg_single_changepoint([1, 1, 1, 9, 9, 9])
    assert cp[0][1] == 2 and cp[1][0] == 3, cp
    ev = seg_event_candidates([0, 0, 0, 50, 0, 0, 0, 0])
    assert ev and ev[0][0] == 3 and ev[0][1] == 3, ev
    print('PURE-FUNC-OK')

    # ---- 端到端冒烟：Spark 曲线产物 → 候选分段（链路验证，非有效性结论）----
    import glob as _g
    curves_p = _g.glob('/home/molly/projects/danmaku-v2/data-out/c1/curves_csv/*.csv')
    if curves_p:
        rows = list(csv.DictReader(open(curves_p[0], encoding='utf-8-sig')))
        by_bv, meta_n = {}, {}
        for r2 in rows:
            bv = r2['bvid']
            meta_n[bv] = max(meta_n.get(bv, 0), int(r2['bucket60']))
            by_bv.setdefault(bv, [0.0])[-1] = 0
            while len(by_bv[bv]) <= int(r2['bucket60']):
                by_bv[bv].append(0.0)
            by_bv[bv][int(r2['bucket60'])] = float(r2['cnt'])
        out_rows, n_post = [], 0
        for bv in sorted(by_bv):
            c = by_bv[bv]
            segs = seg_event_candidates(c)
            for k, (s, e) in enumerate(segs, 1):
                out_rows.append(dict(bvid=bv, seg_id=k, start_bucket=s, end_bucket=e,
                                     n_buckets=len(c), category='B',
                                     note='algo_smoke：2σ 突起候选；标注者禁看'))
            n_post += sum(1 for r2 in rows if r2['bvid'] == bv and r2['bucket_state'] == 'post_snapshot')
        outp = '/home/molly/projects/danmaku-v2/data-out/c1/algo_smoke_segments.csv'
        with open(outp, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=['bvid', 'seg_id', 'start_bucket', 'end_bucket',
                                              'n_buckets', 'category', 'note'])
            w.writeheader()
            w.writerows(out_rows)
        print(f'E2E-SMOKE: {len(by_bv)} 支 → {len(out_rows)} 个候选段（2σ 突起；'
              f'其中 post_snapshot 桶 {n_post} 个已按协议排除在计数外）→ {outp}')
        print('⚠️ 标注者禁看本文件（防锚定）；本结果不是 Gate B 结论')
    else:
        print('E2E-SMOKE: 未找到 curves_csv（先跑 jobs/c1_prepare_annotation.py）')

    print(json.dumps(smoke_peak_truth(), ensure_ascii=False)[:300])
    print('EXP49-SMOKE-DONE（冒烟；正式 Gate B 等待 Gate A 人工标注）')
