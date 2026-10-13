#!/usr/bin/env python3
"""
m2_l2_events.py — L2 节奏事件检测与融合（08 §4.1/§4.2）

输入：dws_density_5s（稀疏落盘=非空桶才写行；稠密化契约见 m2_l1_density 头注：
      序列=[0, max(bucket5s)]，缺失桶 ⇔ wcnt=0——本作业按该契约零填充）
      + dws_video_rhythm_features（n_danmaku 门槛）
输出：{wh}/dws_rhythm_events —— 每 (bvid, batch_date, event_id) 一行（仅融合后事件）
      （sync / δ 列由 m2_l2_sync / m2_l2_lag 追加回写；本作业只产检测与融合字段）
      + 运行报告 data-out/m2_l2_events_report.json

四检测器（输入 = 5s wcnt 稠密序列 c[0..L-1]；参数 config/params.yaml detect 节）：
  D1 郭可岩三点：T_i = P70(c[0..i])（前缀自适应阈值，最近秩）；3 点 OLS 斜率 k=(c_i−c_{i−2})/2
     ≥ d1_slope；命中间隔 < merge_gap 桶合并为段；候选原始分 = 段内 max k
  D2 孔玲玲双门限：c_i ≥ med(c)·3（密度门限）且 |c_{i+1}−c_{i−1}|/2 ≥ d2_k（变化率门限；
     边缘零填充）；候选原始分 = 段内 max 变化率
  D3 Entropy 三重约束：动态阈值 H = max(0.03R, min(0.13R, 0.06R+0.06σ))（R=max(c)，σ=std(c)）；
     峰 = 局部极大（平台取中点，端点不算）；三重约束 = 峰高 ≥ H、突出度 c/(±5 桶均值) ≥ 2、
     峰距 ≥ d3_gap_min（近者高保留）；候选原始分 = 突出度
  D4 spectral flux：F_i = Σ_{j=i−2..i} max(0, c_j−c_{j−1})（因果 3 桶窗，c_{−1}=0）；
     峰拾取 = F 局部极大且 > 局部 20 桶 mean+1.5·std；候选原始分 = F 值
融合（08 §4.2）：各法候选内 min-max 归一到 [0,1]（单候选→1.0；原始分定义见上，标定轮可复核）；
  跨法候选按 IoU>0.3 聚合（并查集，固定常数）；score_fused = Σ w_m·score_m（每法取簇内 max）；
  θ=0.5 截断。w=[0.3,0.3,0.2,0.2]（D1/D2/D3/D4）为未标定默认值。
标注纪律：fusion_w/θ 待 30 视频人工标注网格搜索（B3.3）；此前 calibrated_on='uncalibrated'
  （08 §4.2 + 异常降级条）；DoD「融合 F1 ≥ 最佳单法」在标注完成前不可验证，诚实标注。
降级：n_danmaku < l2.min_danmaku(100) 的视频跳过（事件表空；指纹相关列置空，08 §4 降级条）。
实现口径（10 §2.4 / 11 §0「算法保真优先」）：四检测器为逐视频顺序算法（D1 前缀分位数 O(L²)，
  即 11 §0 预告的"前缀分位数滑窗"例外情形）→ 以数据并行方式执行（RDD groupByKey + NumPy
  函数，跨视频并行）；论文口径 = 逐视频迭代算法数据并行执行（不宣称"算法分布式"）。
幂等：全量覆盖写；检测为纯函数（无随机），重跑结果一致（08 §0.1）。
"""
import argparse
import json
import math
import os

import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b2-v1'
METHODS = ('d1', 'd2', 'd3', 'd4')
IOU_THR = 0.3  # 08 §4.2 固定常数


# ══════════════════ 纯函数：检测器与融合（可单测，tests/test_l2_detectors.py） ══════════════════

from l2_common import platform_peaks  # noqa: E402  （共享纯函数；spark-submit 需 --py-files）


def segment_hits(hits, gap):
    """命中点序列 → 段列表：相邻间隔 < gap 桶合并（08 §4.1 D1 合并规则，各法统一采用）。"""
    segs = []
    for h in hits:
        if segs and h - segs[-1][1] < gap:
            segs[-1][1] = h
        else:
            segs.append([h, h])
    return segs


def detect_d1(c, p):
    """郭可岩三点：前缀 P70 阈值 + 3 点 OLS 斜率。"""
    L = len(c)
    if L < 3:
        return []
    import bisect
    pref = []
    hits, kv = [], {}
    for i in range(L):
        bisect.insort(pref, c[i])
        if i < 2:
            continue
        k_i = (c[i] - c[i - 2]) / 2.0
        t_i = pref[int(math.ceil(p['d1_quantile'] * len(pref))) - 1]
        if c[i] >= t_i and k_i >= p['d1_slope']:
            hits.append(i)
            kv[i] = k_i
    return [{'start': s, 'end': e, 'raw': max(kv[i] for i in range(s, e + 1) if i in kv)}
            for s, e in segment_hits(hits, p['merge_gap'])]


def detect_d2(c, p):
    """孔玲玲双门限：密度门限（med·3）+ 变化率门限（|c_{i+1}−c_{i−1}|/2）。"""
    L = len(c)
    if L == 0:
        return []
    d_val = float(np.median(c)) * 3.0
    pad = np.concatenate([[0.0], np.asarray(c, dtype=float), [0.0]])
    hits, sv = [], {}
    for i in range(L):
        cv_ = abs((pad[i + 2] - pad[i]) / 2.0)
        if c[i] >= d_val and cv_ >= p['d2_k']:
            hits.append(i)
            sv[i] = cv_
    return [{'start': s, 'end': e, 'raw': max(sv[i] for i in range(s, e + 1) if i in sv)}
            for s, e in segment_hits(hits, p['merge_gap'])]


def detect_d3(c, p):
    """Entropy 三重约束：动态阈值 H + 峰距 + 突出度。"""
    if len(c) == 0:
        return []
    R = float(np.max(c))
    if R <= 0:
        return []
    sd = float(np.std(c))
    H = max(0.03 * R, min(0.13 * R, 0.06 * R + 0.06 * sd))
    cands = []
    for i in platform_peaks(c):
        if c[i] < H:
            continue
        lo, hi = max(0, i - 5), min(len(c), i + 6)
        neigh = [c[j] for j in range(lo, hi) if j != i]
        if not neigh:
            continue
        nm = float(np.mean(neigh))
        prom = float('inf') if nm <= 0 else c[i] / nm
        if prom >= 2.0:
            cands.append((i, prom))
    keep = []
    for i, pr in cands:
        drop_cur = False
        while keep and i - keep[-1][0] < p['d3_gap_min']:
            if pr > keep[-1][1]:
                keep.pop()
            else:
                drop_cur = True
                break
        if not drop_cur:
            keep.append((i, pr))
    return [{'start': i, 'end': i, 'raw': pr} for i, pr in keep]


def detect_d4(c, p):
    """spectral flux onset（MIR 迁移）。"""
    L = len(c)
    if L < 4:
        return []
    c = np.asarray(c, dtype=float)
    d = np.diff(np.concatenate([[0.0], c]))
    d = np.maximum(d, 0.0)
    cs = np.concatenate([[0.0], np.cumsum(d)])
    Fv = np.array([cs[i + 1] - cs[max(0, i - 2)] for i in range(L)])
    peaks = []
    for i in platform_peaks(Fv):
        lo, hi = max(0, i - 10), min(L, i + 10)
        w = Fv[lo:hi]
        thr = float(np.mean(w)) + p['flux_delta'] * float(np.std(w))
        if Fv[i] > 0 and Fv[i] > thr:
            peaks.append(i)
    hits = [i for i in peaks]
    return [{'start': s, 'end': e, 'raw': float(np.max(Fv[s:e + 1]))}
            for s, e in segment_hits(hits, p['merge_gap'])]


def norm_minmax(segs):
    """候选原始分 → [0,1] min-max（单候选/同分 → 1.0；08 §4.2）。"""
    if not segs:
        return segs
    raws = [s['raw'] for s in segs]
    lo, hi = min(raws), max(raws)
    for s in segs:
        s['score'] = 1.0 if hi <= lo else (s['raw'] - lo) / (hi - lo)
    return segs


def _iou(a, b):
    inter = min(a['end'], b['end']) - max(a['start'], b['start']) + 1
    if inter <= 0:
        return 0.0
    union = (a['end'] - a['start'] + 1) + (b['end'] - b['start'] + 1) - inter
    return inter / union


def fuse_video(by_method, w, theta):
    """跨法 IoU>0.3 聚合（并查集）→ score_fused=Σ w_m·score_m → θ 截断。"""
    items = [(m, s) for m in METHODS for s in by_method.get(m, [])]
    if not items:
        return []
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _iou(items[i][1], items[j][1]) > IOU_THR:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    clusters = {}
    for idx in range(len(items)):
        clusters.setdefault(find(idx), []).append(idx)
    events = []
    for idxs in clusters.values():
        scores = {m: 0.0 for m in METHODS}
        for k in idxs:
            m, s = items[k]
            scores[m] = max(scores[m], s['score'])
        fused = sum(w[m] * scores[m] for m in METHODS)
        if fused >= theta:
            events.append({
                'start': min(items[k][1]['start'] for k in idxs),
                'end': max(items[k][1]['end'] for k in idxs),
                'score_fused': fused, **{f'score_{m}': scores[m] for m in METHODS},
                'n_methods': sum(1 for v in scores.values() if v > 0),
            })
    events.sort(key=lambda e: (e['start'], e['end']))
    for k, e in enumerate(events):
        e['event_id'] = k
    return events


def run_detectors(w_series, params):
    """单视频：四检测器 → 归一 → 融合。返回融合事件列表（dict）。纯函数、无随机。"""
    d = params['detect']
    c = np.asarray(w_series, dtype=float)
    by_method = {
        'd1': norm_minmax(detect_d1(c, d)),
        'd2': norm_minmax(detect_d2(c, d)),
        'd3': norm_minmax(detect_d3(c, d)),
        'd4': norm_minmax(detect_d4(c, d)),
    }
    w = dict(zip(METHODS, d['fusion_w']))
    return fuse_video(by_method, w, d['fusion_theta'])


# ══════════════════ Spark 作业 ══════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config', default='/opt/work/config/params.yaml')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    import yaml
    with open(args.config, encoding='utf-8') as f:
        params = yaml.safe_load(f)
    d = params['detect']
    min_dm = int(params.get('l2', {}).get('min_danmaku', 100))

    b = (SparkSession.builder.appName('m2_l2_events')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # 每视频：稀疏点集 + n_danmaku（门槛）
    d5 = spark.read.parquet(f'{wh}/dws_density_5s')
    fp = (spark.read.parquet(f'{wh}/dws_video_rhythm_features')
          .select('bvid', 'batch_date', 'n_danmaku'))
    grouped = (d5.select('bvid', 'batch_date',
                         F.struct(F.col('bucket5s').alias('b'), F.col('wcnt').alias('v')).alias('pt'))
               .groupBy('bvid', 'batch_date')
               .agg(F.sort_array(F.collect_list('pt')).alias('pts'))
               .join(fp, ['bvid', 'batch_date'], 'left'))

    n_total = grouped.count()
    n_skipped = grouped.filter(F.coalesce(F.col('n_danmaku'), F.lit(0)) < min_dm).count()

    bucket_sec = int(d['bucket_sec'])

    def per_video(row):
        """RDD map：Row → 事件 tuple 列表（跨视频数据并行）。"""
        if row['pts'] is None or row['n_danmaku'] is None or row['n_danmaku'] < min_dm:
            return []
        mx = max(pt['b'] for pt in row['pts'])
        series = np.zeros(int(mx) + 1, dtype=float)
        for pt in row['pts']:
            series[int(pt['b'])] = float(pt['v'] or 0.0)
        out = []
        for e in run_detectors(series, params):
            out.append((row['bvid'], row['batch_date'], int(e['event_id']),
                        int(e['start']), int(e['end']),
                        int(e['start']) * bucket_sec, (int(e['end']) + 1) * bucket_sec,
                        float(e['score_fused']), float(e['score_d1']), float(e['score_d2']),
                        float(e['score_d3']), float(e['score_d4']), int(e['n_methods']),
                        'uncalibrated', ENGINE_VERSION))
        return out

    rows = grouped.rdd.flatMap(per_video)

    schema = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.StringType()),
        T.StructField('event_id', T.IntegerType()),
        T.StructField('start_bucket', T.IntegerType()), T.StructField('end_bucket', T.IntegerType()),
        T.StructField('start_sec', T.IntegerType()), T.StructField('end_sec', T.IntegerType()),
        T.StructField('score_fused', T.DoubleType()),
        T.StructField('score_d1', T.DoubleType()), T.StructField('score_d2', T.DoubleType()),
        T.StructField('score_d3', T.DoubleType()), T.StructField('score_d4', T.DoubleType()),
        T.StructField('n_methods', T.IntegerType()),
        T.StructField('calibrated_on', T.StringType()), T.StructField('engine_version', T.StringType()),
    ])
    ev = spark.createDataFrame(rows, schema=schema)
    ev.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_rhythm_events')
    n_ev = ev.count()
    n_videos_with = ev.select('bvid', 'batch_date').distinct().count()

    report = {
        'job': 'm2_l2_events', 'engine_version': ENGINE_VERSION,
        'n_video_batches': n_total, 'n_skipped_lt_min_danmaku': n_skipped,
        'min_danmaku': min_dm,
        'n_events': n_ev, 'n_videos_with_events': n_videos_with,
        'fusion_w': d['fusion_w'], 'fusion_theta': d['fusion_theta'],
        'calibrated_on': 'uncalibrated',
        'note': 'DoD「融合 F1 ≥ 最佳单法」待 30 视频人工标注完成（B3.3）后网格搜索标定',
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm2_l2_events_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m2_l2_events：{n_ev} 事件（{n_videos_with} 个视频有事件；'
          f'跳过 {n_skipped} 个 <{min_dm} 弹幕的视频）')
    spark.stop()


if __name__ == '__main__':
    main()