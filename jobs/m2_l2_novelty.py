#!/usr/bin/env python3
"""
m2_l2_novelty.py — Novelty 分段（Foote 棋盘核，08 §4.6）

输入：dws_density_5s（wcnt 序列，稠密化契约见 m2_l1_density）+ dwd_danmaku_sent（sent_score，经
      fingerprint.duration 还原秒位/桶位）+ dws_rhythm_events（段 sync = 与段重叠事件的 sync_score 中位数）
输出：{wh}/dws_segments —— 每 (bvid, batch_date, seg_id) 一行：
      seg_id / start,end（桶 + 秒）/ dm_cnt（段内 cnt 和）/ avg_sent / sync / engine_version
      + 报告 data-out/m2_l2_novelty_report.json

口径（08 §4.6 + 2026-10-13 实现决策）：
  - 特征向量 v_i = wcnt[i−(win−1)//2 .. i+win//2]（窗口 10 桶，居中，边缘零填充）
  - SSM = 余弦相似（零向量行 → 相似度 0）；棋盘核 = (2k)×(2k)（k=8）四象限 +/−/−/+；
      novelty(i) = Σ_{s,t} K[s,t]·SSM[i+s, i+t]（SSM 零填充；等价于核对角滑动）
  - 切分 = novelty 局部极大（平台取中点；值 > 0）→ 边界桶开启新段；末段至 L−1；
      L < win 的短视频不切分（整片一段）
  - avg_sent = 段内弹幕 sent_score 均值（有打分弹幕；空 → NULL）
  - sync = 与 [start,end] 重叠事件的 sync_score 中位数（无重叠 → NULL）
实现：逐视频 SSM/核累加 → 数据并行（RDD + NumPy）；纯确定性（无随机）→ 重跑幂等（08 §0.1）。
"""

import argparse
import json
import os

import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

from l2_common import platform_peaks

ENGINE_VERSION = 'b2-v1'


# ══════════════════ 纯函数（可单测） ══════════════════

def foote_segments(c, win, k):
    """单视频 Foote 分段；返回 [[start,end],...] 桶区间（含端点）。"""
    c = np.asarray(c, dtype=float)
    L = len(c)
    if L == 0:
        return []
    if L < win:
        return [[0, L - 1]]
    half_l = (win - 1) // 2
    idx = np.arange(L)
    FM = np.zeros((L, win))
    for j in range(win):
        src = idx + (j - half_l)
        valid = (src >= 0) & (src < L)
        FM[valid, j] = c[src[valid]]
    norms = np.linalg.norm(FM, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    Fn = FM / norms[:, None]
    S = Fn @ Fn.T
    pad = np.zeros((L + 2 * k, L + 2 * k))
    pad[k:k + L, k:k + L] = S
    ar = np.arange(L)
    nov = np.zeros(L)
    for s in range(2 * k):
        for t in range(2 * k):
            sign = 1.0 if (s < k) == (t < k) else -1.0
            nov += sign * pad[ar + s, ar + t]
    peaks = [i for i in platform_peaks(nov) if nov[i] > 0]
    bounds = [0] + peaks + [L]
    return [[bounds[i], bounds[i + 1] - 1] for i in range(len(bounds) - 1)]


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
    win = int(params['novelty']['ssm_win'])
    k = int(params['novelty']['kernel_k'])
    bucket_sec = int(params['detect']['bucket_sec'])

    b = (SparkSession.builder.appName('m2_l2_novelty')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    d5 = spark.read.parquet(f'{wh}/dws_density_5s')
    grouped = (d5.select('bvid', 'batch_date',
                         F.struct(F.col('bucket5s').alias('b'), F.col('wcnt').alias('v')).alias('pt'))
               .groupBy('bvid', 'batch_date')
               .agg(F.sort_array(F.collect_list('pt')).alias('pts')))

    def per_video(row):
        if row['pts'] is None:
            return []
        mx = max(pt['b'] for pt in row['pts'])
        series = np.zeros(int(mx) + 1, dtype=float)
        for pt in row['pts']:
            series[int(pt['b'])] = float(pt['v'] or 0.0)
        out = []
        for sid, (s, e) in enumerate(foote_segments(series, win, k)):
            out.append((row['bvid'], row['batch_date'], sid, int(s), int(e)))
        return out

    seg_schema = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.StringType()),
        T.StructField('seg_id', T.IntegerType()),
        T.StructField('start_bucket', T.IntegerType()), T.StructField('end_bucket', T.IntegerType()),
    ])
    seg = spark.createDataFrame(grouped.rdd.flatMap(per_video), schema=seg_schema)
    seg = (seg.withColumn('start_sec', F.col('start_bucket') * bucket_sec)
           .withColumn('end_sec', (F.col('end_bucket') + 1) * bucket_sec))
    seg.cache()
    n_seg = seg.count()
    n_vid = seg.select('bvid', 'batch_date').distinct().count()

    sg = seg.alias('sg')

    # dm_cnt：段 × 密度桶（区间 join）
    dmc = (sg.join(d5.alias('dd'),
                   (F.col('sg.bvid') == F.col('dd.bvid')) & (F.col('sg.batch_date') == F.col('dd.batch_date'))
                   & (F.col('dd.bucket5s') >= F.col('sg.start_bucket'))
                   & (F.col('dd.bucket5s') <= F.col('sg.end_bucket')))
           .groupBy('sg.bvid', 'sg.batch_date', 'sg.seg_id')
           .agg(F.sum('dd.cnt').alias('dm_cnt'))
           .select(F.col('bvid'), F.col('batch_date'), F.col('seg_id'), 'dm_cnt'))

    # avg_sent：段 × 打分弹幕（桶位 = offset_pct × duration / bucket_sec）
    dv = (spark.read.parquet(f'{wh}/dws_video_rhythm_features')
          .select('bvid', 'duration').dropDuplicates(['bvid']))
    sent = (spark.read.parquet(f'{wh}/dwd_danmaku_sent')
            .select('bvid', 'batch_date', 'sent_score', 'offset_pct')
            .join(dv, 'bvid')
            .withColumn('sb', F.floor(F.col('offset_pct') * F.col('duration') / bucket_sec).cast('int')))
    snt = (sg.join(sent.alias('ss'),
                   (F.col('sg.bvid') == F.col('ss.bvid')) & (F.col('sg.batch_date') == F.col('ss.batch_date'))
                   & (F.col('ss.sb') >= F.col('sg.start_bucket'))
                   & (F.col('ss.sb') <= F.col('sg.end_bucket')))
           .groupBy('sg.bvid', 'sg.batch_date', 'sg.seg_id')
           .agg(F.avg('ss.sent_score').alias('avg_sent'))
           .select(F.col('bvid'), F.col('batch_date'), F.col('seg_id'), 'avg_sent'))

    # sync：段 × 重叠事件（sync_score 中位数）
    ev = spark.read.parquet(f'{wh}/dws_rhythm_events').select(
        'bvid', 'batch_date', 'start_bucket', 'end_bucket', 'sync_score')
    syc = (sg.join(ev.alias('ev'),
                   (F.col('sg.bvid') == F.col('ev.bvid')) & (F.col('sg.batch_date') == F.col('ev.batch_date'))
                   & (F.col('ev.start_bucket') <= F.col('sg.end_bucket'))
                   & (F.col('ev.end_bucket') >= F.col('sg.start_bucket')))
           .groupBy('sg.bvid', 'sg.batch_date', 'sg.seg_id')
           .agg(F.expr('percentile(sync_score, 0.5)').alias('sync'))
           .select(F.col('bvid'), F.col('batch_date'), F.col('seg_id'), 'sync'))

    out = (seg.join(dmc, ['bvid', 'batch_date', 'seg_id'], 'left')
           .join(snt, ['bvid', 'batch_date', 'seg_id'], 'left')
           .join(syc, ['bvid', 'batch_date', 'seg_id'], 'left')
           .withColumn('engine_version', F.lit(ENGINE_VERSION))
           .select('bvid', 'batch_date', 'seg_id', 'start_bucket', 'end_bucket',
                   'start_sec', 'end_sec', 'dm_cnt', 'avg_sent', 'sync', 'engine_version'))
    out.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_segments')
    n_out = out.count()

    report = {'job': 'm2_l2_novelty', 'engine_version': ENGINE_VERSION,
              'ssm_win': win, 'kernel_k': k,
              'n_segments': n_seg, 'n_videos': n_vid, 'n_out_rows': n_out}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm2_l2_novelty_report.json'), 'w', encoding='utf-8') as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m2_l2_novelty：{n_seg} 段（{n_vid} 视频均 {n_seg / max(n_vid, 1):.1f} 段/视频）')
    spark.stop()


if __name__ == '__main__':
    main()