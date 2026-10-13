#!/usr/bin/env python3
"""
m2_l2_lag.py — 预警-回溯锚滞后校正（08 §4.5）

输入：dws_rhythm_events（含同步列的事件表）+ dwd_danmaku_sent（is_anchor / anchor_type，M5 打标）
      + dim_video（duration / tname）+ dws_density_5s（校正曲线回写）
输出：① 回写 dws_rhythm_events（追加 delta_hat_sec / n_anchors / n_anchors_aft / n_anchors_fore）
      ② {wh}/dws_lag_calibration —— 每 tname 一行（分区级 δ；DoD：n_anchors < 30 不输出 δ）
      ③ 回写 dws_density_5s（追加 cal_cnt：曲线左移 δ/5 桶；无 δ 分区 = 原值）
      ④ 报告 data-out/m2_l2_lag_report.json

口径（08 §4.5）：
  - 锚弹幕 = dwd_danmaku_sent.is_anchor（词表 = M5 dict_anchor；fore=预警型 / aft=回溯型）
  - 锚秒位 = offset_pct × duration（sent 表无 time_offset，用 pct×时长还原）；桶 = floor(秒位/5)
  - 每事件：窗 [start−win, end+win] 内锚弹幕相对事件中心 center=(start+end+1)/2·5（秒）的偏移分布；
      δ_hat(事件) = median(aft 锚偏移)（回溯型 = 观众反应延迟）；无 aft 锚 → δ NULL（计数仍输出）
  - 分区级 δ = median(众事件 δ_hat)；分区锚弹幕总数 < anchor.min_anchors_part(30) 或无可估事件 → 不输出
  - 校正曲线 cal_cnt(v,b) = cnt(v, b + round(δ_part/5))（左移=从右侧读取）；越界 → 0；
      无 δ 分区 → cal_cnt = cnt（原始/校正双口径，无校正时相同）
实现：逐视频事件×锚 窗扫描 → 数据并行（RDD）；纯确定性计算（无随机）→ 重跑幂等（08 §0.1）。
"""

import argparse
import json
import os

import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b2-v1'


def per_video_lag(kv, win, bucket_sec):
    """单视频：事件 × 锚弹幕 → δ_hat 行（纯函数，可单测）。"""
    (bvid, batch_date), (evs, ans) = kv
    evs = sorted(evs)
    ans = list(ans)
    if not evs:
        return []
    out = []
    for eid, s, e in evs:
        w0, w1 = s - win, e + win
        center = (s + e + 1) / 2.0 * bucket_sec
        sel = [(t, sec) for (t, sec, bk) in ans if w0 <= bk <= w1]
        n_aft = sum(1 for t, _ in sel if t == 'aft')
        n_fore = len(sel) - n_aft
        aft = [sec - center for t, sec in sel if t == 'aft']
        delta = float(np.median(aft)) if aft else None
        out.append((bvid, batch_date, int(eid), delta, len(sel), n_aft, n_fore, float(center)))
    return out


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
    win = int(params['anchor']['win_bucket'])
    min_anch = int(params['anchor']['min_anchors_part'])
    bucket_sec = int(params['detect']['bucket_sec'])

    b = (SparkSession.builder.appName('m2_l2_lag')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # ⚠️ 自读自写保护：本作业回写 dws_rhythm_events / dws_density_5s（overwrite）——先物化再写
    #    （惰性扫描在 overwrite 删旧文件后 FileNotFound；机制详见 m2_l2_sync 头注，2026-10-13 实测）
    ev = spark.read.parquet(f'{wh}/dws_rhythm_events').cache()
    ev.count()
    dv = (spark.read.parquet(f'{wh}/dim_video')
          .select('bvid', 'duration', 'tname').dropDuplicates(['bvid']))
    anch = (spark.read.parquet(f'{wh}/dwd_danmaku_sent')
            .filter(F.col('is_anchor'))
            .select('bvid', 'batch_date', 'anchor_type', 'offset_pct')
            .join(dv.select('bvid', 'duration', 'tname'), 'bvid')
            .withColumn('sec', F.col('offset_pct') * F.col('duration').cast('double'))
            .withColumn('bucket', F.floor(F.col('sec') / bucket_sec).cast('int'))
            .select('bvid', 'batch_date', 'anchor_type', 'sec', 'bucket', 'tname'))

    # ── 逐视频：事件窗 × 锚弹幕 → 事件级 δ ──
    # ⚠️ cogroup 自身完成分组——输入必须是原始 (k,v) RDD（双重分组的坑见 m2_l2_sync 头注）
    ev_r = (ev.select('bvid', 'batch_date', 'event_id', 'start_bucket', 'end_bucket')
            .rdd.map(lambda r: ((r['bvid'], r['batch_date']),
                                (int(r['event_id']), int(r['start_bucket']), int(r['end_bucket'])))))
    an_r = (anch.select('bvid', 'batch_date', 'anchor_type', 'sec', 'bucket')
            .rdd.map(lambda r: ((r['bvid'], r['batch_date']),
                                (r['anchor_type'], float(r['sec']), int(r['bucket'])))))

    def f(kv):
        return per_video_lag(kv, win, bucket_sec)

    lag_schema = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.StringType()),
        T.StructField('event_id', T.IntegerType()),
        T.StructField('delta_hat_sec', T.DoubleType()),
        T.StructField('n_anchors', T.IntegerType()),
        T.StructField('n_anchors_aft', T.IntegerType()),
        T.StructField('n_anchors_fore', T.IntegerType()),
        T.StructField('center_sec', T.DoubleType()),
    ])
    lag_ev = spark.createDataFrame(ev_r.cogroup(an_r).flatMap(f), schema=lag_schema)
    lag_ev.cache()
    n_delta = lag_ev.filter(F.col('delta_hat_sec').isNotNull()).count()
    n_ev_all = lag_ev.count()

    # ── 回写事件表（保留同步列 + δ 列）──
    ev2 = (ev.join(lag_ev, ['bvid', 'batch_date', 'event_id'], 'left')
           .select('bvid', 'batch_date', 'event_id', 'start_bucket', 'end_bucket',
                   'start_sec', 'end_sec', 'score_fused',
                   'score_d1', 'score_d2', 'score_d3', 'score_d4', 'n_methods',
                   'sync_score', 'sync_p', 'h_pos', 'h_time', 'n_window',
                   'delta_hat_sec', 'n_anchors', 'n_anchors_aft', 'n_anchors_fore',
                   'calibrated_on', 'engine_version'))
    ev2.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_rhythm_events')
    print(f'↩️ dws_rhythm_events 已回写 δ 列：{n_delta}/{n_ev_all} 事件有 δ')

    # ── 分区级 δ（DoD：n_anchors ≥ min_anchors_part 才输出）──
    part_anch = anch.groupBy('tname').agg(F.count('*').alias('n_anchors'))
    part_delta = (lag_ev.filter(F.col('delta_hat_sec').isNotNull())
                  .join(dv.select('bvid', 'tname'), 'bvid')
                  .groupBy('tname')
                  .agg(F.expr('percentile(delta_hat_sec, 0.5)').alias('_delta'),
                       F.count('*').alias('n_events_with_delta')))
    base72 = spark.read.parquet(f'{wh}/dws_partition_rhythm').select('tname')
    calib = (base72.join(part_anch, 'tname', 'left')
             .join(part_delta, 'tname', 'left')
             .withColumn('n_anchors', F.coalesce(F.col('n_anchors'), F.lit(0)))
             .withColumn('eligible', (F.col('n_anchors') >= min_anch) & F.col('_delta').isNotNull())
             .withColumn('delta_hat_sec', F.when(F.col('eligible'), F.col('_delta')).otherwise(F.lit(None)))
             .withColumn('engine_version', F.lit(ENGINE_VERSION))
             .select('tname', 'n_anchors', 'n_events_with_delta',
                     'delta_hat_sec', 'eligible', 'engine_version'))
    calib.write.mode('overwrite').parquet(f'{wh}/dws_lag_calibration')
    n_elig = calib.filter('eligible').count()
    n_calib = calib.count()

    # ── 校正曲线回写 dws_density_5s（cal_cnt）──
    d5 = spark.read.parquet(f'{wh}/dws_density_5s').cache()
    d5.count()
    vmap = dv.select('bvid', 'tname')
    shift = (calib.select('tname', 'delta_hat_sec')
             .withColumn('shift', F.when(F.col('delta_hat_sec').isNull(), F.lit(0))
                         .otherwise(F.round(F.col('delta_hat_sec') / bucket_sec).cast('int'))))
    a = (d5.join(vmap, 'bvid', 'left').join(shift, 'tname', 'left')
         .withColumn('shift', F.coalesce(F.col('shift'), F.lit(0)))
         .select('bvid', 'batch_date', 'bucket5s', 'cnt', 'wcnt',
                 F.col('shift').cast('int').alias('shift')))
    s = (d5.select(F.col('bvid').alias('s_bvid'), F.col('batch_date').alias('s_bd'),
                   F.col('bucket5s').alias('s_bkt'), F.col('cnt').alias('s_cnt')))
    out5 = (a.join(s, (F.col('bvid') == F.col('s_bvid')) & (F.col('batch_date') == F.col('s_bd'))
                   & (F.col('s_bkt') == F.col('bucket5s') + F.col('shift')), 'left')
            .withColumn('cal_cnt', F.coalesce(F.col('s_cnt'), F.lit(0)).cast('int'))
            .withColumn('engine_version', F.lit(ENGINE_VERSION))
            .select('bvid', 'batch_date', 'bucket5s', 'cnt', 'wcnt', 'cal_cnt', 'engine_version'))
    out5.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_density_5s')
    n5 = out5.count()

    report = {'job': 'm2_l2_lag', 'engine_version': ENGINE_VERSION,
              'n_events': n_ev_all, 'n_events_with_delta': n_delta,
              'n_partitions': n_calib, 'n_partitions_eligible': n_elig,
              'min_anchors_part': min_anch,
              'density_rows_rewritten': n5}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm2_l2_lag_report.json'), 'w', encoding='utf-8') as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m2_l2_lag：δ 覆盖 {n_delta}/{n_ev_all} 事件；分区 eligible {n_elig}/{n_calib}；'
          f'density 回写 {n5} 行（含 cal_cnt）')
    spark.stop()


if __name__ == '__main__':
    main()