#!/usr/bin/env python3
"""
probe_l2_density_contract.py — L2 输入契约诊断（前置②，2026-10-13 交接）

问题：dws_density_1pct=45,074 ≠ 509×101=51,409；dws_density_5s=70,007。
诊断项（全部一手重算，改写自 dwd_danmaku_clean）：
  1. 两表存储行数 vs dwd 重算的非空桶集合（anti-join 双向：丢失/多余）
  2. 桶域与域内空洞（内部空桶数）；1pct 尾部（max bucket→99）空桶量化
  3. NULL offset_pct 行排除量（duration<=0 的视频在 1pct 表缺失的原因）
  4. 稠密口径对照：1pct=[0,99]→nv*100；含 clamp 前 [0,100]→nv*101；5s=Σ域全长
  5. dwd_danmaku_sent 状态取值（'neutral'/'objective' 命名现状——12 §2 重命名未执行）
产出：stdout + /opt/work/data-out/l2_density_contract_probe.json
"""
import argparse
import json
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--out-name', default='l2_density_contract_probe.json')
    args = ap.parse_args()
    wh = args.wh_root

    spark = (SparkSession.builder.appName('probe_l2_density_contract')
             .config('spark.sql.shuffle.partitions', '8')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')

    out = {}
    dwd = spark.read.parquet(f'{wh}/dwd_danmaku_clean').filter(~F.col('is_spam'))

    # ── 0) dwd 基况 ──
    agg0 = dwd.agg(F.count('*').alias('n'),
                   F.sum(F.when(F.col('offset_pct').isNull(), 1).otherwise(0)).alias('null_pct'),
                   F.sum(F.when(F.col('time_offset').isNull(), 1).otherwise(0)).alias('null_off'),
                   F.sum(F.when(F.col('offset_pct') >= 0.99, 1).otherwise(0)).alias('pct_ge_099'),
                   F.min('time_offset').alias('min_off')).collect()[0]
    n_vid_pctnull = (dwd.filter(F.col('offset_pct').isNull())
                     .select('bvid').distinct().count())
    out['dwd'] = {'n_rows': agg0['n'], 'null_offset_pct': agg0['null_pct'],
                  'null_time_offset': agg0['null_off'], 'pct_ge_0.99': agg0['pct_ge_099'],
                  'min_time_offset': float(agg0['min_off'] or 0),
                  'n_videos_with_null_pct': n_vid_pctnull}

    # ── 1) 重算非空桶集合 ──
    d5set = (dwd.filter(F.col('time_offset').isNotNull())
             .select('bvid', 'batch_date', F.floor(F.col('time_offset') / 5).cast('long').alias('bucket5s'))
             .distinct())
    d1set = (dwd.filter(F.col('offset_pct').isNotNull())
             .select('bvid', 'batch_date', F.least(F.floor(F.col('offset_pct') * 100), F.lit(99)).cast('int').alias('bucket1pct'))
             .distinct())
    t5 = spark.read.parquet(f'{wh}/dws_density_5s')
    t1 = spark.read.parquet(f'{wh}/dws_density_1pct')

    out['recomputed_nonempty'] = {'density5s': d5set.count(), 'density1pct': d1set.count()}
    out['stored'] = {'density5s': t5.count(), 'density1pct': t1.count()}

    # ── 2) 双向 anti-join（丢失应为 0 = 存储忠实于非空集合）──
    miss5 = d5set.join(t5.select('bvid', 'batch_date', 'bucket5s'), ['bvid', 'batch_date', 'bucket5s'], 'left_anti').count()
    extra5 = t5.select('bvid', 'batch_date', 'bucket5s').join(d5set, ['bvid', 'batch_date', 'bucket5s'], 'left_anti').count()
    miss1 = d1set.join(t1.select('bvid', 'batch_date', 'bucket1pct'), ['bvid', 'batch_date', 'bucket1pct'], 'left_anti').count()
    extra1 = t1.select('bvid', 'batch_date', 'bucket1pct').join(d1set, ['bvid', 'batch_date', 'bucket1pct'], 'left_anti').count()
    out['set_compare'] = {'density5s_missing': miss5, 'density5s_extra': extra5,
                          'density1pct_missing': miss1, 'density1pct_extra': extra1}

    # ── 3) 域与空洞 ──
    g5 = (t5.groupBy('bvid', 'batch_date')
          .agg(F.count('*').alias('n'), F.min('bucket5s').alias('mn'), F.max('bucket5s').alias('mx'))
          .withColumn('holes', F.col('mx') + 1 - F.col('n')))
    g1 = (t1.groupBy('bvid', 'batch_date')
          .agg(F.count('*').alias('n'), F.min('bucket1pct').alias('mn'), F.max('bucket1pct').alias('mx'))
          .withColumn('holes', F.col('mx') + 1 - F.col('n')))

    a5 = g5.agg(F.count('*').alias('nv'), F.sum('n').alias('sum_n'),
                F.sum('holes').alias('holes'), F.max('holes').alias('max_holes'),
                F.avg('holes').alias('avg_holes'),
                F.sum(F.when(F.col('mn') > 0, 1).otherwise(0)).alias('head_gap_videos'),
                F.avg('mx').alias('avg_mx')).collect()[0]
    a1 = g1.agg(F.count('*').alias('nv'), F.sum('n').alias('sum_n'),
                F.sum('holes').alias('holes'), F.max('holes').alias('max_holes'),
                F.avg('holes').alias('avg_holes'),
                F.sum(F.when(F.col('mn') > 0, 1).otherwise(0)).alias('head_gap_videos'),
                F.avg('mx').alias('avg_mx')).collect()[0]
    tail1 = g1.withColumn('tail', F.lit(99) - F.col('mx')).agg(
        F.sum('tail').alias('sum_tail'), F.avg('tail').alias('avg_tail'),
        F.sum(F.when(F.col('mx') < 99, 1).otherwise(0)).alias('videos_ending_before_99')).collect()[0]

    out['table5s'] = {'video_batches': a5['nv'], 'sum_rows': a5['sum_n'],
                      'sum_internal_holes': a5['holes'], 'max_holes': a5['max_holes'],
                      'avg_holes': float(a5['avg_holes'] or 0),
                      'head_gap_videos': a5['head_gap_videos'],
                      'avg_max_bucket': float(a5['avg_mx'] or 0)}
    out['table1pct'] = {'video_batches': a1['nv'], 'sum_rows': a1['sum_n'],
                        'sum_internal_holes': a1['holes'], 'max_holes': a1['max_holes'],
                        'avg_holes': float(a1['avg_holes'] or 0),
                        'head_gap_videos': a1['head_gap_videos'],
                        'avg_max_bucket': float(a1['avg_mx'] or 0),
                        'sum_tail_zeros_to99': int(tail1['sum_tail'] or 0),
                        'avg_tail_zeros_to99': float(tail1['avg_tail'] or 0),
                        'videos_ending_before_99': tail1['videos_ending_before_99']}

    out['dense_expect'] = {
        'density1pct_100buckets_nv100': int((a1['nv'] or 0)) * 100,
        'density1pct_101buckets_nv101': int((a1['nv'] or 0)) * 101,
        'density5s_domain_full': int(a5['sum_n'] or 0) + int(a5['holes'] or 0),
        'note': '1pct 表按实现口径 clamp 到 [0,99]；5s 域 = [0, 各视频 max] 数据决定',
    }

    q5 = g5.approxQuantile('n', [0.0, 0.25, 0.5, 0.75, 1.0], 0.01)
    q1 = g1.approxQuantile('n', [0.0, 0.25, 0.5, 0.75, 1.0], 0.01)
    out['per_video_rows_quantiles'] = {'density5s': q5, 'density1pct': q1}

    # ── 4) 5s 域末 vs duration ──
    dv = (spark.read.parquet(f'{wh}/dim_video')
          .select('bvid', F.col('duration').cast('double').alias('dur')).dropDuplicates(['bvid']))
    sp = g5.join(dv, 'bvid', 'left').select((F.col('mx') * 5).alias('dom_end'), 'dur')
    a = sp.agg(F.avg('dom_end').alias('avg_dom_end_sec'), F.avg('dur').alias('avg_dur'),
               F.sum(F.when(F.col('dom_end') > F.col('dur'), 1).otherwise(0)).alias('n_dom_end_gt_dur')).collect()[0]
    out['span_5s'] = {'avg_domain_end_sec': float(a['avg_dom_end_sec'] or 0),
                      'avg_duration': float(a['avg_dur'] or 0),
                      'n_videos_dom_end_gt_dur': a['n_dom_end_gt_dur']}

    # ── 5) dwd_danmaku_sent 状态取值 + quote 完整性 ──
    sent = spark.read.parquet(f'{wh}/dwd_danmaku_sent')
    sc = sent.groupBy('sent_state').count().collect()
    out['sent_states'] = {r['sent_state']: r['count'] for r in sc}
    out['sent_cols'] = sent.columns
    out['sent_rows'] = sent.count()

    # ── 6) 指纹表现状 ──
    fp = spark.read.parquet(f'{wh}/dws_video_rhythm_features')
    out['fingerprint_rows'] = fp.count()
    out['fingerprint_cols'] = fp.columns
    nulls = fp.agg(*[F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
                     for c in ['peak_cnt', 'peak_max_z', 'peak_gap_cv', 'herd_mean', 'lag_delta', 'replay_ratio']]).collect()[0].asDict()
    out['fingerprint_null_counts'] = nulls
    # NaN 检查（DoD：无 NaN；Spark 中 NaN != NaN，用自比较捕获）
    num_cols = [f.name for f in fp.schema.fields if f.dataType.simpleString() in ('double', 'float')]
    nan_counts = fp.agg(*[F.sum(F.when(F.col(c).isNotNull() & (F.col(c) != F.col(c)), 1).otherwise(0)).alias(c)
                          for c in num_cols]).collect()[0].asDict()
    out['fingerprint_nan_counts'] = {k: v for k, v in nan_counts.items() if v}
    if 'no_hit_ratio' in fp.columns:
        s = fp.agg(F.avg('pos_ratio').alias('pos'), F.avg('neg_ratio').alias('neg'),
                   F.avg('neu_ratio').alias('neu'), F.avg('no_hit_ratio').alias('no_hit')).collect()[0]
        out['fingerprint_ratio_avgs'] = {k: float(s[k]) for k in ('pos', 'neg', 'neu', 'no_hit')}

    # ── 7) L2 产出核验（2026-10-13 批次；表不存在时记为 error）──
    try:
        ev = spark.read.parquet(f'{wh}/dws_rhythm_events')
        sy = spark.read.parquet(f'{wh}/dws_sync_features')
        lg = spark.read.parquet(f'{wh}/dws_lag_calibration')
        sg = spark.read.parquet(f'{wh}/dws_segments')
        out['l2'] = {
            'events_rows': ev.count(),
            'events_videos': ev.select('bvid', 'batch_date').distinct().count(),
            'events_sync_p_null': ev.filter(F.col('sync_p').isNull()).count(),
            'events_delta_null': ev.filter(F.col('delta_hat_sec').isNull()).count(),
            'events_uncalibrated': ev.filter(F.col('calibrated_on') == 'uncalibrated').count(),
            'sync_rows': sy.count(),
            'sync_herd_null': sy.filter(F.col('herd_mean').isNull()).count(),
            'lag_rows': lg.count(),
            'lag_eligible': lg.filter('eligible').count(),
            'segments_rows': sg.count(),
        }
        bf = fp.agg(
            F.sum(F.when(F.col('peak_cnt').isNotNull(), 1).otherwise(0)).alias('peak_filled'),
            F.sum(F.when(F.col('herd_mean').isNotNull(), 1).otherwise(0)).alias('herd_filled'),
            F.sum(F.when(F.col('lag_delta').isNotNull(), 1).otherwise(0)).alias('lag_delta_filled'),
            F.sum(F.when(F.col('peak_max_z').isNotNull(), 1).otherwise(0)).alias('peak_max_z_filled')).collect()[0].asDict()
        out['fingerprint_backfill'] = bf
        out['fingerprint_z_cols'] = len([c for c in fp.columns if c.endswith('_z_part')])
    except Exception as e:
        out['l2'] = {'error': repr(e)}

    os.makedirs(args.out_dir, exist_ok=True)
    p = os.path.join(args.out_dir, args.out_name)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f'✅ 探针完成 → {p}')
    spark.stop()


if __name__ == '__main__':
    main()