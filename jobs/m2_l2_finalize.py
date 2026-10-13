#!/usr/bin/env python3
"""
m2_l2_finalize.py — L2 收尾：回填 L1 指纹（08 §4 输出「回填 L1」）+ feat_z_part 配套列（08 §3.2）

输入：dws_video_rhythm_features + dws_rhythm_events（peak_*）+ dws_sync_features（herd_mean）
      + dws_lag_calibration（分区 δ）+ dim_video（tname）+ dws_partition_rhythm（分区 med/MAD）
输出：重写 {wh}/dws_video_rhythm_features：
  - peak_cnt（事件数）/ peak_max_z（最大融合分）/ peak_gap_cv（事件起点间隔 CV）
  - herd_mean（sync 表）/ lag_delta（分区级 δ，eligible 才非空）
    ※ n_danmaku < l2.min_danmaku 的跳过视频 → 以上 L2 列全 NULL（08 §4 降级条）
    ※ replay_ratio 保持 NULL（历史批次无 user_hash——占位维持）
  - <feat>_z_part（16 个基准特征稳健 z = (x − med_part) / (MAD_part×1.4826) 的已缩放 MAD；
      low_support 分区或 MAD=0 → NULL）——基线 med/MAD 来自 m2_l1_baseline
  + 报告 data-out/m2_l2_finalize_report.json
幂等：全量覆盖写；纯确定性 → 重跑一致（08 §0.1）。
"""

import argparse
import json
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window

from m2_l1_baseline import FEAT_COLS as Z_FEATS

ENGINE_VERSION = 'b2-v1'


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
    min_dm = int(params.get('l2', {}).get('min_danmaku', 100))

    b = (SparkSession.builder.appName('m2_l2_finalize')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # ⚠️ 自读自写保护：本作业回写指纹表（overwrite）——先物化 fp 再写
    #    （惰性扫描在 overwrite 删旧文件后 FileNotFound；机制详见 m2_l2_sync 头注，2026-10-13 实测）
    fp = spark.read.parquet(f'{wh}/dws_video_rhythm_features').cache()
    fp.count()
    ev = spark.read.parquet(f'{wh}/dws_rhythm_events')

    # ── 事件聚合：数量 / 最大融合分 / 起点间隔 CV ──
    g = (ev.groupBy('bvid', 'batch_date')
         .agg(F.count('*').alias('n_ev'),
              F.max('score_fused').alias('gap_max_z_src')))
    w = Window.partitionBy('bvid', 'batch_date').orderBy('start_bucket')
    gaps = (ev.withColumn('gap', F.col('start_bucket') - F.lag('start_bucket').over(w))
            .filter(F.col('gap').isNotNull())
            .groupBy('bvid', 'batch_date')
            .agg(F.stddev('gap').alias('gs'), F.avg('gap').alias('gm'))
            .withColumn('peak_gap_cv_src', F.when(F.col('gm') > 0, F.col('gs') / F.col('gm'))))

    dv = spark.read.parquet(f'{wh}/dim_video').select('bvid', 'tname').dropDuplicates(['bvid'])
    sync = (spark.read.parquet(f'{wh}/dws_sync_features')
            .select('bvid', 'batch_date', F.col('herd_mean').alias('herd_mean_src')))
    calib = (spark.read.parquet(f'{wh}/dws_lag_calibration')
             .select('tname', 'delta_hat_sec'))

    filled = (fp.join(dv, 'bvid', 'left')
              .join(g, ['bvid', 'batch_date'], 'left')
              .join(gaps, ['bvid', 'batch_date'], 'left')
              .join(sync, ['bvid', 'batch_date'], 'left')
              .join(calib, 'tname', 'left')
              .withColumn('eligible', F.col('n_danmaku') >= min_dm)
              .withColumn('peak_cnt', F.when(F.col('eligible'),
                          F.coalesce(F.col('n_ev'), F.lit(0))).cast('int'))
              .withColumn('peak_max_z', F.when(F.col('eligible'), F.col('gap_max_z_src')))
              .withColumn('peak_gap_cv', F.when(F.col('eligible'), F.col('peak_gap_cv_src')))
              .withColumn('herd_mean', F.when(F.col('eligible'), F.col('herd_mean_src')))
              .withColumn('lag_delta', F.when(F.col('eligible'), F.col('delta_hat_sec'))))

    # ── feat_z_part（08 §3.2）：稳健 z 配套列（基准 med/MAD；low_support → NULL）──
    pr = spark.read.parquet(f'{wh}/dws_partition_rhythm')
    keep = ['tname', 'low_support'] + [f'{c}_med' for c in Z_FEATS] + [f'{c}_mad' for c in Z_FEATS]
    filled = filled.join(pr.select(*keep), 'tname', 'left')
    z_cols = []
    for c in Z_FEATS:
        zc = f'{c}_z_part'
        filled = filled.withColumn(
            zc,
            F.when(F.col('low_support') | (F.col(f'{c}_mad') == 0), F.lit(None))
             .otherwise((F.col(c) - F.col(f'{c}_med')) / F.col(f'{c}_mad')))
        z_cols.append(zc)

    # ── 组装（原列序 + 回填列在位 + z 列追加）──
    # ⚠️ 幂等（08 §0.1）：fp 若已含上一轮写入的 *_z_part 列，须先剔除再追加，
    #    否则二次运行 select 出现重名列 → COLUMN_ALREADY_EXISTS（2026-10-13 实测踩坑）。
    src_cols = [c for c in fp.columns if c != 'batch_date' and c not in z_cols]
    final = filled.select(*src_cols, *z_cols, F.col('batch_date'))
    final.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_video_rhythm_features')
    n = final.count()

    n_peak = final.filter(F.col('peak_cnt').isNotNull()).count()
    n_z = final.filter(F.col('dm_per_min_z_part').isNotNull()).count()
    n_skip = final.filter(F.col('n_danmaku') < min_dm).count()
    report = {'job': 'm2_l2_finalize', 'engine_version': ENGINE_VERSION,
              'n_rows': n, 'min_danmaku': min_dm, 'n_skipped': n_skip,
              'n_peak_cnt_filled': n_peak, 'n_z_filled': n_z,
              'z_feats': len(Z_FEATS),
              'note': 'replay_ratio 保持 NULL（历史批次无 user_hash）'}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm2_l2_finalize_report.json'), 'w', encoding='utf-8') as fp2:
        json.dump(report, fp2, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m2_l2_finalize：{n} 行指纹回填完成（peak {n_peak}；z {n_z}；跳过 {n_skip}）')
    spark.stop()


if __name__ == '__main__':
    main()