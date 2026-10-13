#!/usr/bin/env python3
"""
m2_l1_fingerprint.py — L1 指纹计算器（08 §3.2）

输入：dwd_danmaku_clean（is_spam=false）+ dws_density_5s/1pct（m2_l1_density）+ dwd_danmaku_sent（M5）
输出：{wh}/dws_video_rhythm_features —— 每 (bvid, batch_date) 一行，17 维节奏指纹

口径（08 §3.2 + 12 §3.2）：
  - 情感三比例（pos/neg/neu）分母 = 命中子集（pos+neg+neu）；no_hit_ratio 单列（全体 no_hit 占比）
    ——防"全弹幕情感分布"误读；字段名与 12 §3.2 契约一致（原 obj_ratio 更名，2026-10-13）
  - ⚠️ 状态名映射：M5 现行输出 'objective'（=词典未命中）/ 'neutral'（=有词但不偏）；
    12 §2 的"四值重命名（objective→no_hit、neutral→neu）"属未执行项——本作业按现行取值聚合，
    输出字段按 12 §3.2 契约命名（no_hit_ratio / neu_ratio），并修复旧版把 'neutral' 误读为
    'neu' 的聚合 bug（导致 n_neu 恒 0、三比例分母退化为 pos+neg）
  - 回填占位（L2/m6 后回填，本期置 NULL）：peak_cnt / peak_max_z / peak_gap_cv / herd_mean / lag_delta / replay_ratio
  - feat_z_part（每特征稳健 z）由 m2_l1_baseline 回填，本作业只产原始值
  - 全量覆盖写、幂等（08 §0.1）
"""
import argparse

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b2-v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('m2_l1_fingerprint')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean')
           .filter(~F.col('is_spam')))

    # ── 主聚合（体积/位置/时间/样式/长度）──
    base = (dwd.groupBy('bvid', 'batch_date')
            .agg(F.count('*').alias('n_danmaku'),
                 F.sum('attr_weight').alias('wsum'),
                 F.max('duration').alias('duration'),
                 F.avg(F.when(F.col('offset_pct') < 0.05, 1.0).otherwise(0.0)).alias('opening_5pct'),
                 F.avg(F.when(F.col('offset_pct') > 0.95, 1.0).otherwise(0.0)).alias('ending_5pct'),
                 F.avg(F.when((F.col('offset_pct') >= 0.05) & (F.col('offset_pct') <= 0.95), 1.0).otherwise(0.0)).alias('mid_ratio'),
                 F.avg(F.when(F.col('lag_sec') < 21600, 1.0).otherwise(0.0)).alias('early_lag_ratio'),
                 F.avg(F.when(F.col('is_colored'), 1.0).otherwise(0.0)).alias('color_ratio'),
                 F.avg(F.when(F.col('is_special_mode'), 1.0).otherwise(0.0)).alias('top_mode_ratio'),
                 F.avg(F.length(F.col('text'))).alias('avg_text_len')))

    # ── rhythm_entropy（1% 桶分布，−Σp·ln p ÷ ln100）──
    d1 = spark.read.parquet(f'{wh}/dws_density_1pct')
    ent = (d1.groupBy('bvid', 'batch_date')
           .agg(F.sum('cnt').alias('tot'),
                F.sum(F.col('cnt') * F.log(F.col('cnt'))).alias('sum_clnc'))
           .withColumn('rhythm_entropy',
                       F.when(F.col('tot') > 0,
                              (-F.col('sum_clnc') / F.col('tot') + F.log(F.col('tot')))
                              / F.log(F.lit(100.0))).otherwise(0.0))
           .select('bvid', 'batch_date', 'rhythm_entropy'))

    # ── burst_cv / w_burst_cv（5s 桶 std/mean）──
    d5 = spark.read.parquet(f'{wh}/dws_density_5s')
    burst = (d5.groupBy('bvid', 'batch_date')
             .agg(F.stddev('cnt').alias('std_c'), F.mean('cnt').alias('mean_c'),
                  F.stddev('wcnt').alias('std_w'), F.mean('wcnt').alias('mean_w'))
             .withColumn('burst_cv', F.when(F.col('mean_c') > 0, F.col('std_c') / F.col('mean_c')).otherwise(0.0))
             .withColumn('w_burst_cv', F.when(F.col('mean_w') > 0, F.col('std_w') / F.col('mean_w')).otherwise(0.0))
             .select('bvid', 'batch_date', 'burst_cv', 'w_burst_cv'))

    # ── 情感比例（口径 12 §3.2：三比例分母=命中子集 pos+neg+neu；no_hit 单列=全体占比）──
    # ⚠️ 状态名映射：M5 现行输出 'objective'（词典未命中）/ 'neutral'（有词但不偏）；
    #    12 §2 的"四值重命名"属未执行项（12 §8 清单）——本作业按现行取值聚合，输出字段按
    #    12 §3.2 契约命名；同时修复旧版把 'neutral' 误写为 'neu'（n_neu 恒 0）的聚合 bug。
    sent = (spark.read.parquet(f'{wh}/dwd_danmaku_sent')
            .groupBy('bvid', 'batch_date')
            .agg(F.count('*').alias('n_sent'),
                 F.sum(F.when(F.col('sent_state') == 'pos', 1).otherwise(0)).alias('n_pos'),
                 F.sum(F.when(F.col('sent_state') == 'neg', 1).otherwise(0)).alias('n_neg'),
                 F.sum(F.when(F.col('sent_state') == 'neutral', 1).otherwise(0)).alias('n_neu'),
                 F.sum(F.when(F.col('sent_state') == 'objective', 1).otherwise(0)).alias('n_no_hit'))
            .withColumn('hit', F.col('n_pos') + F.col('n_neg') + F.col('n_neu'))
            .withColumn('pos_ratio', F.when(F.col('hit') > 0, F.col('n_pos') / F.col('hit')).otherwise(0.0))
            .withColumn('neg_ratio', F.when(F.col('hit') > 0, F.col('n_neg') / F.col('hit')).otherwise(0.0))
            .withColumn('neu_ratio', F.when(F.col('hit') > 0, F.col('n_neu') / F.col('hit')).otherwise(0.0))
            .withColumn('no_hit_ratio', F.when(F.col('n_sent') > 0, F.col('n_no_hit') / F.col('n_sent')).otherwise(0.0))
            .select('bvid', 'batch_date', 'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio'))

    # ── 组装 ──
    feat = (base
            .join(ent, ['bvid', 'batch_date'], 'left')
            .join(burst, ['bvid', 'batch_date'], 'left')
            .join(sent, ['bvid', 'batch_date'], 'left')
            .withColumn('dm_per_min', F.when(F.col('duration') > 0,
                        F.col('n_danmaku') / (F.col('duration') / 60.0)).otherwise(0.0))
            .withColumn('wdm_per_min', F.when(F.col('duration') > 0,
                        F.col('wsum') / (F.col('duration') / 60.0)).otherwise(0.0))
            # 回填占位（L2/m6 后回填）
            .withColumn('peak_cnt', F.lit(None).cast('int'))
            .withColumn('peak_max_z', F.lit(None).cast('double'))
            .withColumn('peak_gap_cv', F.lit(None).cast('double'))
            .withColumn('herd_mean', F.lit(None).cast('double'))
            .withColumn('lag_delta', F.lit(None).cast('double'))
            .withColumn('replay_ratio', F.lit(None).cast('double'))  # 历史批次无 user_hash
            .withColumn('engine_version', F.lit(ENGINE_VERSION))
            .select('bvid', 'batch_date', 'duration', 'n_danmaku',
                    'dm_per_min', 'wdm_per_min',
                    'opening_5pct', 'ending_5pct', 'mid_ratio',
                    'rhythm_entropy', 'burst_cv', 'w_burst_cv',
                    'early_lag_ratio', 'replay_ratio',
                    'color_ratio', 'top_mode_ratio',
                    'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio',
                    'avg_text_len',
                    'peak_cnt', 'peak_max_z', 'peak_gap_cv', 'herd_mean', 'lag_delta',
                    'engine_version'))

    feat.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_video_rhythm_features')
    n = feat.count()
    print(f'✅ m2_l1_fingerprint：{n} 行（视频-批次）指纹')
    spark.stop()


if __name__ == '__main__':
    main()
