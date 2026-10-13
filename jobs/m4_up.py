#!/usr/bin/env python3
"""
m4_up.py — M4 UP 主档案（08 §7「UP 档案：up_mid 聚合上榜次数/累计指标/互动结构」）

输入：dim_video（up_mid/up_name/tname）+ dwd_video_metric_snapshot（每 bvid 最新批快照）
输出：{wh}/dws_up_profile —— 每 up_mid 一行：
  up_name / n_videos / n_onlist_batches（上榜次数=视频批次数） / total_play / avg_play
  + 互动结构（比率，分母=累计 play）：like_rate / coin_rate / fav_rate / share_rate / reply_rate / dm_rate
  + main_tname（主分区）/ play_rank（按 avg_play 的池内名次）
注：互动结构「vs 分区基准」的比较在服务层页面用 ads_partition_fingerprint 作基准并列展示
  （本表提供 UP 侧比率；池级基准值随页面接口返回）。
"""
import argparse
import json
import os

from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F

ENGINE_VERSION = 'b2-v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('m4_up')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    dim = spark.read.parquet(f'{wh}/dim_video').select('bvid', 'up_mid', 'up_name', 'tname')
    snap = (spark.read.parquet(f'{wh}/dwd_video_metric_snapshot')
            .select('bvid', 'batch_date', 'play', 'like', 'coin', 'favorite', 'share',
                    'reply', 'danmaku_reported'))
    # 每 bvid 最新批快照
    w = Window.partitionBy('bvid').orderBy(F.col('batch_date').desc())
    latest = (snap.withColumn('rn', F.row_number().over(w)).filter(F.col('rn') == 1)
              .drop('rn', 'batch_date'))
    d = dim.join(latest, 'bvid', 'left')

    agg = (d.groupBy('up_mid')
           .agg(F.first('up_name').alias('up_name'),
                F.count('*').alias('n_videos'),
                F.sum(F.when(F.col('play').isNotNull(), 1).otherwise(0)).alias('n_onlist_videos'),
                F.sum('play').alias('total_play'),
                F.avg('play').alias('avg_play'),
                F.sum('like').alias('_like'), F.sum('coin').alias('_coin'),
                F.sum('favorite').alias('_fav'), F.sum('share').alias('_share'),
                F.sum('reply').alias('_reply'), F.sum('danmaku_reported').alias('_dm')))
    eps = 1.0
    agg = (agg.withColumn('like_rate', F.col('_like') / (F.col('total_play') + eps))
           .withColumn('coin_rate', F.col('_coin') / (F.col('total_play') + eps))
           .withColumn('fav_rate', F.col('_fav') / (F.col('total_play') + eps))
           .withColumn('share_rate', F.col('_share') / (F.col('total_play') + eps))
           .withColumn('reply_rate', F.col('_reply') / (F.col('total_play') + eps))
           .withColumn('dm_rate', F.col('_dm') / (F.col('total_play') + eps)))
    # 主分区 = 该 UP 视频数最多的分区
    tn = (d.groupBy('up_mid', 'tname').count()
          .withColumn('rn', F.row_number().over(
              Window.partitionBy('up_mid').orderBy(F.col('count').desc(), F.col('tname'))))
          .filter(F.col('rn') == 1).select('up_mid', F.col('tname').alias('main_tname')))
    out = (agg.join(tn, 'up_mid', 'left')
           .withColumn('play_rank', F.rank().over(Window.orderBy(F.col('avg_play').desc_nulls_last())))
           .withColumn('engine_version', F.lit(ENGINE_VERSION))
           .select('up_mid', 'up_name', 'n_videos', 'n_onlist_videos', 'total_play', 'avg_play',
                   'like_rate', 'coin_rate', 'fav_rate', 'share_rate', 'reply_rate', 'dm_rate',
                   'main_tname', 'play_rank', 'engine_version'))
    out.write.mode('overwrite').parquet(f'{wh}/dws_up_profile')
    n = out.count()
    report = {'job': 'm4_up', 'engine_version': ENGINE_VERSION, 'n_ups': n,
              'note': '每 bvid 最新批快照；比率分母=累计 play（eps=1.0）'}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm4_up_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m4_up：{n} 个 UP主档案')
    spark.stop()


if __name__ == '__main__':
    main()
