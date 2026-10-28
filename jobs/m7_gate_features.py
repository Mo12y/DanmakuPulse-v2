#!/usr/bin/env python3
"""
m7_gate_features.py — 「早期弹幕 + 元数据 → 互动质量」门控：特征表构建（2026-10-22）

用途：为 ML 门控（任务书第 1 步）产出**一张逐视频-批次的特征宽表**，
      标签与切分在 jobs/m7_gate_eval.py 里做（那边用 sklearn，便于 ROC/PR/重要性）。

输出（data-out/）：
  m7_gate_features.parquet / .csv   888 行 × 若干列
  m7_gate_features_report.json      行数/缺失/口径

特征口径（**禁止泄漏**）：
  ✅ 早期弹幕 = 播放进度前 10%（offset_pct ≤ 0.10，不是前 60 秒）
  ✅ 元数据：标题长度、标题是否含数字/问号/感叹号、分区、时长、发布小时/星期
  ⛔ 不用：最终播放量 / 点赞 / 投币 / 收藏 / 分享 / 总弹幕数（标签的分子分母）
  ⛔ 不用：reply（同批快照的结果变量，和标签同时点，属泄漏）
  ⚠️ 缺失字段如实记：简介长度 / UP 主粉丝数 / 早期评论数 —— 本仓库没有这两张源表

用法：
  make m7-gate          # 先跑本作业，再跑 eval
  docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
    /opt/work/jobs/m7_gate_features.py
"""
import argparse
import csv
import json
import math
import os

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

ENGINE_VERSION = 'm7-gate-v1'
EARLY_PCT = 0.10          # 早期窗口 = 播放进度前 10%
BUCKET_SEC = 5            # 与精彩时刻检测同一网格（峰＝5 秒桶内同文本 ≥3 条）


def _entropy(counts):
    """文本分布的 Shannon 熵（自然对数）。空 → 0。"""
    n = float(sum(counts or []))
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    args = ap.parse_args()
    wh = args.wh_root

    spark = (SparkSession.builder.appName('m7_gate_features')
             .config('spark.sql.shuffle.partitions', '8').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')

    # ── 1) 弹幕明细：只留分析口径（is_spam=false + 时间/文本有效）──
    d = (spark.read.parquet(f'{wh}/dwd_danmaku_clean')
         .filter(~F.coalesce(F.col('is_spam'), F.lit(False)))
         .filter(F.col('time_offset').isNotNull() & (F.col('time_offset') >= 0))
         .filter(F.col('text_norm').isNotNull() & (F.length('text_norm') > 0))
         .select('bvid', 'batch_date', 'user_hash', 'time_offset', 'offset_pct',
                 'text_norm', F.length('text_norm').alias('len'),
                 F.coalesce(F.col('duration'), F.lit(0)).alias('duration'),
                 F.col('pubdate').cast('long').alias('pub_ts')))
    early = d.filter(F.col('offset_pct') <= EARLY_PCT)
    early.cache()

    # 1.1 文本分布 → 熵、复读率
    text_cnt = (early.groupBy('bvid', 'batch_date', 'text_norm')
                .agg(F.count('*').alias('n')))
    entropy_udf = F.udf(_entropy, T.DoubleType())
    ent = (text_cnt.groupBy('bvid', 'batch_date')
           .agg(F.collect_list('n').alias('cnts'),
                F.count('*').alias('early_texts'),
                F.sum('n').alias('early_n_sum'),
                F.max('n').alias('early_max_text_rep'))
           .withColumn('early_entropy', entropy_udf(F.col('cnts')))
           .drop('cnts'))

    # 1.2 5 秒桶峰数（桶内同文本 ≥3 条 = 一次复读爆发）
    bucket = (early.withColumn('b', F.floor(F.col('time_offset') / F.lit(float(BUCKET_SEC))).cast('int'))
              .groupBy('bvid', 'batch_date', 'b', 'text_norm')
              .agg(F.count('*').alias('n'))
              .groupBy('bvid', 'batch_date', 'b')
              .agg(F.max('n').alias('mx'))
              .filter(F.col('mx') >= 3)
              .groupBy('bvid', 'batch_date')
              .agg(F.count('*').alias('early_peak_cnt')))

    # 1.3 基础量：条数 / 独立用户 / 短弹幕占比 / 平均长度
    base = (early.groupBy('bvid', 'batch_date')
            .agg(F.count('*').alias('early_n'),
                 F.countDistinct('user_hash').alias('early_users'),
                 F.avg(F.when(F.col('len') <= 4, 1.0).otherwise(0.0)).alias('early_short_ratio'),
                 F.avg('len').alias('early_len_mean'),
                 F.max('duration').alias('dur_from_dwd')))

    feats = (base.join(ent, ['bvid', 'batch_date'], 'left')
             .join(bucket, ['bvid', 'batch_date'], 'left')
             .fillna({'early_peak_cnt': 0, 'early_entropy': 0.0, 'early_texts': 0}))
    feats = (feats.withColumn('early_repeat_rate',
                              1.0 - F.col('early_texts') / F.greatest(F.col('early_n'), F.lit(1)))
             .withColumn('early_users_per_msg', F.col('early_users') / F.greatest(F.col('early_n'), F.lit(1)))
             .withColumn('early_per_min',
                         F.col('early_n') / F.greatest(F.col('dur_from_dwd') * F.lit(EARLY_PCT / 60.0),
                                                       F.lit(0.1))))

    # ── 2) 结果变量（只用于造标签，不进特征）──
    m = (spark.read.parquet(f'{wh}/dwd_video_metric_snapshot')
         .select('bvid', 'batch_date', 'play',
                 F.col('like').cast('double').alias('like'),
                 F.col('coin').cast('double').alias('coin'),
                 F.col('favorite').cast('double').alias('favorite'),
                 F.col('share').cast('double').alias('share'),
                 F.col('reply').cast('double').alias('reply'),
                 'danmaku_reported', 'danmaku_got', 'tname', 'title'))

    # ── 3) 元数据：时长/分区/标题/发布时间（不上 play/交互数）──
    meta = (m.select('bvid', 'batch_date', 'tname', 'title', 'play',
                     'like', 'coin', 'favorite', 'share', 'reply',
                     'danmaku_reported', 'danmaku_got',
                     F.length('title').alias('title_len'),
                     F.when(F.col('title').rlike('[0-9]'), 1).otherwise(0).alias('title_has_digit'),
                     F.when(F.col('title').rlike('[?？]'), 1).otherwise(0).alias('title_has_q'),
                     F.when(F.col('title').rlike('[!！]'), 1).otherwise(0).alias('title_has_excl')))
    # pubdate 是 epoch 秒；每视频-批次取最早一条，拆成小时 / 星期
    pub2 = (d.filter(F.col('pub_ts').isNotNull())
            .groupBy('bvid', 'batch_date').agg(F.min('pub_ts').alias('pub_ts')))
    pub3 = (pub2.withColumn('pub_hour', F.hour(F.to_timestamp(F.from_unixtime('pub_ts'))))
            .withColumn('pub_wday', F.dayofweek(F.to_timestamp(F.from_unixtime('pub_ts'))))
            .select('bvid', 'batch_date', 'pub_hour', 'pub_wday'))

    out = (meta.join(feats, ['bvid', 'batch_date'], 'left')
           .join(pub3, ['bvid', 'batch_date'], 'left')
           .fillna({'early_n': 0, 'early_users': 0, 'early_short_ratio': 0.0,
                    'early_len_mean': 0.0, 'early_per_min': 0.0,
                    'early_users_per_msg': 0.0, 'early_repeat_rate': 0.0}))
    out = out.withColumn('duration', F.col('dur_from_dwd')).drop('dur_from_dwd')
    out = out.withColumn('engine_version', F.lit(ENGINE_VERSION))
    out.cache()
    n = out.count()

    os.makedirs(args.out_dir, exist_ok=True)
    # 888 行 × 30 列：collect 到 driver，用标准库 csv 落盘。
    # ⚠️ 不用 Spark 写宿主挂载：集群模式下 file:// 落在 **executor 容器**的本地盘，
    #    无 scheme 的绝对路径又会落到 HDFS（本文件 2026-10-22 两种都实测踩过）；
    #    Spark 镜像里也没有 pandas（只有主机 venv 有）。
    cols = out.columns
    rows = out.collect()
    csv_path = os.path.join(args.out_dir, 'm7_gate_features.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow(['' if r[c] is None else r[c] for c in cols])

    report = {
        'job': 'm7_gate_features', 'engine_version': ENGINE_VERSION,
        'rows': int(n),
        'early_window': {'definition': 'offset_pct <= 0.10（播放进度前 10%）', 'pct': EARLY_PCT},
        'peak_definition': '5 秒桶内同一 text_norm >= 3 条',
        'features': {
            'early_danmaku': ['early_n', 'early_users', 'early_texts', 'early_repeat_rate',
                              'early_peak_cnt', 'early_entropy', 'early_short_ratio',
                              'early_len_mean', 'early_users_per_msg', 'early_per_min'],
            'metadata': ['title_len', 'title_has_digit', 'title_has_q', 'title_has_excl',
                         'duration', 'tname', 'pub_hour', 'pub_wday'],
        },
        'outcome_only': ['play', 'like', 'coin', 'favorite', 'share', 'reply',
                         'danmaku_reported', 'danmaku_got'],
        'forbidden_as_features': ['play', 'like', 'coin', 'favorite', 'share',
                                  'danmaku_reported', 'danmaku_got', 'reply'],
        'missing_sources': ['简介长度（无 desc 字段）', 'UP 主粉丝数（无该维表）',
                            '早期评论数（无按时间的评论明细）'],
        'early_zero_rows': int(out.filter(F.col('early_n') == 0).count()),
        'early_lt10_rows': int(out.filter(F.col('early_n') < 10).count()),
        'csv': csv_path,
    }
    with open(os.path.join(args.out_dir, 'm7_gate_features_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    spark.stop()


if __name__ == '__main__':
    main()
