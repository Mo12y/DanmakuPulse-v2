#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp57_export.py — 八路线统一数据盘点与样本导出

输出（file:///opt/work/data-out/exp57/）：
  quality.json        字段质量盘点（分批次：覆盖/缺失/分布/去重键/比值）
  text_sample.csv     逐条弹幕样本（user_hash 覆盖≥90% 的批次，200 支视频；含文本/表达/时间/用户）
  video_meta.csv      视频粒度元数据（up_mid/tname/duration/got/reported/桶数/跨批次数/用户覆盖）

口径：DWD 主口径（is_spam 只作标记不作删除；刷量分析单独看）；不填充、不伪造。
"""
import json

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

OUT = 'file:///opt/work/data-out/exp57'
spark = (SparkSession.builder.appName('exp57_export')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

d = spark.read.parquet('/warehouse/dwd_danmaku_clean')
N = d.count()

# ---------- 1) 字段质量（总体 + 分批次） ----------
agg = d.agg(
    F.count('*').alias('rows'),
    F.countDistinct('bvid').alias('videos'),
    F.sum(F.when(F.col('user_hash').isNotNull(), 1).otherwise(0)).alias('user_hash_nonnull'),
    F.sum(F.when(F.col('text_norm') != F.col('text'), 1).otherwise(0)).alias('text_norm_differs'),
    F.sum(F.when(F.col('is_spam'), 1).otherwise(0)).alias('is_spam_true'),
    F.sum(F.when(F.col('is_subtitle_pool') | (F.col('pool') == 1), 1).otherwise(0)).alias('subtitle_pool'),
    F.sum(F.when(F.col('is_colored'), 1).otherwise(0)).alias('colored'),
    F.sum(F.when(F.col('is_special_mode'), 1).otherwise(0)).alias('special_mode'),
    F.sum(F.when(F.col('is_ultra_short'), 1).otherwise(0)).alias('ultra_short'),
    F.sum(F.when(F.col('is_ad'), 1).otherwise(0)).alias('ad'),
    F.sum(F.when(F.col('is_meaningless'), 1).otherwise(0)).alias('meaningless'),
    F.sum(F.when(F.col('time_offset').isNull(), 1).otherwise(0)).alias('offset_null'),
    F.sum(F.when((F.col('time_offset') < 0) | (F.col('time_offset') > F.col('duration')), 1).otherwise(0)).alias('offset_invalid'),
    F.sum(F.when(F.col('send_time').isNull(), 1).otherwise(0)).alias('send_time_null'),
    F.sum(F.when(F.col('lag_sec').isNull(), 1).otherwise(0)).alias('lag_null'),
    F.sum(F.when(F.col('status') != 'ok', 1).otherwise(0)).alias('status_not_ok'),
    F.sum(F.when(F.col('part_count').cast('int') >= 2, 1).otherwise(0)).alias('multipart_rows'),
).collect()[0].asDict()

uniq = d.select(F.countDistinct('bvid', 'cid', 'row_id').alias('u')).collect()[0]['u']
mode_dist = {str(r['mode']): r['count'] for r in d.groupBy('mode').count().collect()}
fs_dist = {str(r['font_size']): r['count'] for r in d.groupBy('font_size').count().orderBy(F.desc('count')).limit(6).collect()}

per_batch = []
for r in (d.groupBy('batch_date')
          .agg(F.count('*').alias('rows'), F.countDistinct('bvid').alias('videos'),
               F.round(F.avg(F.when(F.col('user_hash').isNotNull(), 1).otherwise(0)), 4).alias('user_cov'),
               F.round(F.avg(F.when(F.col('is_spam'), 1).otherwise(0)), 5).alias('spam_rate'),
               F.round(F.avg(F.when(F.col('text_norm') != F.col('text'), 1).otherwise(0)), 4).alias('norm_diff_rate'),
               F.round(F.avg(F.col('danmaku_got') / F.greatest(F.col('danmaku_reported'), F.lit(1))), 4).alias('got_over_reported'))
          .orderBy('batch_date').collect()):
    per_batch.append({k: (float(v) if isinstance(v, float) else v) for k, v in r.asDict().items() if k != 'batch_date'} | {'batch_date': r['batch_date']})

quality = dict(rows=N, videos=agg['videos'], distinct_row_keys=uniq,
               key_unique=(uniq == N),
               shares={k: round(agg[k] / N, 6) for k in agg if k not in ('rows', 'videos')},
               mode_top=mode_dist, font_size_top=fs_dist, per_batch=per_batch)
import os
os.makedirs('/opt/work/data-out/exp57', exist_ok=True)
with open('/opt/work/data-out/exp57/quality.json', 'w', encoding='utf-8') as f:
    json.dump(quality, f, ensure_ascii=False, indent=1)

# ---------- 2) 视频粒度：用户覆盖 → 选 200 支样本 ----------
vm = (d.groupBy('bvid', 'batch_date')
      .agg(F.first('up_mid').alias('up_mid'), F.first('tname').alias('tname'),
           F.first('duration').alias('duration'), F.first('danmaku_got').alias('got'),
           F.first('danmaku_reported').alias('reported'), F.count('*').alias('rows'),
           F.round(F.avg(F.when(F.col('user_hash').isNotNull(), 1).otherwise(0)), 4).alias('user_cov'),
           F.max('time_offset').alias('max_offset'),
           F.round(F.avg(F.when(F.col('is_spam'), 1).otherwise(0)), 4).alias('spam_rate'))
      .withColumn('n_buckets', F.greatest(F.floor(F.col('duration') / 60).cast('long'), F.lit(1)))
      .withColumn('coverage', F.round(F.col('rows') / F.greatest(F.col('got'), F.lit(1)), 4)))
vm.write.mode('overwrite').option('header', True).csv(f'{OUT}/video_meta_csv')

pick = (vm.filter((F.col('user_cov') >= 0.9) & (F.col('rows') >= 300) & (F.col('duration') >= 300))
        .withColumn('rk', F.row_number().over(Window.orderBy('bvid')))
        .filter(F.col('rk') <= 200).select('bvid', 'batch_date'))
n_pick = pick.count()

s = (d.join(pick, ['bvid', 'batch_date'], 'inner')
     .filter(F.col('time_offset').isNotNull())
     .filter((F.col('time_offset') >= 0) & (F.col('time_offset') <= F.col('duration')))
     .withColumn('bucket60', F.floor(F.col('time_offset') / 60).cast('long'))
     .withColumn('txt', F.regexp_replace(F.col('text'), r'[\r\n\t]+', ' '))
     .withColumn('tnorm', F.regexp_replace(F.coalesce(F.col('text_norm'), F.col('text')), r'[\r\n\t]+', ' '))
     .select('bvid', 'batch_date', 'bucket60', 'txt', 'tnorm', 'user_hash', 'text_len',
             'mode', 'color', 'font_size', 'attr_weight', F.col('lag_sec').alias('lag_s'),
             F.col('offset_pct').alias('pct')))
n_s = s.count()
(s.coalesce(1).write.mode('overwrite').option('header', True)
 .option('quote', '"').option('escape', '"').csv(f'{OUT}/text_sample_csv'))

print(json.dumps({'rows': N, 'videos': agg['videos'], 'key_unique': uniq == N,
                  'user_cov': round(agg['user_hash_nonnull'] / N, 4),
                  'sample_videos': n_pick, 'sample_rows': n_s}, ensure_ascii=False))
print('✅ exp57_export 完成 →', OUT)
spark.stop()
