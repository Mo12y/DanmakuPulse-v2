#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp56_export.py — 四条新路线（相似检索/峰生命周期/条件差异/表示学习）的分析宽表导出

输出（file:///opt/work/data-out/exp56/）：
  units.csv     每 bvid 一行（去重：取 n_danmaku 最大批次）：tname/duration/got/桶数/覆盖/资格/热榜位次
  curves1pct.csv 稀疏 100 点归一曲线（bvid, bucket1pct, cnt）
  feats.csv      dws_video_rhythm_features 的节奏特征（bvid + 数值列）

口径：主口径 is_spam=false（dws_* 已是）；bvid 去重取最完整快照；不填零、只导稀疏（缺失=无行）。
"""
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

OUT = 'file:///opt/work/data-out/exp56'
spark = (SparkSession.builder.appName('exp56_export')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
wh = '/warehouse'

meta = spark.read.parquet(f'{wh}/dws_rhythm_60s_meta')
w = Window.partitionBy('bvid').orderBy(F.desc('n_danmaku'))
u = (meta.withColumn('rk', F.row_number().over(w)).filter(F.col('rk') == 1).drop('rk')
     .select('bvid', 'batch_date', 'tname', 'duration', 'danmaku_got', 'n_buckets_expected',
             'n_buckets_nonzero', 'coverage_ratio', 'fetch_available', 'c1_eligible',
             'n_interior_holes', 'n_post_snapshot'))

hot = (spark.read.parquet(f'{wh}/ods_hotlist_video')
       .select('bvid', 'batch_date', F.col('rank_proxy').cast('int').alias('rank_proxy'),
               F.col('play').cast('long').alias('play')))
u = u.join(hot, ['bvid', 'batch_date'], 'left')

u.coalesce(1).write.mode('overwrite').option('header', True).csv(f'{OUT}/units_csv')

feats = spark.read.parquet(f'{wh}/dws_video_rhythm_features')
wf = Window.partitionBy('bvid').orderBy(F.desc('n_danmaku'))
feats = (feats.withColumn('rk', F.row_number().over(wf)).filter(F.col('rk') == 1).drop('rk', 'batch_date', 'engine_version'))
feats.coalesce(1).write.mode('overwrite').option('header', True).csv(f'{OUT}/feats_csv')

d1 = spark.read.parquet(f'{wh}/dws_density_1pct')
keep = u.select('bvid', 'batch_date')
cur = (d1.join(keep, ['bvid', 'batch_date'], 'inner')
       .select('bvid', 'bucket1pct', 'cnt'))
cur.coalesce(1).write.mode('overwrite').option('header', True).csv(f'{OUT}/curves1pct_csv')

n_u, n_f, n_c = u.count(), feats.count(), cur.count()
print(f'✅ exp56_export：units={n_u} feats={n_f} curves={n_c} → {OUT}')
spark.stop()
