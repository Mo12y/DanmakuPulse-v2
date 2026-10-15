# -*- coding: utf-8 -*-
'''Step 1：节奏签名表（28 号 §9 Step1）-> data-out/rhythm_signature.csv
三组特征分开，供 T1 消融：SHAPE（进度归一化，无绝对强度）/ INTENSITY（绝对量）/ SENT（情绪通道）
'''
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.appName('sig').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
WH = '/warehouse'
fp = spark.read.parquet(WH + '/dws_video_rhythm_features').select(
    'bvid','batch_date','duration','n_danmaku','dm_per_min','wdm_per_min','opening_5pct','ending_5pct',
    'mid_ratio','rhythm_entropy','burst_cv','w_burst_cv','early_lag_ratio','avg_text_len',
    'peak_cnt','peak_max_z','peak_gap_cv')
# 主峰位置（进度轴 0-100）
d1 = spark.read.parquet(WH + '/dws_density_1pct').select('bvid','batch_date','bucket1pct','wcnt')
mx = (d1.groupBy('bvid','batch_date').agg(F.max(F.struct('wcnt','bucket1pct')).alias('top'))
        .select('bvid','batch_date', F.col('top.bucket1pct').alias('main_peak_pct')))
# 情绪通道：前/中/后段
sc = spark.read.parquet(WH + '/dws_sentiment_curve_1pct').select('bvid','batch_date','bucket','sent_avg')
seg = lambda a, b: F.when((F.col('bucket') >= a) & (F.col('bucket') < b), F.col('sent_avg'))
sx = (sc.groupBy('bvid','batch_date').agg(
        F.avg(seg(0, 34)).alias('sent_early'),
        F.avg(seg(34, 67)).alias('sent_mid'),
        F.avg(seg(67, 101)).alias('sent_late'),
        F.count('*').alias('sent_n')).filter(F.col('sent_n') > 10).drop('sent_n'))
dim = spark.read.parquet(WH + '/dim_video').select('bvid','tname')
out = fp.join(mx, ['bvid','batch_date'], 'left').join(sx, ['bvid','batch_date'], 'left').join(dim, 'bvid', 'left')
P = '/opt/work/data-out/rhythm_signature'
out.write.mode('overwrite').option('header', True).csv(P)
n = out.count()
print('签名表行数', n)
print('列:', out.columns)
# 缺失统计
nn = out.select([F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in out.columns]).collect()[0].asDict()
print('各列缺失:', {k: v for k, v in nn.items() if v})
print('tname 缺失:', nn.get('tname'))
spark.stop()