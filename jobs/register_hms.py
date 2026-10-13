#!/usr/bin/env python3
"""
register_hms.py — 把 warehouse parquet 路径注册为 Hive Metastore 外部表（幂等）
B0 的 ETL 写路径不依赖 HMS（正确性解耦）；本作业让 HMS 真正被用起来：
  外部表 + MSCK REPAIR 同步分区 → spark-sql/HiveSQL 可直接查。
⚠️ 2026-10-13：改为 DROP + CREATE（schema 刷新）——原 IF NOT EXISTS 在列变更时无法更新目录
  （M5 补 text_norm 列时曾需手动 drop）。外部表 DROP 不删数据，重跑即恢复；MSCK 重建分区。
"""
import argparse

from pyspark.sql import SparkSession

# 分区表（batch_date 分区，MSCK REPAIR）
# ⚠️ 只有 write.partitionBy('batch_date') 写出的表才放这里——平铺表误放会导致 MSCK
#    零分区 → COUNT=0（2026-10-13 实测：dws_curve_clusters/dws_hawkes_params 曾误放）
TABLES = [
    # B0（min 对拍链）
    ('ods_danmaku_full', '/warehouse/ods_danmaku_full'),
    ('dwd_danmaku_clean_min', '/warehouse/dwd_danmaku_clean_min'),
    ('ods_batch_meta', '/warehouse/ods_batch_meta'),
    ('ods_hotlist_video', '/warehouse/ods_hotlist_video'),
    ('dws_video_curve_min', '/warehouse/dws_video_curve_min'),
    ('dws_video_metrics_min', '/warehouse/dws_video_metrics_min'),
    # B1（生产链）
    ('dwd_danmaku_clean', '/warehouse/dwd_danmaku_clean'),
    ('dwd_video_metric_snapshot', '/warehouse/dwd_video_metric_snapshot'),
    ('dwd_danmaku_sent', '/warehouse/dwd_danmaku_sent'),
    ('dws_sentiment_curve_1pct', '/warehouse/dws_sentiment_curve_1pct'),
    ('dws_density_5s', '/warehouse/dws_density_5s'),
    ('dws_density_1pct', '/warehouse/dws_density_1pct'),
    ('dws_video_rhythm_features', '/warehouse/dws_video_rhythm_features'),
    ('dws_reconcile_detail', '/warehouse/dws_reconcile_detail'),
    # B2-L2（生产链）
    ('dws_rhythm_events', '/warehouse/dws_rhythm_events'),
    ('dws_sync_features', '/warehouse/dws_sync_features'),
    ('dws_segments', '/warehouse/dws_segments'),
    ('dws_evolution_feats', '/warehouse/dws_evolution_feats'),
    ('dws_onlist_track', '/warehouse/dws_onlist_track'),
    ('dws_video_daily', '/warehouse/dws_video_daily'),
]

# 非分区表（含 batch_date 作为普通列，不 MSCK）
TABLES_PLAIN = [
    ('dws_batch_audit', '/warehouse/dws_batch_audit'),
    ('dws_partition_rhythm', '/warehouse/dws_partition_rhythm'),
    ('dws_lag_calibration', '/warehouse/dws_lag_calibration'),
    ('dws_curve_clusters', '/warehouse/dws_curve_clusters'),   # 平铺小表（509 行，非分区）
    ('dws_hawkes_params', '/warehouse/dws_hawkes_params'),     # 平铺小表（231 行，非分区）
    ('dws_lifecycle', '/warehouse/dws_lifecycle'),
    ('dws_up_profile', '/warehouse/dws_up_profile'),
    ('ml_features', '/warehouse/ml_features'),   # M6 特征矩阵（平铺）
    ('ml_panel', '/warehouse/ml_panel'),         # M6 截断面板（平铺）
    ('dim_video', '/warehouse/dim_video'),
    ('dict_sentiment', '/warehouse/dict_sentiment'),
    ('dict_slang_meme', '/warehouse/dict_slang_meme'),
    ('dict_anchor', '/warehouse/dict_anchor'),
    ('dict_emoticon', '/warehouse/dict_emoticon'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--local', action='store_true', help='本地模式（跳过——HMS 仅集群有）')
    args = ap.parse_args()
    if args.local:
        print('本地模式：跳过 HMS 注册')
        return

    spark = (SparkSession.builder.appName('register_hms')
             .config('spark.sql.catalogImplementation', 'hive')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')

    for name, loc in TABLES:
        loc = loc.replace('/warehouse', args.wh_root) if args.wh_root != '/warehouse' else loc
        # 列 schema 从 parquet 数据推导，DDL 程序化生成（免手写 29 列）
        df = spark.read.parquet(loc)
        cols = ', '.join(f'`{f.name}` {f.dataType.simpleString()}'
                         for f in df.schema.fields if f.name != 'batch_date')
        spark.sql(f'DROP TABLE IF EXISTS {name}')  # schema 刷新（外部表，不动数据）
        spark.sql(f'CREATE EXTERNAL TABLE IF NOT EXISTS {name} ({cols}) '
                  f'PARTITIONED BY (batch_date STRING) STORED AS PARQUET LOCATION \'{loc}\'')
        spark.sql(f'MSCK REPAIR TABLE {name}')
        n = spark.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'✅ {name}: {n} 行（HMS 已注册）')

    for name, loc in TABLES_PLAIN:
        loc = loc.replace('/warehouse', args.wh_root) if args.wh_root != '/warehouse' else loc
        df = spark.read.parquet(loc)
        cols = ', '.join(f'`{f.name}` {f.dataType.simpleString()}' for f in df.schema.fields)
        spark.sql(f'DROP TABLE IF EXISTS {name}')  # schema 刷新（外部表，不动数据）
        spark.sql(f'CREATE EXTERNAL TABLE IF NOT EXISTS {name} ({cols}) '
                  f'STORED AS PARQUET LOCATION \'{loc}\'')
        n = spark.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'✅ {name}: {n} 行（HMS 已注册，非分区）')
    spark.stop()


if __name__ == '__main__':
    main()
