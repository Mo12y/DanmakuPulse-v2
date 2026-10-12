#!/usr/bin/env python3
"""
register_hms.py — 把 warehouse parquet 路径注册为 Hive Metastore 外部表（幂等）
B0 的 ETL 写路径不依赖 HMS（正确性解耦）；本作业让 HMS 真正被用起来：
  外部表 + MSCK REPAIR 同步分区 → spark-sql/HiveSQL 可直接查。
"""
import argparse

from pyspark.sql import SparkSession

TABLES = [
    ('ods_danmaku_full', '/warehouse/ods_danmaku_full'),
    ('dwd_danmaku_clean_min', '/warehouse/dwd_danmaku_clean_min'),
    ('ods_batch_meta', '/warehouse/ods_batch_meta'),
    ('ods_hotlist_video', '/warehouse/ods_hotlist_video'),
    ('dws_video_curve_min', '/warehouse/dws_video_curve_min'),
    ('dws_video_metrics_min', '/warehouse/dws_video_metrics_min'),
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
        spark.sql(f'CREATE EXTERNAL TABLE IF NOT EXISTS {name} ({cols}) '
                  f'PARTITIONED BY (batch_date STRING) STORED AS PARQUET LOCATION \'{loc}\'')
        spark.sql(f'MSCK REPAIR TABLE {name}')
        n = spark.sql(f'SELECT COUNT(*) FROM {name}').collect()[0][0]
        print(f'✅ {name}: {n} 行（HMS 已注册）')
    spark.stop()


if __name__ == '__main__':
    main()
