#!/usr/bin/env python3
"""
ads_min.py — ADS 最小层：dws → PostgreSQL ads 表 + 规范导出（幂等全量重建）

输出：
  PG: ads_video_metrics / ads_video_curve / ads_batch_meta（JDBC overwrite 全量重建——
      B0 仅 4 批 ~509 视频，全量重建即幂等；B2 起按 07 §3 批次流改造）
  本地: {out_dir}/canonical/metrics.csv + curve.csv（全局排序，重跑一致性比对用，
      比对用容差规则见 tools/compare_canonical.py——不做 md5，浮点位级不确定）
"""
import argparse
import csv
import os
import shutil

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--pg-host', default='postgres')
    ap.add_argument('--pg-db', default='ads')
    ap.add_argument('--local', action='store_true', help='本地模式（跳过 PG 写入，只出规范导出）')
    args = ap.parse_args()
    wh = args.wh_root
    out_dir = args.out_dir or ('/opt/work/data-out' if not args.local
                               else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out'))
    os.makedirs(os.path.join(out_dir, 'canonical'), exist_ok=True)

    b = (SparkSession.builder.appName('ads_min')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    m = spark.read.parquet(f'{wh}/dws_video_metrics_min')
    cv = spark.read.parquet(f'{wh}/dws_video_curve_min')
    meta = spark.read.parquet(f'{wh}/ods_batch_meta')

    curve = cv.select('batch_date', 'bvid', F.posexplode('curve').alias('bucket', 'c'))

    # 批次统计（行数/弹幕量 = 对账证据链一环）
    stats = (m.groupBy('batch_date').agg(F.count('*').alias('n_videos'),
                                         F.sum('danmaku_count').alias('n_danmaku')))
    bmeta = (meta.join(stats, 'batch_date', 'left')
             .select('batch_date', 'cadence', 'real_list_date', 'snapshot_present',
                     'status', 'coverage', 'list_fetched_at', 'n_videos', 'n_danmaku',
                     'known_issues'))

    # ── 1) PG 全量重建（JDBC 在 executor 执行 → worker 也挂了 pg jar，见 compose）──
    if not args.local:
        url = f'jdbc:postgresql://{args.pg_host}:5432/{args.pg_db}'
        props = {'user': os.environ.get('PG_USER', 'danmaku'),
                 'password': os.environ.get('PG_PASSWORD', ''),
                 'driver': 'org.postgresql.Driver'}
        (m.orderBy('batch_date', 'bvid').coalesce(1)
         .write.jdbc(url, 'ads_video_metrics', mode='overwrite', properties=props))
        (curve.orderBy('batch_date', 'bvid', 'bucket').coalesce(1)
         .write.jdbc(url, 'ads_video_curve', mode='overwrite', properties=props))
        (bmeta.orderBy('batch_date').coalesce(1)
         .write.jdbc(url, 'ads_batch_meta', mode='overwrite', properties=props))
        print('✅ PG ads 三表全量重建完成')

    # ── 2) 规范导出（重跑一致性比对基准）──
    # ⚠️ 必须 collect 到 driver 后用 Python 写文件，不能 df.write（B0 实测坑，2026-10-12）：
    #    集群模式 Spark 的默认 fs 是 HDFS（core-site.xml: fs.defaultFS=hdfs://namenode:9000），
    #    df.write 会把无 scheme 路径解析到 hdfs://namenode:9000/opt/work/data-out/...，
    #    宿主机 data-out/ 只剩 makedirs 建的空目录 → make replay 的 mv/比对报 FileNotFoundError。
    #    executor 写 file:// 也不可行（文件散落 worker 容器）；driver 所在 spark-master 容器
    #    挂载了 ../data-out:/opt/work/data-out，Python open() 写进程本地 FS = 宿主机 data-out/。
    #    数据量小（metrics 509 行 / curve 51,409 行），collect 到 driver 无压力。
    canon_dir = os.path.join(out_dir, 'canonical')
    os.makedirs(canon_dir, exist_ok=True)

    def export_local(df, name):
        path = os.path.join(canon_dir, name)
        if os.path.isdir(path):   # 清理早期 df.write 产出的 part 目录残留
            shutil.rmtree(path, ignore_errors=True)
        rows = df.collect()
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(df.columns)
            for r in rows:
                w.writerow(['' if v is None else v for v in r])
        print(f'  → {path}: {len(rows)} 行')

    export_local(m.orderBy('batch_date', 'bvid'), 'metrics.csv')
    export_local(curve.orderBy('batch_date', 'bvid', 'bucket'), 'curve.csv')
    print(f'✅ 规范导出完成：{canon_dir}/（driver 本地写；比对见 tools/compare_canonical.py）')
    spark.stop()


if __name__ == '__main__':
    main()
