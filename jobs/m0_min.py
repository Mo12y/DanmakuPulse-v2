#!/usr/bin/env python3
"""
m0_min.py — M0 最小版：raw → ods → dwd（B0.3，11 §2；完整 M0 属 B1.1）

口径对齐旧仓库 pipeline/clean.py（对拍锚点 676,931 / 672,439 / 99.3364% 的前提，逐条对应）：
  - CSV 读取：RFC4180（quote='"' escape='"'，坑：Python csv.writer 双写引号 vs Spark 默认反斜杠转义）
    + BOM 剥离（旧 clean.py read_union 同款处理）
  - ODS = cast(time_offset→double, duration→int, play→long, danmaku_reported→int, danmaku_got→int,
               status→string) + filter(time_offset IS NOT NULL)
  - DWD = ODS.filter(status='ok').dropDuplicates(['bvid','text','time_offset'])
    ⚠️ 去重键 = 旧口径 (bvid,text,time_offset)，非 08 §1.4 新键 (bvid,cid,row_id)——
       新键切换属 B1 M0 完整版决策，届时须记口径变更并重新对账
  - quality_flag = invalid_offset if (offset<0 or offset>duration) else ok（打标不删除；B0 实测 4 批全 ok）

输出（parquet，batch_date 分区，幂等分区覆盖）：
  {wh}/ods_danmaku_full / {wh}/dwd_danmaku_clean_min / {wh}/ods_batch_meta / {wh}/ods_hotlist_video

用法：
  集群：spark-submit --master spark://spark-master:7077 jobs/m0_min.py            # 路径默认 HDFS
  本地：python jobs/m0_min.py --raw-root file:data/raw-local --wh-root file:data-out/dev-wh --local
"""
import argparse
import json
import os
import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def build_spark(local=False):
    b = (SparkSession.builder.appName('m0_min')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.sources.partitionOverwriteMode', 'dynamic'))
    if local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    return b.getOrCreate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-root', default='/raw')
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config-root', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config'))
    ap.add_argument('--out-dir', default=None, help='运行报告输出目录（本地文件系统；集群默认 /opt/work/data-out）')
    ap.add_argument('--local', action='store_true', help='本地模式（file:// 路径开发调试用）')
    args = ap.parse_args()

    spark = build_spark(local=args.local)
    spark.sparkContext.setLogLevel('ERROR')
    raw, wh = args.raw_root, args.wh_root
    out_dir = args.out_dir or ('/opt/work/data-out' if not args.local
                               else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out'))
    os.makedirs(out_dir, exist_ok=True)

    # ── 1) 弹幕 CSV → ODS（cast + time_offset 非空过滤，口径=旧 clean.py）──
    df = (spark.read.option('header', True)
          .option('quote', '"').option('escape', '"')
          .csv(f'{raw}/batch_date=*/danmaku.csv'))

    # BOM：Spark 多文件读 header 以第一个文件为准，其余文件 header 行按位置跳过 → 单次改名即可
    c0 = df.columns[0]
    if c0.lstrip('\ufeff') != c0:
        df = df.withColumnRenamed(c0, c0.lstrip('\ufeff'))

    df = (df
          .withColumn('time_offset', F.col('time_offset').cast('double'))
          .withColumn('duration', F.col('duration').cast('int'))
          .withColumn('play', F.col('play').cast('long'))
          .withColumn('danmaku_reported', F.col('danmaku_reported').cast('int'))
          .withColumn('danmaku_got', F.col('danmaku_got').cast('int'))
          .withColumn('status', F.col('status').cast('string'))
          .filter(F.col('time_offset').isNotNull())
          .withColumn('batch_date',
                      F.regexp_extract(F.input_file_name(), r'batch_date=(\d{8})', 1)))

    ods_rows = df.count()
    df.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/ods_danmaku_full')

    # ── 2) ODS → DWD（status=ok + 旧去重键 + quality_flag 打标）──
    # ⚠️ 旧 clean.py 是"逐批次独立去重"（每批一次快照，各自 dropDuplicates）——
    #    全表一次去重会把跨批次重复视频（如 BV1mwY96jEom 同时在 0914/0917 两批，
    #    弹幕池快照大量重合）的行错误吃掉（实测少 2,433 行）。
    #    去重键加入 batch_date = 精确复刻批内去重语义。
    dwd = (df.filter(F.col('status') == 'ok')
           .dropDuplicates(['batch_date', 'bvid', 'text', 'time_offset'])
           .withColumn('quality_flag',
                       F.when((F.col('time_offset') < 0) | (F.col('time_offset') > F.col('duration')),
                              'invalid_offset').otherwise('ok')))
    dwd_rows = dwd.count()
    dwd.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dwd_danmaku_clean_min')

    # ── 3) ods_batch_meta：meta.json 字段 + 血缘标注（cadence/快照存留/已知问题）──
    lineage = json.load(open(os.path.join(args.config_root, 'batch_lineage.json')))
    meta = (spark.read.option('multiLine', True).json(f'{raw}/batch_date=*/meta.json')
            .withColumn('batch_date', F.regexp_extract(F.input_file_name(), r'batch_date=(\d{8})', 1))
            .select('batch_date', 'csv', 'list_fetched_at', 'count', 'depth_requested',
                    'status', 'coverage', 'note'))
    # 快照存在性（JVM FileSystem；本地模式 file:// 同样适用）
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)

    def snapshot_exists(bd):
        return bool(fs.exists(jvm.org.apache.hadoop.fs.Path(f'{raw}/batch_date={bd}/hotlist.json')))

    snap_map = {bd: snapshot_exists(bd) for bd in [r['batch_date'] for r in meta.select('batch_date').collect()]}
    snap_df = spark.createDataFrame([(bd, p) for bd, p in snap_map.items()], ['batch_date', 'snapshot_present'])
    lin_rows = [(bd, v.get('cadence', 'daily'), v.get('real_list_date'), v.get('snapshot_present'),
                 v.get('known_issues', '')) for bd, v in lineage.items()]
    lin = spark.createDataFrame(lin_rows, ['batch_date', 'cadence', 'real_list_date', 'lineage_snapshot', 'known_issues'])
    # 血缘标注与实测存在性不一致时以实测为准并告警（防血缘档案过期）
    for bd, present in snap_map.items():
        if bd in lineage and lineage[bd].get('snapshot_present') != present:
            print(f'[warn] batch={bd} 血缘档案 snapshot_present={lineage[bd].get("snapshot_present")} '
                  f'与实测 {present} 不一致，以实测为准')
    meta_full = (meta.join(lin.drop('lineage_snapshot'), 'batch_date', 'left')
                 .join(snap_df, 'batch_date', 'left'))
    meta_full.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/ods_batch_meta')

    # ── 4) ods_hotlist_video：有快照的批次解析 videos 数组，rank_proxy = 数组下标（07 §2 口径）──
    hot_parts = []
    for bd, present in sorted(snap_map.items()):
        if not present:
            continue
        j = spark.read.option('multiLine', True).json(f'{raw}/batch_date={bd}/hotlist.json')
        v = (j.selectExpr('fetched_at', 'posexplode(videos) as (rank_proxy, vv)')
             .selectExpr('fetched_at', 'rank_proxy', 'vv.*')
             .withColumn('batch_date', F.lit(bd)))
        hot_parts.append(v)
    if hot_parts:
        hot = hot_parts[0]
        for p in hot_parts[1:]:
            hot = hot.unionByName(p, allowMissingColumns=True)
        hot.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/ods_hotlist_video')
        hot_n = hot.count()
    else:
        hot_n = 0

    # ── 5) 运行报告（行数即对账证据）──
    report = {
        'job': 'm0_min', 'engine_version': 'b0-min-v1',
        'ods_rows': ods_rows, 'dwd_rows': dwd_rows, 'hotlist_rows': hot_n,
        'snapshot_map': snap_map,
        'anchors_expect': {'ods_rows': 676931, 'dwd_rows': 672439, 'retention': '0.993364'},
        'anchors_match': {'ods_rows': ods_rows == 676931, 'dwd_rows': dwd_rows == 672439},
    }
    with open(os.path.join(out_dir, 'm0_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m0_min 完成：ods={ods_rows:,} dwd={dwd_rows:,} '
          f'(锚点匹配: {report["anchors_match"]})')
    spark.stop()


if __name__ == '__main__':
    main()
