#!/usr/bin/env python3
"""
m0_daily.py — M0 日频增量装载（B0.2「次日新批次自动到达」落地；08 §0.1 日频链第一环）

输入：{raw}/batch_date=<bd>/{danmaku.csv, meta.json, hotlist.json?}（--batches 指定批次，必填）
输出（**dynamic partition overwrite**——只替换 --batches 指定批次的分区，legacy/他批零触碰）：
  ① {wh}/ods_danmaku_full   新批次弹幕（ODS 口径与 m0_min 逐字一致：cast + time_offset 非空过滤）
  ② {wh}/ods_batch_meta     新批次元数据行（cadence：lineage 有则沿用，无则 'daily'；snapshot_present 实测）
  ③ {wh}/ods_hotlist_video  新批次榜单快照（有 hotlist.json 时；rank_proxy = videos 数组下标）
  ④ {wh}/ods_video_fetch_times  视频级采集时刻（有 fetch_times.csv 时；2026-10-14 起）

职责边界（⚠️ 关键）：
  - 本作业**不写** dwd_danmaku_clean_min——对拍链产物冻结于 4 批 legacy，由
    `m0_min --batches 20260902,20260907,20260914,20260917` 维护；
  - 生产 dwd 由 m0_full（读全量 ods）重算——08 §0.1「滚动全量表每日全量重算」；
  - 对拍/复放链（etl/replay/dev-etl）经 Makefile 变量 $(LEGACY_BATCHES) 限定，天然与新批次隔离。
幂等：同批次重跑结果一致；--batches 可单批或多批（逗号分隔）。
"""
import argparse
import json
import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

ENGINE_VERSION = 'b2-v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', default='', help='逗号分隔批次（必填），如 20261012,20261013')
    ap.add_argument('--raw-root', default='/raw')
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config-root',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config'))
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    if not args.batches.strip():
        raise SystemExit('❌ --batches 必填（如 --batches 20261012,20261013）——'
                         '日频增量只针对指定批次，绝不可全量重建（会并入对拍链）')
    batches = [b.strip() for b in args.batches.split(',') if b.strip()]
    raw, wh = args.raw_root, args.wh_root

    b = (SparkSession.builder.appName('m0_daily')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.sources.partitionOverwriteMode', 'dynamic'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # ── 1) 弹幕 CSV → ODS（口径与 m0_min 逐字一致）──
    csv_paths = [f'{raw}/batch_date={bd}/danmaku.csv' for bd in batches]
    df = (spark.read.option('header', True)
          .option('quote', '"').option('escape', '"')
          .csv(csv_paths))
    c0 = df.columns[0]
    if c0.lstrip('\ufeff') != c0:
        df = df.withColumnRenamed(c0, c0.lstrip('\ufeff'))
    ods = (df.withColumn('time_offset', F.col('time_offset').cast('double'))
           .withColumn('duration', F.col('duration').cast('int'))
           .withColumn('play', F.col('play').cast('long'))
           .withColumn('danmaku_reported', F.col('danmaku_reported').cast('int'))
           .withColumn('danmaku_got', F.col('danmaku_got').cast('int'))
           .withColumn('status', F.col('status').cast('string'))
           .filter(F.col('time_offset').isNotNull())
           .withColumn('batch_date', F.regexp_extract(F.input_file_name(), r'batch_date=(\d{8})', 1)))
    per_batch_rows = {r['batch_date']: r['n'] for r in
                      ods.groupBy('batch_date').agg(F.count('*').alias('n')).collect()}
    ods.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/ods_danmaku_full')
    n_ods = ods.count()

    # ── 2) ods_batch_meta（meta.json + 血缘；新批次默认 cadence='daily'，自描述）──
    # ⚠️ 新旧 meta.json 字段集不同（新批缺 depth_requested/note，2026-10-13 实测）——
    #    以库中已有表 schema 为准则，缺失列补 typed null（避免 select 报缺列 / void 类型落盘失败）
    lineage_path = os.path.join(args.config_root, 'batch_lineage.json')
    lineage = json.load(open(lineage_path)) if os.path.isfile(lineage_path) else {}
    meta_paths = [f'{raw}/batch_date={bd}/meta.json' for bd in batches]
    meta = (spark.read.option('multiLine', True).json(meta_paths)
            .withColumn('batch_date', F.regexp_extract(F.input_file_name(), r'batch_date=(\d{8})', 1)))
    meta_cols = ['csv', 'list_fetched_at', 'count', 'depth_requested', 'status', 'coverage', 'note']
    try:
        old = spark.read.parquet(f'{wh}/ods_batch_meta')
        type_map = {f.name: f.dataType for f in old.schema.fields}
        for c in meta_cols:
            if c not in meta.columns:
                meta = meta.withColumn(c, F.lit(None).cast(type_map.get(c, 'string')))
    except Exception:
        for c in meta_cols:
            if c not in meta.columns:
                meta = meta.withColumn(c, F.lit(None).cast('string'))
    meta = meta.select('batch_date', *meta_cols)
    # 快照存在性实测（JVM FileSystem）
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)

    def snapshot_exists(bd):
        return bool(fs.exists(jvm.org.apache.hadoop.fs.Path(f'{raw}/batch_date={bd}/hotlist.json')))

    snap_rows = []
    for bd in batches:
        lin = lineage.get(bd, {})
        snap_rows.append((bd, lin.get('cadence', 'daily'), lin.get('real_list_date', bd),
                          snapshot_exists(bd), lin.get('known_issues', '')))
        if bd in lineage and lineage[bd].get('snapshot_present') != snap_rows[-1][3]:
            print(f'[warn] batch={bd} 血缘档案 snapshot_present={lineage[bd].get("snapshot_present")} '
                  f'与实测 {snap_rows[-1][3]} 不一致，以实测为准')
    snap_df = spark.createDataFrame(snap_rows,
                                    ['batch_date', 'cadence', 'real_list_date',
                                     'snapshot_present', 'known_issues'])
    # 只写指定批次行 + dynamic 分区覆盖 ⇒ 旧批次分区零触碰（无需 union 旧行）
    meta_full = meta.join(snap_df, 'batch_date', 'left')
    meta_full.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/ods_batch_meta')

    # ── 3) ods_hotlist_video（有快照的批次；rank_proxy = 数组下标）──
    hot_parts = []
    for bd, cad, _, present, _ in snap_rows:
        if not present:
            print(f'[warn] batch={bd} 无 hotlist.json 快照 → 跳过榜单装载')
            continue
        j = spark.read.option('multiLine', True).json(f'{raw}/batch_date={bd}/hotlist.json')
        v = (j.selectExpr('fetched_at', 'posexplode(videos) as (rank_proxy, vv)')
             .selectExpr('fetched_at', 'rank_proxy', 'vv.*')
             .withColumn('batch_date', F.lit(bd).cast('int')))
        hot_parts.append(v)
    hot_n = 0
    if hot_parts:
        hot = hot_parts[0]
        for p in hot_parts[1:]:
            hot = hot.unionByName(p, allowMissingColumns=True)
        hot.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/ods_hotlist_video')
        hot_n = hot.count()

    # ── 4) ods_video_fetch_times（视频级采集时刻 sidecar；无 sidecar 的批次零触碰）──
    ft_parts, ft_batches = [], []
    for bd in batches:
        p = f'{raw}/batch_date={bd}/fetch_times.csv'
        if fs.exists(jvm.org.apache.hadoop.fs.Path(p)):
            t = spark.read.option('header', True).csv(p)
            t = t.toDF(*[c.lstrip('\ufeff') for c in t.columns])
            t = (t.select(F.col('bvid'), F.col('fetched_at'), F.col('src'))
                 .withColumn('batch_date', F.lit(bd)))
            ft_parts.append(t)
            ft_batches.append(bd)
    ft_n = 0
    if ft_parts:
        ft = ft_parts[0]
        for t in ft_parts[1:]:
            ft = ft.unionByName(t)
        ft.write.mode('overwrite').partitionBy('batch_date').parquet(
            f'{wh}/ods_video_fetch_times')
        ft_n = ft.count()
    else:
        print('[ftimes] 指定批次均无 fetch_times.csv → ods_video_fetch_times 未触碰')

    report = {
        'job': 'm0_daily', 'engine_version': ENGINE_VERSION,
        'batches': batches, 'ods_rows_by_batch': per_batch_rows, 'ods_rows_total': n_ods,
        'snapshot_map': {r[0]: r[3] for r in snap_rows},
        'cadence_map': {r[0]: r[1] for r in snap_rows},
        'hotlist_rows': hot_n,
        'fetch_times_rows': ft_n, 'fetch_times_batches': ft_batches,
        'note': 'dynamic partition overwrite——仅替换指定批次分区；不写 dwd_min（对拍链冻结 4 批）',
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm0_daily_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m0_daily：{len(batches)} 批装载完成（ods {n_ods} 行；hotlist {hot_n} 行）')
    spark.stop()


if __name__ == '__main__':
    main()
