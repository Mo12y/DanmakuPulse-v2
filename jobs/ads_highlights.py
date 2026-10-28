#!/usr/bin/env python3
"""
ads_highlights.py — 精彩时刻 ADS 导出（PLAN_HIGHLIGHTS §2.0）：4 张 DWS parquet → PostgreSQL ads 表

  dws_resonance_peaks        → ads_resonance_peaks        （复读聚集 → 共振峰/段）
  dws_critique_segments      → ads_critique_segments      （批判层段，宁缺毋滥）
  dws_video_feedback_profile → ads_video_feedback_profile （从众/中性/批判画像 + 分区对标）
  dict_critique              → ads_dict_critique          （质疑词表）

口径：
  - batch_date：Hive 里 string → PG integer（与 ads_video_highlights 一致）；
  - 数组列 top_danmaku / top_counts 原样写出，PG 侧为 text[] / integer[]；
  - engine_version 等派生列不在本作业里重算，一律沿用 DWS 产物；
  - 写库方式与 ads_build.py 相同：Spark JDBC overwrite 全量重建（可重复跑）。
连接参数同 ads_build.py：PG_USER / PG_PASSWORD 环境变量 + postgresql-42.7.4.jar（Makefile 传入）。
"""
import argparse
import json
import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

ENGINE_VERSION = 'hl-v1'

PEAK_COLS = ['bvid', 'batch_date', 'peak_id', 'start_sec', 'end_sec', 'peak_sec',
             'peak_score', 'n_repeat', 'n_rep_texts', 'top_danmaku', 'top_counts',
             'pos_ratio', 'neg_ratio', 'neu_ratio',
             'in_opening', 'pctl', 'partition_n', 'engine_version']
CRIT_COLS = ['bvid', 'batch_date', 'seg_id', 'start_sec', 'end_sec', 'critique_score',
             'n_critique', 'top_danmaku', 'top_counts', 'neg_ratio',
             'method', 'ml_score',          # 2026-10-20：dict / ml 并存（G2 弱监督泛化）
             'engine_version']
FEEDBACK_COLS = ['bvid', 'batch_date', 'n_danmaku', 'n_herd', 'n_neutral', 'n_critique',
                 'ratio_herd', 'ratio_neutral', 'ratio_critique', 'n_peaks',
                 'resonance_total', 'max_peak_score', 'duration', 'tname',
                 'partition_n', 'resonance_pctl', 'resonance_z', 'engine_version']
DICT_COLS = ['word', 'is_critique', 'weight', 'source', 'confirmed', 'engine_version']

# (DWS 目录名, PG 表名, 导出列)
TABLES = [
    ('dws_resonance_peaks', 'ads_resonance_peaks', PEAK_COLS),
    ('dws_critique_segments', 'ads_critique_segments', CRIT_COLS),
    ('dws_video_feedback_profile', 'ads_video_feedback_profile', FEEDBACK_COLS),
    ('dict_critique', 'ads_dict_critique', DICT_COLS),
]


def _pick(df, cols, name):
    """按契约列导出；缺列直接报错（宁可在导出层红，也不静默少列）。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit('DWS %s 缺列：%s\n现有列：%s' % (
            name, ', '.join(missing), ', '.join(df.columns)))
    out = df.select(*cols)
    if 'batch_date' in cols:
        out = out.withColumn('batch_date', F.col('batch_date').cast('int'))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--pg-host', default='postgres')
    ap.add_argument('--pg-db', default='ads')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true', help='本地模式（跳过 PG 写入，只做构建演练）')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('ads_highlights')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    url = f'jdbc:postgresql://{args.pg_host}:5432/{args.pg_db}'
    props = {'user': os.environ.get('PG_USER', 'danmaku'),
             'password': os.environ.get('PG_PASSWORD', ''),
             'driver': 'org.postgresql.Driver'}
    written = {}

    def write_pg(df, name, order=None):
        if args.local:
            written[name] = df.count()
            print(f'  [local] {name}: {written[name]} 行（跳过 PG）')
            return
        d = df.coalesce(2)
        d.write.jdbc(url, name, mode='overwrite', properties=props)
        written[name] = df.count()
        print(f'  ✅ {name}: {written[name]} 行 → PG')

    # ── 读 DWS（4 张新表）──
    frames = {}
    for src, dst, cols in TABLES:
        path = f'{wh}/{src}'
        print(f'  读 {path}')
        frames[dst] = _pick(spark.read.parquet(path), cols, src)

    # ── 写 PG（先写小表，再写画像/明细；overwrite 幂等）──
    write_pg(frames['ads_dict_critique'], 'ads_dict_critique')
    write_pg(frames['ads_video_feedback_profile'], 'ads_video_feedback_profile')
    write_pg(frames['ads_resonance_peaks'], 'ads_resonance_peaks')
    write_pg(frames['ads_critique_segments'], 'ads_critique_segments')

    report = {
        'job': 'ads_highlights',
        'engine_version': ENGINE_VERSION,
        'tables': written,
        'pg_target': f'{args.pg_host}/{args.pg_db}',
        'source': f'{wh}/dws_* + dict_critique',
        'note': 'batch_date 已 cast int；数组列保持 PG 数组类型；overwrite 全量重建',
    }
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, 'ads_highlights_report.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ ads_highlights：{len(written)} 张表 → PG')
    spark.stop()


if __name__ == '__main__':
    main()
