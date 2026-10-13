#!/usr/bin/env python3
"""
probe_partition_guard.py — 分区隔离守卫（日频增量装载的验收工具）

对指定表每个 batch_date 分区计算内容指纹（行数 + sum/bit_xor(xxhash64(全列))）输出 JSON；
--compare-with 对比两次快照：列出 变化/新增/消失 分区；--expect-unchanged 指定「必须未变」的
分区——若变化即退出码 1。
用途：m0_daily / `m0_min --batches` 增量后，证明「只动目标批次、他批零触碰」（2026-10-13 建）。

⚠️ 指纹为守卫用途（xxhash64 聚合，非密码学强度）；不做行级 md5（676k 行 × 30 列代价高）。
用法（容器内）：
  spark-submit /tmp/probe_partition_guard.py --tables ods_danmaku_full,ods_batch_meta \
      --out /opt/work/data-out/guard_before.json
  # …增量动作…
  spark-submit /tmp/probe_partition_guard.py --tables ods_danmaku_full,ods_batch_meta \
      --out /opt/work/data-out/guard_after.json --compare-with .../guard_before.json \
      --expect-unchanged 20260902,20260907,20260914
"""
import argparse
import json
import os
import sys

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def fp_table(spark, wh, name):
    df = spark.read.parquet(f'{wh}/{name}')
    cols = [c for c in df.columns if c != 'batch_date']
    h = df.withColumn('_h', F.xxhash64(*[F.col(c) for c in cols]))
    agg = (h.groupBy('batch_date')
           .agg(F.count('*').alias('n'),
                F.sum(F.col('_h')).alias('hsum'),
                F.expr('bit_xor(_h)').alias('hxor')))
    return {str(r['batch_date']): {'n': int(r['n']), 'hsum': str(r['hsum']), 'hxor': str(r['hxor'])}
            for r in agg.collect()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--tables', required=True, help='逗号分隔表名')
    ap.add_argument('--out', required=True)
    ap.add_argument('--compare-with', default=None)
    ap.add_argument('--expect-unchanged', default='', help='逗号分隔 batch_date（必须未变）')
    ap.add_argument('--assert-no-changes', action='store_true',
                    help='日频装载断言：任何既有分区若变化/消失即失败（只允许新增分区）')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()

    b = (SparkSession.builder.appName('partition_guard')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    tables = [t.strip() for t in args.tables.split(',') if t.strip()]
    snapshot = {t: fp_table(spark, args.wh_root, t) for t in tables}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f'✅ 指纹快照 → {args.out}')
    for t, parts in snapshot.items():
        print(f'  {t}: {len(parts)} 分区（{sorted(parts)}）')

    if args.compare_with:
        before = json.load(open(args.compare_with, encoding='utf-8'))
        expect = {e.strip() for e in args.expect_unchanged.split(',') if e.strip()}
        problems = []
        print('\n== 对比 ==')
        for t in snapshot:
            bpart = before.get(t, {})
            apart = snapshot[t]
            changed = sorted([k for k in bpart if k in apart and bpart[k] != apart[k]])
            added = sorted([k for k in apart if k not in bpart])
            removed = sorted([k for k in bpart if k not in apart])
            print(f'  [{t}] 变化={changed or "无"} 新增={added or "无"} 消失={removed or "无"}')
            for k in changed:
                if k in expect or args.assert_no_changes:
                    problems.append(f'{t} 分区 {k} 指纹变化（{"assert-no-changes" if args.assert_no_changes else "expect 清单"}）：'
                                    f'{bpart[k]} → {apart[k]}')
            for k in removed:
                if k in expect or args.assert_no_changes:
                    problems.append(f'{t} 分区 {k} 已消失（{"assert-no-changes" if args.assert_no_changes else "expect 清单"}）')
        if problems:
            print('\n❌ 守卫失败：')
            for p in problems:
                print('  -', p)
            spark.stop()
            sys.exit(1)
        print('\n✅ 守卫通过：所有 expect-unchanged 分区指纹一致、无消失')
    spark.stop()


if __name__ == '__main__':
    main()
