#!/usr/bin/env python3
"""
make_amplified.py — B5.1 合成放大数据集（**全标注「合成放大（重放+扰动）」**）

用途：性能实验的规模轴（1×/4×/8×）。基准 = dwd_danmaku_clean（真实 676k 行）；
      k=0 复制原样；k≥1 重放+扰动（虚拟 bvid 后缀 / send_time 平移 / 位置加性扰动 / 权重微扰）。
输出：{wh}/perf_amp_{1,4,8}x（partitionBy batch_date；与 dwd 同构，供 perf_harness 专用）
⚠️ 该数据**仅用于性能测量**，不得进入任何生产表；一切下游报告须带「合成放大（重放+扰动）」标注。
"""
import argparse

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

COLS = ['bvid', 'batch_date', 'send_time', 'send_date', 'time_offset', 'offset_pct',
        'duration', 'attr_weight', 'text', 'text_len', 'is_colored', 'is_special_mode', 'pubdate']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--mult', type=int, required=True, choices=[1, 4, 8])
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = SparkSession.builder.appName(f'make_amp_{args.mult}x') \
        .config('spark.sql.shuffle.partitions', '8')
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    base = spark.read.parquet(f'{wh}/dwd_danmaku_clean').select(*COLS)
    parts = [base]
    for k in range(1, args.mult):
        parts.append(
            base.select(
                F.concat(F.col('bvid'), F.lit(f'#a{k}')).alias('bvid'),
                'batch_date',
                (F.col('send_time').cast('long') + k * 3600).alias('send_time'),
                'send_date',
                F.least(F.col('time_offset').cast('double') + k * 1.7,
                        F.col('duration').cast('double')).alias('time_offset'),
                F.least(F.col('offset_pct').cast('double') + 0.001 * k, F.lit(0.999)).alias('offset_pct'),
                'duration',
                (F.col('attr_weight').cast('double') * (1 + 0.01 * k)).alias('attr_weight'),
                'text', 'text_len', 'is_colored', 'is_special_mode', 'pubdate'))
    out = parts[0]
    for p in parts[1:]:
        out = out.unionByName(p)
    out.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/perf_amp_{args.mult}x')
    n = out.count()
    nv = out.select('bvid').distinct().count()
    print(f'✅ perf_amp_{args.mult}x：{n} 行 / {nv} 虚拟 bvid（合成放大（重放+扰动））')
    spark.stop()


if __name__ == '__main__':
    main()
