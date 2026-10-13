#!/usr/bin/env python3
"""
perf_harness.py — B5.1 性能测量台（合成放大（重放+扰动）专用；不写任何生产表）

模式（均以 m2_* 真实作业的同构逻辑复刻，输入为 perf_amp_Nx 或真实 dwd）：
  density --amp N : m2_l1_density 同构（5s + 1pct 聚合 → 写 perf_out）→ 墙钟 + 分区倾斜比
  events  --amp N : m2_l2_events 同构（5s wcnt 序列 → RDD 四检测器）→ 墙钟
  hawkes          : 真实事件流：串行 vs RDD 并行（231 组 L-BFGS-B）→ 并行加速比
  skew            : 真实 dwd：普通 groupBy vs 加盐两阶段聚合 → 墙钟 + 倾斜比对照
并行度由 spark-submit --master local[P] 决定（P=1/2；VM 单 worker×2 核，4 不可得——记录）。
输出：追加 /opt/work/data-out/perf_results.jsonl
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

TZ8 = timezone(timedelta(hours=8))
DAY = 86400.0
OUT = '/opt/work/data-out/perf_results.jsonl'


def record(rec):
    rec['ts'] = datetime.now(TZ8).isoformat(timespec='seconds')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print('REC', json.dumps(rec, ensure_ascii=False), flush=True)


def part_stats(df):
    sizes = df.rdd.mapPartitionsWithIndex(
        lambda i, it: [(i, sum(1 for _ in it))]).collect()
    s = np.array([x[1] for x in sizes], dtype=float)
    nz = s[s > 0]
    if nz.size == 0:
        return {'parts': 0, 'max': 0, 'med': 0.0, 'skew': 0.0}
    return {'parts': int(len(s)), 'max': int(nz.max()), 'med': float(np.median(nz)),
            'skew': float(nz.max() / max(np.median(nz), 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=['density', 'events', 'hawkes', 'skew'])
    ap.add_argument('--amp', type=int, default=1)
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config', default='/opt/work/config/params.yaml')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName(f'perf_{args.mode}')
         .config('spark.sql.shuffle.partitions', '8'))
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    master = spark.sparkContext.master
    par = master.split('[')[-1].rstrip(']') if 'local[' in master else master[-1]

    import yaml
    params = yaml.safe_load(open(args.config, encoding='utf-8'))

    if args.mode == 'density':
        d = (spark.read.parquet(f'{wh}/perf_amp_{args.amp}x')
             .select('bvid', 'batch_date', 'time_offset', 'offset_pct', 'attr_weight'))
        agg5 = (d.withColumn('b', (F.col('time_offset').cast('double') / 5).cast('int'))
                .groupBy('bvid', 'batch_date', 'b')
                .agg(F.count('*').alias('cnt'), F.sum('attr_weight').alias('wcnt')))
        agg1 = (d.withColumn('b', F.least(F.floor(F.col('offset_pct').cast('double') * 100),
                                          F.lit(99)).cast('int'))
                .groupBy('bvid', 'batch_date', 'b')
                .agg(F.count('*').alias('cnt'), F.sum('attr_weight').alias('wcnt')))
        agg = agg5.unionByName(agg1, allowMissingColumns=True).cache()
        t0 = time.time()
        n = agg.count()
        agg.write.mode('overwrite').partitionBy('batch_date').parquet(
            f'{wh}/perf_out/density_{args.amp}x')
        t1 = time.time()
        stats = part_stats(agg)
        record({'mode': 'density', 'amp': args.amp, 'par': par, 'wall_s': round(t1 - t0, 1),
                'n_out': n, **stats, 'note': '合成放大（重放+扰动）'})
    elif args.mode == 'events':
        import m2_l2_events as m2e          # noqa（仅 events 模式需要，py-files 分发）
        d = (spark.read.parquet(f'{wh}/perf_amp_{args.amp}x')
             .select('bvid', 'batch_date', 'time_offset', 'attr_weight'))
        w = (d.withColumn('b', (F.col('time_offset').cast('double') / 5).cast('int'))
             .groupBy('bvid', 'batch_date', 'b')
             .agg(F.sum('attr_weight').alias('w')))
        g = w.groupBy('bvid', 'batch_date').agg(
            F.collect_list(F.struct('b', 'w')).alias('bw'))

        def one(r):
            bw = r['bw']
            arr = np.zeros(int(max(x['b'] for x in bw)) + 1)
            for x in bw:
                arr[int(x['b'])] += (x['w'] or 0.0)
            if arr.size < 3:
                return 0
            return len(m2e.run_detectors(arr, params))
        t0 = time.time()
        n_ev = g.rdd.map(one).sum()
        t1 = time.time()
        record({'mode': 'events', 'amp': args.amp, 'par': par,
                'wall_s': round(t1 - t0, 1), 'n_events': int(n_ev),
                'n_series': g.count(), 'note': '合成放大（重放+扰动）'})
    elif args.mode == 'hawkes':
        from hawkes_core import fit_video
        h = params.get('hawkes', {})
        cfg = {'min_events': int(h.get('min_events', 1000)),
               'window_days': float(h.get('window_days', 21)),
               'ks_p': float(h.get('ks_p', 0.05))}
        dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean').filter(~F.col('is_spam'))
               .select('bvid', 'batch_date', 'send_time'))
        dim = (spark.read.parquet(f'{wh}/dim_video')
               .select('bvid', F.col('pubdate').cast('long').alias('pubdate'),
                       'first_batch', 'last_batch'))
        g = (dwd.join(dim, 'bvid')
             .filter(F.col('batch_date') == F.col('last_batch'))
             .filter(F.col('send_time').isNotNull() & F.col('pubdate').isNotNull())
             .groupBy('bvid')
             .agg(F.sort_array(F.collect_list('send_time')).alias('ts'),
                  F.first('pubdate').alias('pubdate'),
                  F.first('first_batch').alias('first_batch')))
        rows = g.collect()

        def fb_epoch(fb):
            return datetime.strptime(str(int(fb)), '%Y%m%d').replace(tzinfo=TZ8).timestamp()

        def run_one(r):
            return fit_video(list(r['ts']), r['pubdate'], fb_epoch(r['first_batch']), cfg)
        t0 = time.time()
        for r in rows:                       # 串行基线（driver 单线程）
            run_one(r)
        t_ser = time.time() - t0
        t0 = time.time()
        g.rdd.map(run_one).count()           # 并行（local[N]）
        t_par = time.time() - t0
        record({'mode': 'hawkes', 'par': par, 'n_groups': len(rows),
                'serial_s': round(t_ser, 1), 'parallel_s': round(t_par, 1),
                'speedup': round(t_ser / t_par, 2) if t_par > 0 else None})
    elif args.mode == 'skew':
        src = f'{wh}/perf_amp_{args.amp}x' if args.amp > 1 else f'{wh}/dwd_danmaku_clean'
        d = (spark.read.parquet(src).filter(~F.col('is_spam'))
             .select('bvid', 'batch_date', 'time_offset', 'attr_weight'))
        k = ['bvid', 'batch_date', 'b']
        bb = ((F.col('time_offset').cast('double') / 5).cast('int')).alias('b')
        # 普通
        plain = d.select('bvid', 'batch_date', 'attr_weight', bb).groupBy(*k) \
                 .agg(F.count('*').alias('cnt'), F.sum('attr_weight').alias('w')).cache()
        t0 = time.time(); plain.count(); t1 = time.time()
        t_plain = t1 - t0
        ps = part_stats(plain)
        # 加盐两阶段
        salted = (d.select('bvid', 'batch_date', 'attr_weight', bb)
                  .withColumn('salt', F.pmod(F.hash(F.col('bvid'), F.col('batch_date')), F.lit(8)))
                  .groupBy(*k, 'salt').agg(F.count('*').alias('c'), F.sum('attr_weight').alias('w'))
                  .groupBy(*k).agg(F.sum('c').alias('cnt'), F.sum('w').alias('w')).cache())
        t0 = time.time(); salted.count(); t1 = time.time()
        t_salt = t1 - t0
        ss = part_stats(salted)
        record({'mode': 'skew', 'par': par, 'plain_s': round(t_plain, 1),
                'salted_s': round(t_salt, 1),
                'plain_skew': ps['skew'], 'salted_skew': ss['skew'],
                'plain': ps, 'salted': ss})
    spark.stop()


if __name__ == '__main__':
    main()
