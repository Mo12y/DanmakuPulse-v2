#!/usr/bin/env python3
"""
verify_l1_baseline.py — L1 分区基准 DoD 验收（08 §3 DoD：基准与 pandas 复算一致率 100%）

两个阶段：
  --export（集群 spark-submit）：导出 3 个非 low_support 分区的 curve_med 数组 + 明细 (bvid, offset_pct) → data-out/l1_verify/
  --check（宿主机 venv pandas）：重算每桶 median → 与 Spark percentile 对比 → 一致率

percentile 口径对齐：Spark `percentile(cnt, 0.5)`（exact）；numpy 侧尝试 method 以复现 Spark 定义，
  匹配到一致即采信该 method（浮点容差 1e-9）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'data-out', 'l1_verify')


def do_export(wh):
    import pyspark.sql.functions as F
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.appName('verify_l1_baseline_export')
             .config('spark.sql.shuffle.partitions', '8').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    os.makedirs(OUT, exist_ok=True)
    base = (spark.read.parquet(f'{wh}/dws_partition_rhythm')
            .filter(~F.col('low_support')).orderBy(F.col('n_videos').desc()).limit(3))
    tnames = [r['tname'] for r in base.select('tname').collect()]
    curves = {}
    for r in base.select('tname', 'curve_med').collect():
        curves[r['tname']] = r['curve_med']
    with open(os.path.join(OUT, 'targets.json'), 'w', encoding='utf-8') as f:
        json.dump({'tnames': tnames, 'curves': curves}, f, ensure_ascii=False)
    dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean')
           .filter(~F.col('is_spam')).filter(F.col('tname').isin(tnames))
           .filter(F.col('offset_pct').isNotNull())
           .select('tname', 'bvid', 'offset_pct'))
    rows = dwd.collect()
    with open(os.path.join(OUT, 'detail.tsv'), 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(f"{r['tname']}\t{r['bvid']}\t{r['offset_pct']}\n")
    print(f'exported {len(rows)} rows for {tnames} -> {OUT}')
    spark.stop()


def do_check():
    import numpy as np
    targets = json.load(open(os.path.join(OUT, 'targets.json'), encoding='utf-8'))
    detail = {}
    for ln in open(os.path.join(OUT, 'detail.tsv'), encoding='utf-8'):
        t, b, o = ln.rstrip('\n').split('\t')
        detail.setdefault(t, []).append((b, float(o)))
    for method in ('linear', 'lower', 'higher', 'nearest'):
        total = match = 0
        worst = 0.0
        for t in targets['tnames']:
            sp_curve = targets['curves'][t]
            rows = detail[t]
            buckets = {}
            for b, o in rows:
                buckets.setdefault(min(int(o * 100), 99), []).append(b)
            # per (bvid,bucket) count → per bucket median
            per = {}
            for b, o in rows:
                per.setdefault((b, min(int(o * 100), 99)), 0)
                per[(b, min(int(o * 100), 99))] += 1
            import collections
            by_bucket = collections.defaultdict(list)
            for (b, bk), c in per.items():
                by_bucket[bk].append(c)
            for bk in range(100):
                cnts = by_bucket.get(bk, [])
                sp = sp_curve[bk] if bk < len(sp_curve) else None
                if not cnts or sp is None:
                    continue
                py = float(np.percentile(np.array(cnts, dtype=float), 50, method=method))
                total += 1
                if abs(py - sp) <= 1e-9:
                    match += 1
                worst = max(worst, abs(py - sp))
        rate = match / max(total, 1)
        print(f'method={method}: match {match}/{total} = {rate:.4f} (worst_abs={worst:.4f})')
        if rate >= 0.9999:
            print(f'✅ 复算一致率 100%（method={method}）——DoD 第二半通过')
            return
    print('⚠️ 未找到 100% 一致的 percentile 定义——需人工核对 Spark percentile 语义')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--wh-root', default='/warehouse')
    args = ap.parse_args()
    if args.export:
        do_export(args.wh_root)
    elif args.check:
        do_check()
    else:
        print('usage: --export (集群) | --check (宿主机 pandas)')
        sys.exit(1)
