#!/usr/bin/env python3
"""
export_goldens.py — 生成 B0 对拍基准（goldens）——在宿主机用旧仓库 venv 运行（只读旧系统）

    ~/danmaku-venv/bin/python tools/export_goldens.py \
        --old-root ~/projects/danmaku --out tests/goldens

基准来源（口径说明）：
  - 曲线：旧仓库 data/warehouse/dws/dt=*/（Spark+NumPy UDF 产物，float64）
  - 8 维：用旧 SSOT 函数 extract_8_indicators 在 float64 下**重算**（非旧 MySQL 的 float32 列——
    FLOAT(32) 存储舍入 ~1e-6 相对误差，不能作 1e-9 容差的基准）
  - 数值用 repr()（17 位有效数字）写 CSV → 读回为同一 double（无损往返）
"""
import argparse
import csv
import os
import sys

OLD_ROOT = os.path.expanduser('~/projects/danmaku')
sys.path.insert(0, os.path.join(OLD_ROOT, 'analysis'))

from danmaku_metrics import extract_8_indicators, INDICATOR_KEYS  # noqa: E402  旧仓库 SSOT（只读 import）

from pyspark.sql import SparkSession  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-root', default=OLD_ROOT)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests', 'goldens'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spark = (SparkSession.builder.master('local[2]').appName('export-goldens')
             .config('spark.driver.memory', '2g').config('spark.sql.shuffle.partitions', '4')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')

    import glob
    dws_parts = sorted(glob.glob(os.path.join(args.old_root, 'data', 'warehouse', 'dws', 'dt=*')))
    assert dws_parts, f'旧仓库无 dws 分区：{args.old_root}'

    curve_rows, metric_rows = [], []
    for p in dws_parts:
        dt = p.split('=')[-1]
        df = spark.read.parquet(p)
        for r in df.select('bvid', 'duration', 'danmaku_count', 'density_curve').collect():
            curve = list(r['density_curve'])
            for b, c in enumerate(curve):
                curve_rows.append([r['bvid'], dt, b, repr(c)])
            # 与旧 metrics.py 完全相同的调用方式：extract_indicators(curve, duration_raw)
            inds = extract_8_indicators(curve, r['duration'])
            metric_rows.append([r['bvid'], dt] + [repr(float(inds[k])) for k in INDICATOR_KEYS]
                               + [repr(float(r['danmaku_count']))])
        print(f'  dt={dt}: {df.count()} 视频')

    with open(os.path.join(args.out, 'goldens_curve.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['bvid', 'dt', 'bucket', 'c'])
        w.writerows(curve_rows)
    with open(os.path.join(args.out, 'goldens_metrics.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['bvid', 'dt'] + INDICATOR_KEYS + ['danmaku_count'])
        w.writerows(metric_rows)

    print(f'✅ goldens 导出：{len(metric_rows)} 视频批次，{len(curve_rows)} 曲线点 → {args.out}')
    spark.stop()


if __name__ == '__main__':
    main()
