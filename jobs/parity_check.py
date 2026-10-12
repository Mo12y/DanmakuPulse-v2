#!/usr/bin/env python3
"""
parity_check.py — B0 对拍验收（DSH 建议①③ + 用户裁定 2026-10-12）

对拍对象：
  A. 逐 (bvid, batch_date)：新 Spark SQL 曲线 101 点 vs 旧 NumPy SSOT 曲线（goldens）
     —— 容差 ≤1e-9（浮点）；逐点最大绝对差报告
  B. 逐 (bvid, batch_date)：8 维指标 vs goldens 重算值（float64，非旧 MySQL 的 float32）
     —— peak_count / peak_fwhm / danmaku_count 精确一致（零容差）
     —— 浮点维（main_peak_pct / cold_pct / front_ratio / mid_std / end_rebound / cv）≤1e-9
  C. 池级锚点（DSH 建议③，2026-10-12 一手实测，出处 docs/00-规范/结论权威口径.md §2.9）：
     视频批次数=509 / DWD 弹幕总量=672,439 / 池 n=508（每 bvid 最新批）/ 池内弹幕=669,995
PASS 条件：A/B/C 全过；任一不过 → FAIL（细节落 parity_report.json 定位到视频与维度）。
"""
import argparse
import json
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import Window
import pyspark.sql.functions as F

FLOAT_COLS = ['main_peak_pct', 'cold_pct', 'front_ratio', 'mid_std', 'end_rebound', 'cv']
INT_COLS = ['peak_count', 'peak_fwhm', 'danmaku_count']
FLOAT_TOL = 1e-9

# 池级锚点（2026-10-12 一手实测：旧仓库 warehouse parquet + MySQL 活库）
ANCHORS = {'n_rows': 509, 'sum_danmaku_dwd': 672439,
           'pool_n': 508, 'pool_sum': 669995}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--goldens-dir', default='/opt/goldens')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    out_dir = args.out_dir or ('/opt/work/data-out' if not args.local
                               else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out'))
    os.makedirs(out_dir, exist_ok=True)

    b = (SparkSession.builder.appName('parity_check')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # ── goldens（旧 NumPy SSOT 输出，float64 repr 写入 → 读回无损）──
    g_curve = (spark.read.option('header', True).csv(f'{args.goldens_dir}/goldens_curve.csv')
               .withColumn('bucket', F.col('bucket').cast('int'))
               .withColumn('c', F.col('c').cast('double'))
               .withColumnRenamed('dt', 'g_batch'))
    g_m_raw = spark.read.option('header', True).csv(f'{args.goldens_dir}/goldens_metrics.csv')
    g_m_r = g_m_raw.withColumnRenamed('dt', 'g_batch')
    g_m = (g_m_r.select([F.col(c).alias('g_' + c) if c not in ('bvid', 'g_batch') else F.col(c)
                         for c in g_m_r.columns]))
    for c in INT_COLS + FLOAT_COLS:
        g_m = g_m.withColumn('g_' + c, F.col('g_' + c).cast('double'))

    m = spark.read.parquet(f'{args.wh_root}/dws_video_metrics_min')
    cv = spark.read.parquet(f'{args.wh_root}/dws_video_curve_min')
    curve = cv.select('batch_date', 'bvid', F.posexplode('curve').alias('bucket', 'c'))
    report = {'job': 'parity_check', 'tolerance': {'float': FLOAT_TOL, 'int_exact': True}}

    # ── A. 曲线 101 点对拍 ──
    jn = (curve.withColumnRenamed('c', 'c_new')
          .join(g_curve.withColumnRenamed('c', 'c_gold'),
                (curve.bvid == g_curve.bvid) & (curve.batch_date == g_curve.g_batch)
                & (curve.bucket == g_curve.bucket), 'inner'))
    orphan_new = curve.join(g_curve, (curve.bvid == g_curve.bvid) &
                            (curve.batch_date == g_curve.g_batch) &
                            (curve.bucket == g_curve.bucket), 'left_anti').count()
    orphan_gold = g_curve.join(curve, (g_curve.bvid == curve.bvid) &
                               (g_curve.g_batch == curve.batch_date) &
                               (g_curve.bucket == curve.bucket), 'left_anti').count()
    curve_max = (jn.select(F.abs(F.col('c_new') - F.col('c_gold')).alias('d'))
                 .agg(F.max('d')).collect()[0][0]) or 0.0
    report['curve'] = {'points_new': curve.count(), 'points_gold': g_curve.count(),
                       'orphan_new': orphan_new, 'orphan_gold': orphan_gold,
                       'max_abs_diff': curve_max,
                       'pass': orphan_new == 0 and orphan_gold == 0 and curve_max <= FLOAT_TOL}

    # ── B. 8 维对拍 ──
    jm = (m.select('batch_date', 'bvid', *INT_COLS, *FLOAT_COLS)
          .join(g_m, (m.bvid == g_m.bvid) & (m.batch_date == g_m.g_batch), 'inner'))
    dims = {}
    for c in INT_COLS + FLOAT_COLS:
        tol = 0 if c in INT_COLS else FLOAT_TOL
        n_bad = jm.filter(F.abs(F.col(c) - F.col('g_' + c)) > tol).count()
        max_d = jm.agg(F.max(F.abs(F.col(c) - F.col('g_' + c)))).collect()[0][0] or 0.0
        dims[c] = {'bad_rows': n_bad, 'max_abs_diff': max_d,
                   'tolerance': tol, 'pass': n_bad == 0}
    report['metrics'] = {'rows_new': m.count(), 'rows_gold': g_m_raw.count(),
                         'rows_matched': jm.count(), 'dims': dims,
                         'pass': (m.count() == g_m_raw.count() == jm.count()
                                  and all(d['pass'] for d in dims.values()))}

    # ── C. 池级锚点 ──
    n_rows = m.count()
    sum_dm = m.agg(F.sum('danmaku_count')).collect()[0][0]
    pool = (m.withColumn('rn', F.row_number().over(
        Window.partitionBy('bvid').orderBy(F.col('batch_date').desc())))
        .filter('rn = 1'))
    pool_n = pool.count()
    pool_sum = pool.agg(F.sum('danmaku_count')).collect()[0][0]
    report['anchors'] = {
        'n_rows': {'got': n_rows, 'expect': ANCHORS['n_rows'], 'pass': n_rows == ANCHORS['n_rows']},
        'sum_danmaku_dwd': {'got': sum_dm, 'expect': ANCHORS['sum_danmaku_dwd'],
                            'pass': sum_dm == ANCHORS['sum_danmaku_dwd']},
        'pool_n': {'got': pool_n, 'expect': ANCHORS['pool_n'], 'pass': pool_n == ANCHORS['pool_n']},
        'pool_sum': {'got': pool_sum, 'expect': ANCHORS['pool_sum'],
                     'pass': pool_sum == ANCHORS['pool_sum']}}
    report['anchors']['pass'] = all(v['pass'] for v in report['anchors'].values())

    report['PASS'] = report['curve']['pass'] and report['metrics']['pass'] and report['anchors']['pass']
    with open(os.path.join(out_dir, 'parity_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('✅ 对拍 PASS' if report['PASS'] else '❌ 对拍 FAIL —— 见 parity_report.json')
    spark.stop()
    sys.exit(0 if report['PASS'] else 1)


if __name__ == '__main__':
    main()
