#!/usr/bin/env python3
"""
m1_reconcile.py — M1 数据质量与对账（B1.2）

输入：{wh}/dwd_danmaku_clean（m0_full 产）+ {wh}/ods_danmaku_full（blocked 视频信息）
输出：{wh}/dws_batch_audit（批次级）+ {wh}/dws_reconcile_detail（bvid×batch 明细）
      + PG ads_quality_report（集群默认写；--skip-pg 关闭）

三值对账（08 §2）：
  reported = danmaku_reported（B站报告数）；got = danmaku_got（采集数）；
  cnt_db   = dwd 行数（含 spam 排除口径 cnt_db_nospam——主口径=排除刷量，10 §3）
  diff = cnt_db − reported；diff_rate = |diff| / max(reported, 1)

归因（v0：证据优先级规则 + 差异方向 + 字幕池计数；见下）
  负差（cnt_db < reported）：
    1. multi_p      : part_count>1（多P 仅采 P1——part_count 字段仅多P视频由采集器填充）
    2. truncated    : |diff| > max(5, 2%×reported)（弹幕池截断——旧系统实测为负差主因，截断率与热度相关 r=-0.377）
    3. deleted      : 其余小量负差（弹幕被删除；⚠️ 无法直接观测，系兜底档）
  正差（cnt_db > reported）：
    4. subtitle_pool: subtitle_cnt>0 且 |diff| ≤ subtitle_cnt + 5（字幕池弹幕计入采集）
    5. excess       : 其余正差（08 四标签之外的残差单列，不强行归因；旧系统口径称"API 快照延迟"）
  对齐（|diff| ≤ 2）：none
  ⚠️ v0 为"证据优先级"规则而非因果判定；闭合率 = Σ|diff|(非 none) / Σ|diff|(全体)，DoD ≥90%

偏差分析：ok vs blocked 视频的 play 分布 Mann-Whitney U（driver 侧自实现 rank+正态近似，
不依赖 scipy——集群 driver 无第三方库）；bias_p = 双侧近似 p 值。

质量分（0-100）= 100 × [0.4·coverage + 0.3·(1−batch_diff_rate) + 0.3·(1−blocked_rate)]
  coverage = ok 视频数 / 批次视频总数；<60 告警（warn）、<40 红标（red）

用法：
  集群：spark-submit ... /opt/work/jobs/m1_reconcile.py
  本地：~/danmaku-venv/bin/python jobs/m1_reconcile.py --local --wh-root file:data-out/dev-wh \
        --dwd-table dwd_danmaku_clean_min（min 表兼容测试）
"""
import argparse
import json
import math
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b1-v1'
TOL_ALIGN = 2          # |diff| ≤ 2 视为对齐
MIN_ABS = 5            # 负差大阈值下限（绝对）
TRUNC_REL = 0.02       # 负差大阈值（相对 reported 2%）


def build_spark(local=False):
    b = (SparkSession.builder.appName('m1_reconcile')
         .config('spark.sql.shuffle.partitions', '8'))
    if local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    return b.getOrCreate()


def mannwhitneyu_pairs_p(a, b):
    """Mann-Whitney U 双侧 p（大样本正态近似 + tie 平均秩；driver 侧纯 stdlib）。"""
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return None, None
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    n = len(combined)
    rank_sum_a = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if combined[k][1] == 0:
                rank_sum_a += avg_rank
        i = j
    u1 = rank_sum_a - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return None, None
    z = (u1 - mu) / sigma
    p = math.erfc(abs(z) / math.sqrt(2))
    return p, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--dwd-table', default='dwd_danmaku_clean')
    ap.add_argument('--config-root',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config'))
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--skip-pg', action='store_true')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root
    out_dir = args.out_dir or ('/opt/work/data-out' if not args.local
                               else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out'))
    os.makedirs(out_dir, exist_ok=True)

    import yaml
    with open(os.path.join(args.config_root, 'params.yaml'), encoding='utf-8') as f:
        params = yaml.safe_load(f)
    q = params['quality']

    spark = build_spark(local=args.local)
    spark.sparkContext.setLogLevel('ERROR')

    dwd = spark.read.parquet(f'{wh}/{args.dwd_table}')
    has_spam = 'is_spam' in dwd.columns

    # ── per (bvid, batch) 三值对账聚合 ──
    agg_cols = [
        F.max('danmaku_reported').alias('reported'),
        F.max('danmaku_got').alias('got'),
        F.count('*').alias('cnt_db'),
        F.max('part_count').alias('part_count'),
        F.sum(F.when(F.col('pool') == 1, 1).otherwise(0)).alias('subtitle_cnt'),
    ]
    if has_spam:
        agg_cols.append(
            F.sum(F.when(F.col('is_spam'), 0).otherwise(1)).alias('cnt_db_nospam'))
    det = dwd.groupBy('bvid', 'batch_date').agg(*agg_cols)
    if not has_spam:
        det = det.withColumn('cnt_db_nospam', F.col('cnt_db'))

    det = (det
           .withColumn('diff', F.col('cnt_db') - F.col('reported'))
           .withColumn('diff_rate', F.abs(F.col('diff')) / F.greatest(F.col('reported'), F.lit(1))))
    # 归因（v0 证据优先级）
    det = det.withColumn('attrib',
        F.when(F.abs(F.col('diff')) <= TOL_ALIGN, 'none')
         .when(F.col('diff') < 0, F.when(
             F.col('part_count') > 1, 'multi_p')
             .when(F.abs(F.col('diff')) > F.greatest(F.lit(MIN_ABS),
                   F.col('reported') * TRUNC_REL), 'truncated')
             .otherwise('deleted'))
         .otherwise(F.when(
             (F.col('subtitle_cnt') > 0) &
             (F.abs(F.col('diff')) <= F.col('subtitle_cnt') + 5), 'subtitle_pool')
             .otherwise('excess')))

    det.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_reconcile_detail')

    # ── 批次汇总 ──
    audit = (det.groupBy('batch_date')
             .agg(F.count('*').alias('n_videos'),
                  F.sum('cnt_db').alias('sum_cnt_db'),
                  F.sum('reported').alias('sum_reported'),
                  F.sum(F.abs('diff')).alias('sum_abs_diff'),
                  F.sum(F.when(F.col('attrib') == 'none', F.abs('diff')).otherwise(0)).alias('sum_abs_diff_none'),
                  F.sum(F.when(F.col('attrib') == 'multi_p', F.abs('diff')).otherwise(0)).alias('diff_multi_p'),
                  F.sum(F.when(F.col('attrib') == 'truncated', F.abs('diff')).otherwise(0)).alias('diff_truncated'),
                  F.sum(F.when(F.col('attrib') == 'deleted', F.abs('diff')).otherwise(0)).alias('diff_deleted'),
                  F.sum(F.when(F.col('attrib') == 'subtitle_pool', F.abs('diff')).otherwise(0)).alias('diff_subtitle_pool'),
                  F.sum(F.when(F.col('attrib') == 'excess', F.abs('diff')).otherwise(0)).alias('diff_excess')))

    # blocked 分析（ods：ok vs 非 ok 的 play 分布）——driver 侧 Mann-Whitney
    ods = spark.read.parquet(f'{wh}/ods_danmaku_full')
    vstat = (ods.groupBy('bvid', 'batch_date')
             .agg(F.first('status').alias('status'), F.max('play').alias('play')))
    # 每批次内：ok 数 / blocked 数（用 det 已过滤 status=ok 的行数做 ok 数）
    n_ok = det.groupBy('batch_date').agg(F.count('*').alias('n_ok'))
    vb = vstat.groupBy('batch_date').agg(F.count('*').alias('n_total'))
    blocked_vids = (vstat.filter(F.col('status') != 'ok')
                    .select('batch_date', 'bvid', 'play'))
    ok_vids = (vstat.join(det.select('bvid', 'batch_date'), ['bvid', 'batch_date'], 'left_semi')
               .select('batch_date', 'bvid', 'play'))

    bias_rows = []
    for bd in [r[0] for r in vb.select('batch_date').collect()]:
        a = [r[0] for r in ok_vids.filter(F.col('batch_date') == bd).select('play').collect()]
        b = [r[0] for r in blocked_vids.filter(F.col('batch_date') == bd).select('play').collect()]
        p, z = mannwhitneyu_pairs_p(a, b)
        bias_rows.append((bd, len(a), len(b), p, z))
    from pyspark.sql.types import (DoubleType, IntegerType, StringType,
                                   StructField, StructType)
    bias_schema = StructType([
        StructField('batch_date', StringType()),
        StructField('n_ok_play', IntegerType()),
        StructField('n_blocked_play', IntegerType()),
        StructField('bias_p', DoubleType()),
        StructField('bias_z', DoubleType()),
    ])
    bias_df = spark.createDataFrame(bias_rows, schema=bias_schema)

    audit = (audit.join(n_ok, 'batch_date')
             .join(vb, 'batch_date')
             .withColumn('n_blocked', F.col('n_total') - F.col('n_ok'))
             .join(bias_df, 'batch_date', 'left')
             .withColumn('coverage', F.col('n_ok') / F.col('n_total'))
             .withColumn('blocked_rate', F.col('n_blocked') / F.col('n_total'))
             .withColumn('batch_diff_rate',
                         F.col('sum_abs_diff') / F.greatest(F.col('sum_reported'), F.lit(1)))
             .withColumn('attrib_closure',
                         (F.col('sum_abs_diff') - F.col('sum_abs_diff_none')) /
                         F.greatest(F.col('sum_abs_diff'), F.lit(1)))
             .withColumn('quality_score',
                         F.round(F.lit(100.0) * (F.lit(0.4) * F.col('coverage') +
                                                 F.lit(0.3) * (F.lit(1.0) - F.col('batch_diff_rate')) +
                                                 F.lit(0.3) * (F.lit(1.0) - F.col('blocked_rate'))), 2))
             .withColumn('quality_alert',
                         F.when(F.col('quality_score') < 40, 'red')
                          .when(F.col('quality_score') < 60, 'warn')
                          .otherwise('none'))
             .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    audit.write.mode('overwrite').parquet(f'{wh}/dws_batch_audit')

    # ── PG ads_quality_report ──
    if not args.skip_pg:
        url = 'jdbc:postgresql://postgres:5432/ads'
        props = {'user': os.environ.get('PG_USER', 'danmaku'),
                 'password': os.environ.get('PG_PASSWORD', ''),
                 'driver': 'org.postgresql.Driver'}
        (audit.orderBy('batch_date').coalesce(1)
              .write.jdbc(url, 'ads_quality_report', mode='overwrite', properties=props))
        print('✅ PG ads_quality_report 写入完成')

    # ── 报告 ──
    report_rows = [r.asDict() for r in audit.orderBy('batch_date').collect()]
    detail_n = det.count()
    report = {'job': 'm1_reconcile', 'engine_version': ENGINE_VERSION,
              'dwd_table': args.dwd_table, 'has_spam_col': has_spam,
              'n_detail_rows': detail_n, 'batches': report_rows,
              'attrib_defs': {'multi_p': 'part_count>1（多P仅采P1）',
                              'truncated': f'负差 > max({MIN_ABS},{TRUNC_REL:.0%}×reported)（弹幕池截断）',
                              'deleted': '小量负差（兜底）',
                              'subtitle_pool': '正差且 ≤ subtitle_cnt+5',
                              'excess': '其余正差（残差单列）',
                              'none': f'|diff| ≤ {TOL_ALIGN}'}}
    with open(os.path.join(out_dir, 'm1_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    for r in report_rows:
        print(f"  批次 {r['batch_date']}: n={r['n_videos']} 质量分={r['quality_score']} "
              f"({r['quality_alert']}) 闭合率={r['attrib_closure']:.4f} "
              f"blocked={r['n_blocked']} bias_p={r['bias_p']}")
    print(f'✅ m1_reconcile 完成：{len(report_rows)} 批 / {detail_n} 条明细')
    spark.stop()


if __name__ == '__main__':
    main()
