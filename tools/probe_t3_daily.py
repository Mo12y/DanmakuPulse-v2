#!/usr/bin/env python3
"""
probe_t3_daily.py — T3 探针（B4.4 / 10 §2.3）：日频批次的「次日重现率」与「≥2 次观测占比」

判定规则（已冻结，11 §3/§5——须积累 2–3 周后裁；当前任何时点跑属**预检**）：
  次日重现率 ≥0.2 且 ≥2 次观测视频占比 ≥30% → 保留 T3；否则降级「再次上榜分析」或移 P2。
口径（冻结决策）：仅 cadence='daily' 批次参与（weekly_legacy 严禁——其重叠≈0 无意义）；
  「次日」= 相邻日批对（严格 gap==1 天为主指标；多为 1 天，异常 gap 单列）。
输入：{wh}/ods_hotlist_video（bvid × batch_date）+ {wh}/ods_batch_meta（cadence）。
输出：stdout + /opt/work/data-out/t3_probe_report.json
"""
import argparse
import json
import os

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--min-batches', type=int, default=14,
                    help='判定所需最少日批次数（2 周=14；不足→预检不下结论）')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()

    b = (SparkSession.builder.appName('probe_t3_daily')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    hot = spark.read.parquet(f'{args.wh_root}/ods_hotlist_video').select('bvid', 'batch_date')
    meta = (spark.read.parquet(f'{args.wh_root}/ods_batch_meta')
            .select(F.col('batch_date').cast('int').alias('batch_date'), 'cadence'))
    daily = (hot.join(meta, 'batch_date', 'inner')
             .filter(F.col('cadence') == 'daily')
             .select('bvid', 'batch_date').distinct().cache())
    n_rows = daily.count()

    batches = sorted([r['batch_date'] for r in daily.select('batch_date').distinct().collect()])
    per_batch = {int(r['batch_date']): int(r['n']) for r in
                 daily.groupBy('batch_date').agg(F.count('*').alias('n')).collect()}
    sets = {}
    for bd in batches:
        sets[bd] = {r['bvid'] for r in daily.filter(F.col('batch_date') == bd).select('bvid').collect()}

    def gap_days(a, bd2):
        from datetime import datetime
        d1 = datetime.strptime(str(a), '%Y%m%d')
        d2 = datetime.strptime(str(bd2), '%Y%m%d')
        return (d2 - d1).days

    pairs = []
    for i in range(len(batches) - 1):
        a, c = batches[i], batches[i + 1]
        inter = len(sets[a] & sets[c])
        pairs.append({'b1': int(a), 'b2': int(c), 'gap_days': gap_days(a, c),
                      'n1': len(sets[a]), 'n2': len(sets[c]), 'overlap': inter,
                      'overlap_frac_of_b1': round(inter / max(len(sets[a]), 1), 4)})
    next_day = [p['overlap_frac_of_b1'] for p in pairs if p['gap_days'] == 1]

    # ≥2 次观测占比
    all_videos = daily.select('bvid').distinct()
    n_all = all_videos.count()
    obs = (daily.groupBy('bvid').agg(F.countDistinct('batch_date').alias('k'))
           .filter(F.col('k') >= 2).count())
    multi_frac = round(obs / max(n_all, 1), 4)

    mean_next = round(sum(next_day) / len(next_day), 4) if next_day else None
    enough = len(batches) >= args.min_batches
    judged = None
    if enough:
        judged = ('保留' if (mean_next is not None and mean_next >= 0.2 and multi_frac >= 0.30)
                  else '降级/移P2（按冻结判据）')
    report = {
        'job': 'probe_t3_daily',
        'n_daily_batches': len(batches), 'batches': batches,
        'n_daily_rows': n_rows, 'per_batch_n': per_batch,
        'pairs': pairs,
        'next_day_replay_frac_mean': mean_next,
        'multi_observed_videos': obs, 'distinct_videos': n_all,
        'multi_observed_frac': multi_frac,
        'thresholds': {'next_day_replay': 0.2, 'multi_observed': 0.30, 'min_batches': args.min_batches},
        'judged': judged,
        'note': ('预检：不足 %d 个日批次，按冻结规则不下结论' % args.min_batches
                 if not enough else '达判定样本量，按冻结规则给出判定'),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 't3_probe_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ T3 探针：{len(batches)} 个日批次；次日重现率(mean)={mean_next}；'
          f'≥2 次观测占比={multi_frac}；判定={judged}')
    spark.stop()


if __name__ == '__main__':
    main()
