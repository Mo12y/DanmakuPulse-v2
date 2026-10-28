#!/usr/bin/env python3
"""
m2_l1_rhythm60s.py — C1 节奏阶段识别的分钟级序列表（任务 A；2026-10-28）

数据链（可追溯到原始爬取 CSV）：
  data/cli_collected/hotlist_top200_*.csv
    → make m0-daily / m0-full   （Spark 清洗：cast/去重/打标/刷量检测 → dwd_danmaku_clean）
    → 本作业                     （Spark 60s 聚合 + 完整性/缺失/异常质量标记）
    → {wh}/dws_rhythm_60s（桶粒度） + {wh}/dws_rhythm_60s_meta（视频粒度）

为什么以 DWD 为输入而不是重读原始 CSV：
  DWD 已是 Spark 对原始 CSV 的清洗产物（口径见 jobs/m0_full.py 头注：dedup_key、打标不删除、
  attr_weight、is_spam）。本作业在其上做 60s 聚合与质量标记，不重复清洗；
  原始 CSV → DWD 的行数对账由 m0_full 的运行报告承载（ods_rows / dwd_rows）。

口径（本作业定死，改动须升版本并记录）：
  - 主口径：is_spam == false（10 §3，与 dws_density_5s 一致）
  - 时间窗：bucket60 = floor(time_offset / 60)，域 = [0, n_exp)，
    n_exp = max(floor(duration / 60), 1)（与 exp47/48.1 的分钟桶口径一致，含末桶截断）
  - 时间戳：session 时区 Asia/Shanghai；pubdate / send_time 为 epoch 秒；
    fetched_at 取 ods_video_fetch_times 按 (bvid,batch) 的 max（采集快照时刻）
  - **"没有弹幕" vs "采集缺失"**：桶的绝对结束时刻 = pubdate + (bucket_idx+1)*60；
    该时刻 > fetched_at ⇒ 桶落在采集快照之后，真实计数未知 ⇒ bucket_state='post_snapshot'，
    **不得当作 0**；快照之前计数为 0 ⇒ 'observed_zero'；有弹幕 ⇒ 'observed_nonzero'
  - fetched_at 缺失 ⇒ 'unknown_fetch'（视频级 fetch_available=false），同样不得当 0
  - 内部空洞：observed 桶中，被两个非零 observed 桶夹着的零桶 ⇒ 视频级 n_interior_holes
    （区别于片头/片尾零段——后者可能是真实低活动）
  - 多 P：part_count >= 2 的时间轴是"拼接整轴"，与单 P 语义不同 ⇒ 保留在表内但
    is_multi_part=true、c1_eligible=false（C1 只用单 P；记录而非静默丢弃）
  - 异常时间戳：time_offset < 0 或 > duration ⇒ 不进桶，计入 invalid_offset_rows
    （DWD 的 quality_flag 已打标，此处只统计，不改变 DWD）
  - 去重核对：DWD 应已按 params.etl.dedup_key 去重；本作业对 (batch_date,bvid,text,time_offset)
    复数一遍，期望 0——非 0 说明上游口径变化，直接 fail

增量：--batches 20261020,20261021 只重算指定批（dynamic partition overwrite）。
用法：
  集群：spark-submit ... /opt/work/jobs/m2_l1_rhythm60s.py [--batches a,b]
  本地：~/danmaku-venv/bin/python jobs/m2_l1_rhythm60s.py --local --wh-root file:data-out/dev-wh
"""
import argparse
import json
import os
import time

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

ENGINE_VERSION = 'c1-v1'
BUCKET_SEC = 60
C1_GOT_MAX = 600          # got < 600：池=最近 N 条后缀的截断风险低，全片可观测（40 号 §C1）
C1_DUR_MIN = 300          # duration >= 300s
C1_MIN_BUCKETS = 5        # 至少 5 个分钟桶才有"阶段"可言


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--batches', default='', help='逗号分隔批次；空=全量')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    batches = [b.strip() for b in args.batches.split(',') if b.strip()]
    wh = args.wh_root
    t0 = time.time()

    b = (SparkSession.builder.appName('m2_l1_rhythm60s')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.sources.partitionOverwriteMode', 'dynamic')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    d = spark.read.parquet(f'{wh}/dwd_danmaku_clean').filter(~F.col('is_spam'))
    if batches:
        d = d.filter(F.col('batch_date').isin(batches))

    # ---- 0) 输入画像与核对（写进报告，不得省略）----
    # 去重键以 params.etl.dedup_key 为准 = (batch_date, bvid, cid, row_id)（2026-10-12 切换；
    # 旧键 (text,time_offset) 会误删 4,492 条「同文同刻但 row_id 全唯一」的真实复读弹幕——
    # 见 config/params.yaml 注释与 tools/probe_dedup_keys.py）。同文同刻的行数单列统计，
    # 属真实复读，不是重复记录。
    input_rows = d.count()
    DEDUP_KEY = ['batch_date', 'bvid', 'cid', 'row_id']
    dup = (d.groupBy(*DEDUP_KEY).count().filter(F.col('count') > 1).count())
    if dup:
        raise SystemExit(f'❌ DWD 内仍有 {dup} 组 {DEDUP_KEY} 重复——上游去重口径变化，中止')
    same_text_time = (d.groupBy('batch_date', 'bvid', 'text', 'time_offset').count()
                      .filter(F.col('count') > 1).count())
    invalid = d.filter((F.col('time_offset') < 0) | (F.col('time_offset') > F.col('duration')))
    invalid_rows = invalid.count()

    # ---- 1) 视频粒度元数据 + fetched_at ----
    # pubdate 在 DWD 里是字符串（采集 CSV 原样），绝对时间运算前必须显式 cast long
    vid = (d.groupBy('bvid', 'batch_date')
           .agg(F.first('duration').alias('duration'),
                F.first('pubdate').cast('long').alias('pubdate'),
                F.first('danmaku_got').alias('danmaku_got'),
                F.first('tname').alias('tname'),
                F.first('part_count').alias('part_count'),
                F.max('time_offset').alias('last_offset_s'),
                F.count('*').alias('n_danmaku')))
    fetch = (spark.read.parquet(f'{wh}/ods_video_fetch_times')
             .groupBy('bvid', 'batch_date').agg(F.max('fetched_at').alias('fetched_at')))
    vid = vid.join(fetch, ['bvid', 'batch_date'], 'left')

    n_exp = F.greatest(F.floor(F.col('duration') / BUCKET_SEC).cast('long'), F.lit(1))
    # fetched_at 是 ISO 带 T 的字符串（实测 2026-10-20T10:49:16）：unix_timestamp 默认格式
    # 解析 2650/2650 全 null，必须显式 "yyyy-MM-dd'T'HH:mm:ss"，再兜底默认格式
    fetch_epoch = F.coalesce(F.unix_timestamp('fetched_at', "yyyy-MM-dd'T'HH:mm:ss"),
                             F.unix_timestamp('fetched_at'))
    vid = vid.withColumn('n_buckets_expected', n_exp)
    vid = vid.withColumn('fetch_epoch', fetch_epoch)
    vid = vid.withColumn('fetch_available', F.col('fetch_epoch').isNotNull())

    # ---- 2) 桶粒度：稠密域 [0, n_exp) ----
    buckets = (vid.select('bvid', 'batch_date', 'n_buckets_expected', 'pubdate',
                          'fetch_epoch', 'fetch_available')
               .withColumn('bucket60', F.explode(F.sequence(F.lit(0), F.col('n_buckets_expected') - 1))))
    agg = (d.filter(F.col('time_offset').isNotNull())
           .filter((F.col('time_offset') >= 0) & (F.col('time_offset') <= F.col('duration')))
           .withColumn('bucket60', F.floor(F.col('time_offset') / BUCKET_SEC).cast('long'))
           .groupBy('bvid', 'batch_date', 'bucket60')
           .agg(F.count('*').alias('cnt'), F.sum('attr_weight').alias('wcnt')))
    buckets = (buckets.join(agg, ['bvid', 'batch_date', 'bucket60'], 'left')
               .fillna({'cnt': 0, 'wcnt': 0.0})
               .withColumn('t_start_s', F.col('bucket60') * BUCKET_SEC)
               .withColumn('t_end_s', (F.col('bucket60') + 1) * BUCKET_SEC))
    # 桶绝对结束时刻 > 采集快照 ⇒ 采集后时段，真实计数未知
    end_abs = F.col('pubdate') + (F.col('bucket60') + 1) * BUCKET_SEC
    state = (F.when(~F.col('fetch_available'), F.lit('unknown_fetch'))
             .when(end_abs > F.col('fetch_epoch'), F.lit('post_snapshot'))
             .when(F.col('cnt') > 0, F.lit('observed_nonzero'))
             .otherwise(F.lit('observed_zero')))
    buckets = buckets.withColumn('bucket_state', state)

    # ---- 3) 视频粒度质量标记 ----
    w = Window.partitionBy('bvid', 'batch_date').orderBy('bucket60')
    g = Window.partitionBy('bvid', 'batch_date')
    bk = buckets.withColumn('prev_nonzero',
                            F.last(F.when(F.col('bucket_state') == 'observed_nonzero', F.col('bucket60'))).over(w))
    # 内部空洞：observed_zero 且其后（同视频）仍出现 observed_nonzero
    later_nonzero = (bk.withColumn('max_nonzero_after',
                                   F.max(F.when(F.col('bucket_state') == 'observed_nonzero',
                                                F.col('bucket60'))).over(
                                       Window.partitionBy('bvid', 'batch_date')
                                       .orderBy(F.col('bucket60'))
                                       .rowsBetween(1, Window.unboundedFollowing)))
                     .withColumn('is_interior_hole',
                                 (F.col('bucket_state') == 'observed_zero') & F.col('max_nonzero_after').isNotNull()))
    q = (later_nonzero.groupBy('bvid', 'batch_date')
         .agg(F.sum(F.when(F.col('bucket_state') == 'observed_nonzero', 1).otherwise(0)).alias('n_buckets_nonzero'),
              F.sum(F.when(F.col('bucket_state') == 'observed_zero', 1).otherwise(0)).alias('n_zero_observed'),
              F.sum(F.when(F.col('bucket_state') == 'post_snapshot', 1).otherwise(0)).alias('n_post_snapshot'),
              F.sum(F.when(F.col('bucket_state') == 'unknown_fetch', 1).otherwise(0)).alias('n_unknown_fetch'),
              F.sum(F.when(F.col('is_interior_hole'), 1).otherwise(0)).alias('n_interior_holes'),
              F.max(F.when(F.col('bucket_state') != 'post_snapshot', F.col('bucket60'))).alias('last_observed_bucket')))
    meta = (vid.join(q, ['bvid', 'batch_date'], 'left')
            .fillna({'n_buckets_nonzero': 0, 'n_zero_observed': 0,
                     'n_post_snapshot': 0, 'n_unknown_fetch': 0, 'n_interior_holes': 0}))
    meta = (meta
            # part_count 仅多 P 视频才写值（0902 批 2794/129976 非空、1020 批 33609/217145），
            # null ⇒ 单 P；这是采集器行为，不是数据缺失
            .withColumn('is_multi_part',
                        F.coalesce(F.col('part_count').cast('int') >= 2, F.lit(False)))
            .withColumn('coverage_ratio',
                        F.round(F.col('n_buckets_nonzero') / F.col('n_buckets_expected'), 4))
            .withColumn('c1_eligible',
                        (F.col('danmaku_got') < C1_GOT_MAX) &
                        (F.col('duration') >= C1_DUR_MIN) &
                        (~F.col('is_multi_part')) &
                        (F.col('n_buckets_expected') >= C1_MIN_BUCKETS))
            .withColumn('exclude_reason',
                        F.when(F.col('is_multi_part'), F.lit('multi_part'))
                        .when(F.col('danmaku_got') >= C1_GOT_MAX, F.lit('got_ge_600_pool_truncation_risk'))
                        .when(F.col('duration') < C1_DUR_MIN, F.lit('duration_lt_300s'))
                        .when(F.col('n_buckets_expected') < C1_MIN_BUCKETS, F.lit('too_few_buckets'))
                        .otherwise(F.lit('')))
            .withColumn('engine_version', F.lit(ENGINE_VERSION)))

    buckets = buckets.withColumn('engine_version', F.lit(ENGINE_VERSION))
    buckets.select('bvid', 'batch_date', 'bucket60', 't_start_s', 't_end_s', 'cnt', 'wcnt',
                   'bucket_state', 'engine_version') \
           .write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_rhythm_60s')
    meta.select('bvid', 'batch_date', 'tname', 'duration', 'pubdate', 'danmaku_got', 'part_count',
                'is_multi_part', 'n_buckets_expected', 'n_buckets_nonzero', 'n_zero_observed',
                'n_post_snapshot', 'n_unknown_fetch', 'n_interior_holes', 'last_observed_bucket',
                'last_offset_s', 'n_danmaku', 'coverage_ratio', 'fetch_available', 'fetched_at',
                'c1_eligible', 'exclude_reason', 'engine_version') \
        .write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_rhythm_60s_meta')

    # ---- 4) 运行报告 ----
    n_units = meta.count()
    n_bucket_rows = buckets.count()
    n_elig = meta.filter(F.col('c1_eligible')).count()
    n_elig_videos = meta.filter(F.col('c1_eligible')).select('bvid').distinct().count()
    excl = {str(r['exclude_reason'] or 'ELIGIBLE'): int(r['n'])
            for r in meta.groupBy('exclude_reason').agg(F.count('*').alias('n')).collect()}
    rep = {
        'job': 'm2_l1_rhythm60s', 'engine_version': ENGINE_VERSION, 'bucket_sec': BUCKET_SEC,
        'input': {'path': f'{wh}/dwd_danmaku_clean', 'rows': input_rows,
                  'dup_groups_actual_key': dup, 'dedup_key': DEDUP_KEY,
                  'same_text_time_groups_kept': same_text_time,
                  'invalid_offset_rows': invalid_rows,
                  'batches': batches or 'ALL'},
        'output': {'dws_rhythm_60s_rows': n_bucket_rows, 'meta_units': n_units,
                   'c1_eligible_units': n_elig, 'c1_eligible_videos': n_elig_videos,
                   'exclude_breakdown': excl},
        'rules': {'spam': 'is_spam=false', 'bucket': 'floor(time_offset/60), domain [0,floor(duration/60))',
                  'tz': 'Asia/Shanghai', 'post_snapshot': 'pubdate+(bucket+1)*60 > fetched_at → 未知，不当 0',
                  'c1_filter': f'got<{C1_GOT_MAX} & duration>={C1_DUR_MIN} & 单P & 桶数>={C1_MIN_BUCKETS}'},
        'runtime_sec': round(time.time() - t0, 1),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'rhythm60s_report.json'), 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f'✅ m2_l1_rhythm60s：桶 {n_bucket_rows:,} 行 ｜ 单元 {n_units:,} ｜ C1 合格 {n_elig:,} 单元 / {n_elig_videos:,} 支')
    spark.stop()


if __name__ == '__main__':
    main()
