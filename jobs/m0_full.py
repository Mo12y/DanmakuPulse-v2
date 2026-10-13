#!/usr/bin/env python3
"""
m0_full.py — M0 完整版（B1.1）：ods → 标准化 → 清洗打标 → 刷量检测 → 四表输出

输入（复用 B0 min 链已产出的 ODS，不重建）：
  {wh}/ods_danmaku_full       （B0 已产；新批次含 user_hash，历史 4 批无——unionByName 兼容）
  {raw}/batch_date=*/danmaku.csv（仅用于行数对账：csv → ods → dwd 全链等式）

输出（与 min 链完全隔离——min 链锚定对拍基准，本链为生产口径演进）：
  {wh}/dwd_danmaku_clean          标准化+打标+加权后的弹幕明细（打标不删除）
  {wh}/dim_video                  每 bvid 一行（最新批维度 + 首末批血缘）
  {wh}/dwd_video_metric_snapshot  每 (bvid, batch_date) 一行互动快照
  {wh}/dwd_spam_flag              每 (bvid, batch_date, user_hash) 刷量证据（仅 user_hash 可用批次）

口径与设计（08 §1 / 10 修订）：
  - 去重键：params.etl.dedup_key（批内语义；新旧键差异见 tools/probe_dedup_keys.py）
  - 清洗规则（打标不删除）：is_ad / is_meaningless / is_ultra_short（text_len<5，保留计数不入情感）
  - 属性加权（孔玲玲权重表，08 §1.2）：attr_weight = w_mode × w_color × w_font；pool=1（字幕池）→ 0
      权重补全说明：font_size 表仅给 25→1.0 / >25→1.2 / <18→0.9，18–24 未列——补 1.0（v0 裁定，记 CHANGELOG）
  - 时间纪律（08 §0.4）：UTC+8（session timeZone）。send_time(epoch) → send_date/send_hour；
      lag_sec = send_time − pubdate；lag_bucket ∈ {<1h,1-6h,6-24h,1-3d,>3d}
  - 刷量检测（08 §1.4，阈值全参数化自 params.quality）：p_u>share 且 cv_u<cv，或 burst_u>burst；
      视频级 spam_ratio>ratio → is_spam_video；**user_hash 覆盖 <50% 的批次整体跳过**（reason 落报告）。
      阈值定标=合成注入实验（tools/probe_spam_calibration.py；合成数据一律标注"合成放大"）
  - 主口径=排除刷量（10 §3）：本作业只打标；排除由下游读取实现（L1/ads 层按 is_spam=false）

用法：
  集群：spark-submit ... /opt/work/jobs/m0_full.py
  本地：~/danmaku-venv/bin/python jobs/m0_full.py --local --wh-root file:data-out/dev-wh
"""
import argparse
import json
import os
import re as _re

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window

ENGINE_VERSION = 'b1-v1'
LAG_BUCKETS = [
    ('<1h', 3600), ('1-6h', 6 * 3600), ('6-24h', 24 * 3600), ('1-3d', 3 * 24 * 3600),
]
DEFAULT_LAG = '>3d'

# 颜文字/emoji 归一（v0：emoji 类别 token + 区段兜底；原文保留在 text 列）
# pattern 为 Java regex（Spark regexp_replace），使用 \x{...} 码点语法——Python 侧必须用 raw string
EMOJI_RULES = [
    (r'[\x{1F602}\x{1F923}\x{1F606}\x{1F605}\x{1F600}\x{1F601}\x{1F60A}\x{1F642}]', '[EMOJI_LAUGH]'),
    (r'[\x{1F62D}\x{1F622}\x{1F625}\x{1F614}\x{1F972}]', '[EMOJI_CRY]'),
    (r'[\x{1F621}\x{1F620}\x{1F92C}]', '[EMOJI_ANGER]'),
    (r'[\x{1F631}\x{1F628}\x{1F633}\x{1F62F}]', '[EMOJI_SHOCK]'),
    (r'[\x{2764}\x{1F495}\x{1F496}\x{1F497}\x{1F49E}]', '[EMOJI_LOVE]'),
    (r'[\x{1F300}-\x{1FAFF}]', '[EMOJI_OTHER]'),
    (r'[\x{2600}-\x{27BF}]', '[SYM]'),
    (r'[\x{FE00}-\x{FE0F}]', ''),
]


def build_spark(local=False):
    b = (SparkSession.builder.appName('m0_full')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.sources.partitionOverwriteMode', 'dynamic')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai'))
    if local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    return b.getOrCreate()


def norm_text(col):
    """text_norm：emoji/颜文字归一（与 sample_check 共用同一实现）。"""
    out = col
    for pat, tok in EMOJI_RULES:
        out = F.regexp_replace(out, pat, tok)
    return out


def detect_spam(dwd, q, wh):
    """刷量检测（08 §1.4），粒度 (bvid, batch_date, user_hash)。
    批次级门限：仅 user_hash 覆盖率 ≥50% 的批次执行（历史 4 批无此字段——整批跳过并记录）。"""
    if 'user_hash' not in dwd.columns:
        n = dwd.count()
        return None, {'executed': False, 'reason': 'user_hash 列不存在（当前 ODS 仅历史批次）',
                      'n_total': n, 'n_with_user_hash': 0}
    cov = (dwd.groupBy('batch_date')
           .agg(F.count('*').alias('n'),
                F.sum(F.when(F.col('user_hash').isNotNull(), 1).otherwise(0)).alias('n_uh'))
           .withColumn('uh_rate', F.col('n_uh') / F.col('n')).collect())
    eligible = sorted([r['batch_date'] for r in cov if r['uh_rate'] >= 0.5])
    skipped = sorted([(r['batch_date'], round(r['uh_rate'], 4)) for r in cov if r['uh_rate'] < 0.5])
    if not eligible:
        return None, {'executed': False,
                      'reason': '无批次满足 user_hash 覆盖率 ≥50%（历史批次无此字段）',
                      'eligible_batches': [], 'skipped_batches': skipped}
    base = dwd.filter(F.col('batch_date').isin(eligible) & F.col('user_hash').isNotNull())
    n_with = base.count()
    n_total = dwd.count()

    # ⚠️ 2026-10-13 二修（F1）：range frame 的 orderBy 须数值型；此前因 user_hash 从未生效，
    #    该代码块从未被执行，两处隐藏 bug 今日首次暴露（lag.over 顺序 + string 排序键）。
    wu = (Window.partitionBy('bvid', 'batch_date', 'user_hash')
          .orderBy(F.col('send_time').cast('long')))
    w60 = wu.rangeBetween(-59, 0)
    per_u = (base
             .withColumn('gap', F.col('send_time').cast('long') -
                         F.lag('send_time').over(wu).cast('long'))
             .withColumn('burst_win', F.count('*').over(w60))
             .groupBy('bvid', 'batch_date', 'user_hash')
             .agg(F.count('*').alias('cnt_u'),
                  (F.stddev('gap') / F.abs(F.avg('gap'))).alias('cv_u'),
                  F.max('burst_win').alias('burst_u')))
    total_b = base.groupBy('bvid', 'batch_date').agg(F.count('*').alias('total_b'))

    cond_share = ((F.col('p_u') > q['spam_user_share']) &
                  F.col('cv_u').isNotNull() & (F.col('cv_u') < q['spam_cv']))
    cond_burst = F.col('burst_u') > q['spam_burst']
    flag = (per_u.join(total_b, ['bvid', 'batch_date'])
            .withColumn('p_u', F.col('cnt_u') / F.col('total_b'))
            .withColumn('is_spam', cond_share | cond_burst)
            .withColumn('reason',
                        F.when(cond_share & cond_burst, 'share+cv;burst')
                         .when(cond_share, 'share+cv')
                         .when(cond_burst, 'burst')
                         .otherwise('none')))
    spam_v = (flag.groupBy('bvid', 'batch_date')
              .agg(F.sum(F.when(F.col('is_spam'), F.col('cnt_u')).otherwise(0)).alias('spam_cnt'),
                   F.first('total_b').alias('total_b'))
              .withColumn('spam_ratio', F.col('spam_cnt') / F.col('total_b'))
              .withColumn('is_spam_video', F.col('spam_ratio') > q['spam_video_ratio']))
    flag = flag.join(spam_v.select('bvid', 'batch_date', 'spam_ratio', 'is_spam_video'),
                     ['bvid', 'batch_date'])
    flag = flag.select('bvid', 'batch_date', 'user_hash', 'is_spam', 'p_u', 'cv_u',
                       'burst_u', 'reason', 'spam_ratio', 'is_spam_video')

    flag.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dwd_spam_flag')

    n_u = flag.count()
    n_su = flag.filter('is_spam').count()
    n_vb = flag.select('bvid', 'batch_date').distinct().count()
    n_sv = flag.filter('is_spam_video').select('bvid', 'batch_date').distinct().count()
    batches = sorted(r[0] for r in flag.select('batch_date').distinct().collect())
    meta = {'executed': True, 'n_total': n_total, 'n_with_user_hash': n_with,
            'eligible_batches': eligible, 'skipped_batches': skipped,
            'n_users': n_u, 'n_spam_users': n_su,
            'spam_user_rate': round(n_su / max(n_u, 1), 6),
            'n_video_batches': n_vb, 'n_spam_videos': n_sv,
            'spam_video_rate': round(n_sv / max(n_vb, 1), 6),
            'batches': batches,
            'thresholds': {k: q[k] for k in
                           ['spam_user_share', 'spam_cv', 'spam_burst', 'spam_video_ratio']}}
    print(f'[spam] 检出：用户 {n_su}/{n_u}（{meta["spam_user_rate"]:.4f}），'
          f'视频×批次 {n_sv}/{n_vb}（{meta["spam_video_rate"]:.4f}）')
    return flag, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--raw-root', default='/raw')
    ap.add_argument('--config-root',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config'))
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root
    out_dir = args.out_dir or ('/opt/work/data-out' if not args.local
                               else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out'))
    os.makedirs(out_dir, exist_ok=True)

    import yaml
    with open(os.path.join(args.config_root, 'params.yaml'), encoding='utf-8') as f:
        params = yaml.safe_load(f)
    etl, q = params['etl'], params['quality']
    dedup_key = list(etl['dedup_key'])

    spark = build_spark(local=args.local)
    spark.sparkContext.setLogLevel('ERROR')

    # ── 0) csv 行数（对账链首环；读 raw 只为计数）──
    csv_df = (spark.read.option('header', True).option('quote', '"').option('escape', '"')
              .csv(f'{args.raw_root}/batch_date=*/danmaku.csv'))
    csv_rows = csv_df.count()
    csv_df.unpersist()

    # ── 1) 输入：B0 ODS（新批次有 user_hash、历史批无——兼容）──
    # ⚠️ 2026-10-13 修复(F1)：全表读需 mergeSchema——新批次（1012/1013）含 user_hash 列、
    #    历史 4 批无；默认 schema 取代表性文件（老批）会导致 user_hash '不存在' 而跳过刷量检测。
    ods = (spark.read.option('mergeSchema', 'true')
           .parquet(f'{wh}/ods_danmaku_full'))
    ods_rows = ods.count()

    # ── 2) 标准化器（08 §1.2）──
    raw_dur = F.col('duration').cast('double')
    d = (ods
         .withColumn('send_date', F.from_unixtime(F.col('send_time').cast('long'), 'yyyy-MM-dd'))
         .withColumn('send_hour', F.hour(F.to_timestamp(F.col('send_time').cast('long'))))
         .withColumn('lag_sec', F.col('send_time').cast('long') - F.col('pubdate').cast('long'))
         .withColumn('offset_pct',
                     F.when(raw_dur > 0, F.col('time_offset') / raw_dur).otherwise(F.lit(None)))
         .withColumn('is_subtitle_pool', F.col('pool') == 1)
         .withColumn('is_colored', F.col('color') != 16777215)
         .withColumn('is_special_mode', F.col('mode').isin(4, 5, 6, 7, 8, 9)))
    lag_expr = F.when(F.col('lag_sec').isNull(), F.lit(None).cast('string'))
    for label, hi in LAG_BUCKETS + [(DEFAULT_LAG, None)]:
        if hi is None:
            lag_expr = lag_expr.otherwise(label)
        else:
            lag_expr = lag_expr.when(F.col('lag_sec') < hi, label)
    d = d.withColumn('lag_bucket', lag_expr)
    d = d.withColumn('text_norm', norm_text(F.col('text')))
    d = d.withColumn('text_len_calc', F.length(F.col('text')))
    d = d.withColumn('is_ultra_short', F.col('text_len_calc') < int(etl['min_text_len']))
    d = (d.withColumn('is_ad', F.col('text').rlike(etl['ad_regex']))
          .withColumn('is_meaningless', F.col('text').rlike(etl['meaningless_regex'])))

    # ── 3) 属性加权（08 §1.2；pool=1 → 0 显式）──
    w_mode = (F.when(F.col('mode') == 1, 1.0).when(F.col('mode') == 4, 1.2)
              .when(F.col('mode') == 5, 1.3).when(F.col('mode') == 6, 1.1)
              .when(F.col('mode') == 7, 1.2)
              .when(F.col('mode').isin(8, 9), 0.0)
              .otherwise(1.0))
    w_color = F.when(F.col('color') == 16777215, 1.0).otherwise(1.3)
    fs = F.col('font_size')
    w_font = F.when(fs == 25, 1.0).when(fs > 25, 1.2).when(fs < 18, 0.9).otherwise(1.0)
    d = d.withColumn('attr_weight',
                     F.when(F.col('is_subtitle_pool'), F.lit(0.0))
                      .otherwise((w_mode * w_color * w_font).cast('double')))

    # ── 4) 去重（批内；键来自 params）──
    dwd = d.dropDuplicates(['batch_date'] + dedup_key)
    dups_removed = ods_rows - dwd.count()

    # ── 5) 刷量检测 + 回填 is_spam ──
    spam_flag, spam_meta = detect_spam(dwd, q, wh)
    if spam_flag is not None:
        dwd = (dwd.join(spam_flag.select('bvid', 'batch_date', 'user_hash', 'is_spam'),
                        ['bvid', 'batch_date', 'user_hash'], 'left')
                  .withColumn('is_spam', F.coalesce(F.col('is_spam'), F.lit(False))))
    else:
        dwd = dwd.withColumn('is_spam', F.lit(False))
        print(f'[spam] 跳过：{spam_meta["reason"]}')

    dwd_final = dwd.withColumn('engine_version', F.lit(ENGINE_VERSION))
    dwd_final.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dwd_danmaku_clean')
    dwd_rows = dwd_final.count()

    # ── 6) dim_video（每 bvid 最新批维度 + 血缘）──
    wv = Window.partitionBy('bvid').orderBy(F.col('batch_date').desc())
    dim = (dwd_final.select('bvid', 'up_mid', 'up_name', 'title', 'tname', 'duration',
                            'pubdate', 'batch_date', 'part_count')
           .withColumn('rn', F.row_number().over(wv)).filter('rn = 1').drop('rn'))
    agg = (dwd_final.groupBy('bvid')
           .agg(F.min('batch_date').alias('first_batch'),
                F.max('batch_date').alias('last_batch'),
                F.countDistinct('batch_date').alias('n_batches')))
    dim.join(agg, 'bvid').write.mode('overwrite').parquet(f'{wh}/dim_video')

    # ── 7) dwd_video_metric_snapshot（每 bvid×batch 一行）──
    snap = (dwd_final.groupBy('bvid', 'batch_date')
            .agg(F.max('play').alias('play'), F.max('like').alias('like'),
                 F.max('coin').alias('coin'), F.max('favorite').alias('favorite'),
                 F.max('share').alias('share'), F.max('reply').alias('reply'),
                 F.max('danmaku_reported').alias('danmaku_reported'),
                 F.max('danmaku_got').alias('danmaku_got'),
                 F.max('tname').alias('tname'), F.max('title').alias('title')))
    snap.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dwd_video_metric_snapshot')

    # ── 8) 主键零重复 + 抽样一致率 ──
    pk_dups = dwd_final.groupBy(['batch_date'] + dedup_key).count().filter('count > 1').count()
    sample = dwd_final.sample(0.002, seed=42).limit(100)
    n_sample = sample.count()
    nm = sample.filter(F.col('text_norm') != norm_text(F.col('text'))).count()
    aw = sample.filter(F.col('attr_weight').isNull()).count()
    lbl = sample.filter(F.col('lag_bucket').isNull() & F.col('lag_sec').isNotNull()).count()

    # ── 9) 报告 ──
    report = {
        'job': 'm0_full', 'engine_version': ENGINE_VERSION, 'dedup_key': dedup_key,
        'reconcile': {'csv_rows': csv_rows, 'ods_rows': ods_rows,
                      'csv_minus_ods': csv_rows - ods_rows,
                      'dups_removed': dups_removed, 'dwd_rows': dwd_rows,
                      'equation': f'{csv_rows} = {csv_rows - dups_removed - (csv_rows - ods_rows)}'
                                  f' ({dwd_rows}) + null_offset({csv_rows - ods_rows}) + dups({dups_removed})'},
        'pk_duplicates': pk_dups,
        'sample_check': {'n': n_sample, 'text_norm_mismatch': nm,
                         'attr_weight_null': aw, 'lag_bucket_null_bug': lbl},
        'spam': spam_meta,
    }
    with open(os.path.join(out_dir, 'm0_full_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ok = pk_dups == 0 and n_sample == 100 and nm == 0 and aw == 0 and lbl == 0
    print(('✅ m0_full 完成' if ok else '❌ m0_full 有未达标项') +
          f'：dwd={dwd_rows:,}（去重 -{dups_removed:,}）')
    spark.stop()


if __name__ == '__main__':
    main()
