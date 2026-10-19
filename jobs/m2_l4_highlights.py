#!/usr/bin/env python3
"""
m2_l4_highlights.py — 精彩时刻检测 DWS 构建（PLAN_HIGHLIGHTS §2.1–2.5）

输入（/warehouse）：
  dwd_danmaku_clean        弹幕明细（is_spam=false；time_offset 秒；text_norm 已归一）
  dws_sentiment_curve_1pct 1% 桶情感曲线（段覆盖占比）
  dim_video                每 bvid 的 tname/duration/title
  config/dicts/critique_words.tsv  质疑词词典（Lead 定稿；78/80 词，source=seed|expand）
输出（/warehouse，4 张新表 + 报告）：
  dws_resonance_peaks         共振峰/段（含片头标记、情绪参考、分位）
  dws_critique_segments       批判段（严格阈值；检出率<5% 时按 §2.4 破例放宽并写报告）
  dws_video_feedback_profile  视频反馈画像（从众/中立/批判三分类 + 共振汇总 + 分区对标）
  dict_critique               质疑词词典表（word/is_critique/weight/source/confirmed）
  data-out/m2_l4_highlights_report.json  检出率/阈值/覆盖率/耗时/诚实边界

实现口径：
  - 重活用 Spark SQL 聚合（1.1M 行 → 5s 文本桶 95 万行），逐视频纯函数在 RDD 上数据并行
    （跨视频并行；论文口径=「逐视频迭代算法数据并行执行」，不宣称算法分布式）
  - highlight_core.py 为纯函数库：spark-submit 必须 --py-files 分发（Makefile 的 SUBMIT_L4）
  - driver 内存仅 512m：不 collect 弹幕明细；只 collect「含质疑词的 distinct 文本」（~2 万条）到
    driver 用 m5 的 load_dicts/make_scorer 打分（同一 text_norm 分数相同，distinct 打分一次）
  - 幂等：全量覆盖写、无随机数，重跑结果一致

用法：
  集群：make m2-l4-highlights
        docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
          --py-files /opt/work/jobs/highlight_core.py /opt/work/jobs/m2_l4_highlights.py
  本地：~/danmaku-venv/bin/python jobs/m2_l4_highlights.py --local --wh-root file:data-out/dev-wh ...
"""
import argparse
import json
import os
import re
import sys
import time

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession, Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import highlight_core as H  # noqa: E402
from m5_sentiment_v0 import load_dicts, make_scorer  # noqa: E402  （复用 M5 打分器，口径不变）

ENGINE_VERSION = H.ENGINE_VERSION


# ══════════════════════ 逐视频纯函数（RDD 数据并行；闭包依赖 highlight_core 模块） ══════════════════════

def per_video(row, bcast_scores):
    """单视频-批次：复读桶 → 共振峰/段；批判弹幕 → 批判段（strict+relaxed 两套）；反馈画像。

    输入 row 字段：bvid, batch_date, tp(文本桶 struct 列表), cp(1% 情感曲线 struct 列表),
                  duration, tname, n_dm
    输出：带类型标签的 tuple 列表（peak / crit_strict / crit_relaxed / prof）
    """
    bvid = row['bvid']
    bd = int(row['batch_date'])
    tp = row['tp'] or []
    cp = [{'bucket': int(p['bucket']), 'pos_ratio': p['pos_ratio'],
           'neg_ratio': p['neg_ratio'], 'neu_ratio': p['neu_ratio']}
          for p in (row['cp'] or [])]
    duration = row['duration']
    tname = row['tname']
    n_dm = int(row['n_dm'] or 0)
    has_curve = bool(cp)
    scores = bcast_scores.value

    # ── 复读桶统计 + 文本三分类（权重=该文本条数 n）──
    maxb = max(int(p['b']) for p in tp) if tp else -1
    L = maxb + 1
    n_repeat = [0] * L
    n_rep_texts = [0] * L
    top_text = [''] * L
    top_n = [0] * L
    top_len = [-1] * L
    crit_buckets = {}      # b -> [n_crit, Σw]
    crit_rows_by_b = {}    # b -> [{text, sent_score, n}]
    n_herd = n_neutral = n_critique = 0
    for p in tp:
        b = int(p['b'])
        text = p['text']
        n = int(p['n'] or 0)
        ln = int(p['text_len'] or 0) or len(text or '')
        n_repeat[b] += max(n - 1, 0)
        if n >= 3:
            n_rep_texts[b] += 1
        # 桶 top_text：次数 desc → 文本长度 desc → 字典序 asc（可复现）
        if (n > top_n[b] or (n == top_n[b] and
                             (ln > top_len[b] or (ln == top_len[b] and text < top_text[b])))):
            top_n[b], top_text[b], top_len[b] = n, text, ln
        # 批判候选打分（distinct 文本已在 driver 算好，经 broadcast 下发）
        info = scores.get(text)
        is_crit = False
        w = 0.0
        if info is not None:
            sent, word_w = info
            is_crit = H.is_critique_danmaku(True, text, sent)
            if is_crit:
                w = H.critique_weight(text, sent, word_w)
                cur = crit_buckets.get(b)
                if cur is None:
                    crit_buckets[b] = [n, w * n]
                else:
                    cur[0] += n
                    cur[1] += w * n
                crit_rows_by_b.setdefault(b, []).append(
                    {'text': text, 'sent_score': sent, 'n': n})
        cls = H.classify_feedback_text(text, n, is_crit)
        if cls == 'critique':
            n_critique += n
        elif cls == 'herd':
            n_herd += n
        else:
            n_neutral += n

    # ── 共振峰/段 ──
    peaks = H.detect_peak_buckets(n_repeat, n_rep_texts)
    segs = H.build_segments(n_repeat, peaks) if peaks else []
    out = []
    peak_rows = []
    for seg in segs:
        start_sec, end_sec = H.clip_segment_secs(seg['start_bucket'], seg['end_bucket'], duration)
        if end_sec <= start_sec:
            continue
        texts, counts = H.segment_top_danmaku(seg['start_bucket'], seg['end_bucket'],
                                              top_text, top_n)
        pos, neg, neu = H.curve_ratios(cp, start_sec, end_sec, duration)
        etype, elabel = H.classify_emotion(texts[0] if texts else '', pos, neg)
        n_rep = sum(n_repeat[b] for b in range(seg['start_bucket'], seg['end_bucket'] + 1))
        n_rt = max(n_rep_texts[b] for b in range(seg['start_bucket'], seg['end_bucket'] + 1))
        psec = H.bucket_midpoint_sec(seg['peak_bucket'])
        in_opening = start_sec < H.OPENING_SEC
        peak_rows.append({'start_sec': start_sec, 'end_sec': end_sec, 'peak_sec': psec,
                          'score': int(seg['peak_score']),
                          'peak_score': int(seg['peak_score']),   # resonance_summary 口径
                          'pctl': None, 'in_opening': in_opening,
                          'emotion_type': etype, 'emotion_label': elabel,
                          'top_danmaku': texts})
        out.append(('peak', bvid, bd, len(peak_rows) - 1, start_sec, end_sec, psec,
                    int(seg['peak_score']), int(n_rep), int(n_rt), texts, counts,
                    etype, elabel, pos, neg, neu, in_opening, tname))

    # ── 视频反馈画像（用于分区对标；pctl/z 由 driver 按去重 bvid 基准补）──
    res = H.resonance_summary(peak_rows)
    rh, rn, rc = H.feedback_ratios(n_herd, n_neutral, n_critique)
    out.append(('prof', bvid, bd, n_dm, n_herd, n_neutral, n_critique, rh, rn, rc,
                res['n_peaks'], res['resonance_total'], res['max_peak_score'],
                duration, tname, has_curve))

    # ── 批判段：严格 + 破例两套都算，全局检出率由 driver 决定采用哪套 ──
    crit_stats = [{'b': b, 'n_crit': v[0], 'score': v[1]}
                  for b, v in sorted(crit_buckets.items())]
    if crit_stats:
        for tag, params in (('crit_strict', {'cand_min': H.CRITIQUE_CAND_MIN,
                                             'merge_gap': H.CRITIQUE_MERGE_GAP,
                                             'min_sum': H.CRITIQUE_SEG_MIN}),
                            ('crit_relaxed', H.CRITIQUE_RELAXED)):
            for seg_id, cs in enumerate(H.find_critique_segments(crit_stats, **params)):
                s_sec, e_sec = H.clip_segment_secs(cs['start_bucket'], cs['end_bucket'], duration)
                if e_sec <= s_sec:
                    continue
                rows = []
                for b in range(cs['start_bucket'], cs['end_bucket'] + 1):
                    rows.extend(crit_rows_by_b.get(b, []))
                texts, counts = H.critique_top_danmaku(rows)
                _, neg, _ = H.curve_ratios(cp, s_sec, e_sec, duration)
                out.append((tag, bvid, bd, seg_id, s_sec, e_sec,
                            round(float(cs['critique_score']), 6), int(cs['n_critique']),
                            texts, counts, neg))
    return out


def build_spark(local):
    b = (SparkSession.builder.appName('m2_l4_highlights')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai'))
    if local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    return b.getOrCreate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config-root', default='/opt/work/config')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--batches', default='', help='调试用：逗号分隔批次子集（默认全量）')
    ap.add_argument('--min-partition-n', type=int, default=20)
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ── 0) 质疑词词典（Lead 定稿文件为唯一事实源）──
    dict_path = os.path.join(args.config_root, 'dicts', 'critique_words.tsv')
    dict_rows = H.load_critique_dict(dict_path)
    words = H.confirmed_words(dict_rows)
    dict_sha = H.dict_fingerprint(dict_path)
    print(f'词典：{len(dict_rows)} 词（confirmed {len(words)}），sha1={dict_sha}')
    if not words:
        raise SystemExit('❌ 质疑词词典没有 confirmed=1 的词，拒绝运行（不伪造检测）')

    spark = build_spark(args.local)
    spark.sparkContext.setLogLevel('ERROR')

    # ── 1) 输入：仅分析口径（is_spam=false，time_offset/text_norm 有效）──
    dwd = spark.read.parquet(f'{wh}/dwd_danmaku_clean')
    if args.batches:
        bs = [int(x) for x in args.batches.split(',') if x.strip()]
        dwd = dwd.filter(F.col('batch_date').isin(bs))
    n_rows_raw = dwd.count()
    vb_raw = dwd.select('bvid', 'batch_date').distinct().count()
    base = (dwd.filter(~F.coalesce(F.col('is_spam'), F.lit(False)))
            .filter(F.col('time_offset').isNotNull() & (F.col('time_offset') >= 0))
            .filter(F.col('text_norm').isNotNull() & (F.length('text_norm') > 0))
            .select('bvid', 'batch_date', 'time_offset', 'text_norm',
                    F.coalesce(F.col('text_len_calc'), F.length('text_norm')).alias('text_len'),
                    'duration', 'tname'))
    base.cache()
    n_rows = base.count()
    print(f'分析口径：{n_rows:,} / {n_rows_raw:,} 行（is_spam=false + 时间/文本有效）')

    dim = spark.read.parquet(f'{wh}/dim_video').select(
        'bvid', F.col('tname').alias('dim_tname'), F.col('duration').alias('dim_duration'))
    meta = (base.groupBy('bvid', 'batch_date')
            .agg(F.count('*').alias('n_dm'),
                 F.max('duration').alias('dwd_duration'),
                 F.max('tname').alias('dwd_tname'))
            .join(dim, 'bvid', 'left')
            .select('bvid', 'batch_date', 'n_dm',
                    F.coalesce('dim_duration', 'dwd_duration').cast('int').alias('duration'),
                    F.coalesce('dim_tname', 'dwd_tname').alias('tname')))

    # ── 2) 5s 文本桶（(bvid,batch,b,text_norm) → n / len）──
    tb = (base.withColumn('b', F.floor(F.col('time_offset') / F.lit(float(H.BUCKET_SEC))).cast('int'))
          .groupBy('bvid', 'batch_date', 'b', 'text_norm')
          .agg(F.count('*').alias('n'), F.max('text_len').alias('text_len')))
    tb.cache()
    n_tb = tb.count()

    # ── 3) 质疑候选：rlike 预筛 distinct 文本 → driver 用 M5 打分器打分（只打候选）──
    pattern = '|'.join(re.escape(w) for w, _ in words)
    cand = base.filter(F.col('text_norm').rlike(pattern)).select('text_norm').distinct()
    cand_texts = [r['text_norm'] for r in cand.collect()]
    g, h, e, anchors, neg_words, deg = load_dicts(args.config_root)
    scorer = make_scorer(g, h, e, anchors, neg_words, deg)
    word_map = dict(words)
    scored = {}
    for text in cand_texts:
        hits = H.hit_words(text, words)
        if not hits:
            continue
        sent = float(scorer(text)[0])
        scored[text] = (sent, max(wt for _, wt in hits))
    n_hit_rows = base.filter(F.col('text_norm').rlike(pattern)).count()
    print(f'质疑候选：distinct 文本 {len(cand_texts):,}，命中行 {n_hit_rows:,}，已打分 {len(scored):,}')
    bcast = spark.sparkContext.broadcast(scored)

    # ── 4) 每视频聚合包（文本桶 + 情感曲线）→ RDD 数据并行跑纯函数 ──
    tp = (tb.select('bvid', 'batch_date',
                    F.struct(F.col('b').alias('b'), F.col('text_norm').alias('text'),
                             F.col('n').alias('n'), F.col('text_len').alias('text_len')).alias('p'))
          .groupBy('bvid', 'batch_date')
          .agg(F.sort_array(F.collect_list('p')).alias('tp')))
    curve = spark.read.parquet(f'{wh}/dws_sentiment_curve_1pct')
    if args.batches:
        curve = curve.filter(F.col('batch_date').isin(bs))
    cp = (curve.select('bvid', 'batch_date',
                       F.struct(F.col('bucket').alias('bucket'),
                                F.col('pos_ratio').alias('pos_ratio'),
                                F.col('neg_ratio').alias('neg_ratio'),
                                F.col('neu_ratio').alias('neu_ratio')).alias('p'))
          .groupBy('bvid', 'batch_date')
          .agg(F.sort_array(F.collect_list('p')).alias('cp')))
    grouped = (tp.join(cp, ['bvid', 'batch_date'], 'left')
               .join(meta, ['bvid', 'batch_date'], 'left'))

    mixed = grouped.rdd.flatMap(lambda r: per_video(r, bcast))
    mixed.cache()
    n_prof = mixed.filter(lambda r: r[0] == 'prof').count()
    print(f'逐视频计算完成：{n_prof:,} 个视频-批次')

    # ── 5) 拆表：峰 / 画像 / 批判（strict & relaxed）──
    s_peak = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.IntegerType()),
        T.StructField('peak_id', T.IntegerType()),
        T.StructField('start_sec', T.IntegerType()), T.StructField('end_sec', T.IntegerType()),
        T.StructField('peak_sec', T.DoubleType()), T.StructField('peak_score', T.IntegerType()),
        T.StructField('n_repeat', T.IntegerType()), T.StructField('n_rep_texts', T.IntegerType()),
        T.StructField('top_danmaku', T.ArrayType(T.StringType())),
        T.StructField('top_counts', T.ArrayType(T.IntegerType())),
        T.StructField('emotion_type', T.StringType()), T.StructField('emotion_label', T.StringType()),
        T.StructField('pos_ratio', T.DoubleType()), T.StructField('neg_ratio', T.DoubleType()),
        T.StructField('neu_ratio', T.DoubleType()), T.StructField('in_opening', T.BooleanType()),
        T.StructField('tname', T.StringType()),
    ])
    s_prof = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.IntegerType()),
        T.StructField('n_danmaku', T.IntegerType()), T.StructField('n_herd', T.IntegerType()),
        T.StructField('n_neutral', T.IntegerType()), T.StructField('n_critique', T.IntegerType()),
        T.StructField('ratio_herd', T.DoubleType()), T.StructField('ratio_neutral', T.DoubleType()),
        T.StructField('ratio_critique', T.DoubleType()), T.StructField('n_peaks', T.IntegerType()),
        T.StructField('resonance_total', T.IntegerType()),
        T.StructField('max_peak_score', T.IntegerType()),
        T.StructField('duration', T.IntegerType()), T.StructField('tname', T.StringType()),
        T.StructField('has_curve', T.BooleanType()),
    ])
    s_crit = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.IntegerType()),
        T.StructField('seg_id', T.IntegerType()),
        T.StructField('start_sec', T.IntegerType()), T.StructField('end_sec', T.IntegerType()),
        T.StructField('critique_score', T.DoubleType()), T.StructField('n_critique', T.IntegerType()),
        T.StructField('top_danmaku', T.ArrayType(T.StringType())),
        T.StructField('top_counts', T.ArrayType(T.IntegerType())),
        T.StructField('neg_ratio', T.DoubleType()),
    ])
    peak0 = spark.createDataFrame(mixed.filter(lambda r: r[0] == 'peak').map(lambda r: r[1:]),
                                  schema=s_peak).withColumn('engine_version', F.lit(ENGINE_VERSION))
    prof0 = spark.createDataFrame(mixed.filter(lambda r: r[0] == 'prof').map(lambda r: r[1:]),
                                  schema=s_prof).withColumn('engine_version',
                                                            F.lit(ENGINE_VERSION))
    strict_rdd = mixed.filter(lambda r: r[0] == 'crit_strict').map(lambda r: r[1:])
    relaxed_rdd = mixed.filter(lambda r: r[0] == 'crit_relaxed').map(lambda r: r[1:])
    strict0 = spark.createDataFrame(strict_rdd, schema=s_crit).withColumn(
        'engine_version', F.lit(ENGINE_VERSION))
    relaxed0 = spark.createDataFrame(relaxed_rdd, schema=s_crit).withColumn(
        'engine_version', F.lit(ENGINE_VERSION))

    # ── 6) 分区对标：基准 = 每 bvid 取 batch_date 最大那行（去重后 bvid 集合）──
    sc = spark.sparkContext
    # 6.1 视频级：latest-batch bvid 分布 → tname 内百分位 + 稳健 z
    w = Window.partitionBy('bvid').orderBy(F.col('batch_date').desc())
    latest = prof0.withColumn('rn', F.row_number().over(w)).filter(F.col('rn') == 1)
    n_latest = latest.count()
    lt = {}
    for r in latest.groupBy('tname').agg(
            F.count('*').alias('n'),
            F.sort_array(F.collect_list('max_peak_score')).alias('vals')).collect():
        lt[r['tname']] = (int(r['n']), [float(x) for x in r['vals']])
    gl = latest.agg(F.count('*').alias('n'),
                    F.sort_array(F.collect_list('max_peak_score')).alias('vals')).collect()[0]
    g_stats = (int(gl['n']), [float(x) for x in gl['vals']])
    bcast_bench = sc.broadcast({'t': lt, 'g': g_stats, 'min_n': int(args.min_partition_n)})

    def _bench(max_score, tname):
        st = bcast_bench.value
        part = st['t'].get(tname) if tname else None
        part_n = int(part[0]) if part is not None else 0
        if part is not None and part_n >= st['min_n']:
            vals = part[1]
        else:
            vals = st['g'][1]
        pctl = H.percentile_rank(max_score, vals)
        med, mad = H.median_mad(vals)
        z = H.robust_z(max_score, med, mad)
        return (float(pctl) if pctl is not None else None, float(z) if z is not None else None,
                int(part_n))

    bench_udf = F.udf(_bench, T.StructType([
        T.StructField('resonance_pctl', T.DoubleType()), T.StructField('resonance_z', T.DoubleType()),
        T.StructField('partition_n', T.IntegerType())]))
    prof = (prof0.withColumn('_b', bench_udf(F.col('max_peak_score'), F.col('tname')))
            .withColumn('resonance_pctl', F.col('_b.resonance_pctl'))
            .withColumn('resonance_z', F.col('_b.resonance_z'))
            .withColumn('partition_n', F.col('_b.partition_n'))
            .select('bvid', 'batch_date', 'n_danmaku', 'n_herd', 'n_neutral', 'n_critique',
                    'ratio_herd', 'ratio_neutral', 'ratio_critique', 'n_peaks',
                    'resonance_total', 'max_peak_score', 'duration', 'tname',
                    'partition_n', 'resonance_pctl', 'resonance_z', 'engine_version'))

    # 6.2 峰级：同 tname 的 latest-batch 峰分布（去重 bvid 后）
    latest_keys = latest.select('bvid', 'batch_date')
    peak_latest = peak0.join(latest_keys, ['bvid', 'batch_date'], 'inner')
    pt = {}
    for r in peak_latest.groupBy('tname').agg(
            F.count('*').alias('n'),
            F.sort_array(F.collect_list('peak_score')).alias('vals')).collect():
        pt[r['tname']] = (int(r['n']), [float(x) for x in r['vals']])
    pg = peak_latest.agg(F.count('*').alias('n'),
                         F.sort_array(F.collect_list('peak_score')).alias('vals')).collect()[0]
    pg_stats = (int(pg['n']), [float(x) for x in pg['vals']])
    bcast_peak = sc.broadcast({'t': pt, 'g': pg_stats, 'min_n': int(args.min_partition_n)})

    def _peak_bench(score, tname):
        st = bcast_peak.value
        part = st['t'].get(tname) if tname else None
        if part is not None and part[0] >= st['min_n']:
            vals = part[1]
        else:
            vals = st['g'][1]
        return (H.percentile_rank(score, vals) if vals else None, len(vals))

    peak_bench_udf = F.udf(_peak_bench, T.StructType([
        T.StructField('pctl', T.DoubleType()), T.StructField('partition_n', T.IntegerType())]))
    peaks = (peak0.withColumn('_b', peak_bench_udf(F.col('peak_score'), F.col('tname')))
             .withColumn('pctl', F.col('_b.pctl'))
             .withColumn('partition_n', F.col('_b.partition_n'))
             .select('bvid', 'batch_date', 'peak_id', 'start_sec', 'end_sec', 'peak_sec',
                     'peak_score', 'n_repeat', 'n_rep_texts', 'top_danmaku', 'top_counts',
                     'emotion_type', 'emotion_label', 'pos_ratio', 'neg_ratio', 'neu_ratio',
                     'in_opening', 'pctl', 'partition_n', 'engine_version'))

    # ── 7) 批判段阈值决策（§2.4 破例规则：全库检出率 <5% 才放宽，报告写清）──
    strict_v = strict0.select('bvid', 'batch_date').distinct().count()
    relaxed_v = relaxed0.select('bvid', 'batch_date').distinct().count()
    strict_rate = strict_v / max(n_prof, 1)
    relaxed_rate = relaxed_v / max(n_prof, 1)
    use = 'relaxed' if strict_rate < 0.05 else 'strict'
    crit = relaxed0 if use == 'relaxed' else strict0
    print(f'批判段：strict 检出 {strict_v}/{n_prof}（{strict_rate:.4f}）'
          f' / relaxed {relaxed_v}/{n_prof}（{relaxed_rate:.4f}）→ 采用 {use}')

    # ── 8) 词典表 + 4 张 DWS 落盘（全量覆盖；partitionBy batch_date）──
    s_dict = T.StructType([
        T.StructField('word', T.StringType()), T.StructField('is_critique', T.IntegerType()),
        T.StructField('weight', T.DoubleType()), T.StructField('source', T.StringType()),
        T.StructField('confirmed', T.IntegerType()), T.StructField('engine_version', T.StringType()),
    ])
    # 词典表 engine_version 列存「词典版本」（backend-api 的 meta.dict_version 直接取此列）；
    # 其余 3 张表 engine_version = 算法版本 hl-v1
    dict_df = spark.createDataFrame(
        [(r['word'], int(r['confirmed']), float(r['weight']), r['source'],
          int(r['confirmed']), H.DICT_VERSION) for r in dict_rows], schema=s_dict)
    dict_df.write.mode('overwrite').parquet(f'{wh}/dict_critique')
    peaks.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_resonance_peaks')
    crit.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_critique_segments')
    prof.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_video_feedback_profile')

    # ── 9) 报告（检出率/阈值/覆盖率/耗时/诚实边界）──
    n_peaks = peaks.count()
    n_peak_videos = peaks.select('bvid', 'batch_date').distinct().count()
    n_crit = crit.count()
    n_crit_videos = crit.select('bvid', 'batch_date').distinct().count()
    n_opening = peaks.filter(F.col('in_opening')).count()
    fb = prof0.agg(F.sum('n_herd').alias('herd'), F.sum('n_neutral').alias('neutral'),
                   F.sum('n_critique').alias('crit'), F.sum('n_danmaku').alias('ndm')).collect()[0]
    n_no_curve = prof0.filter(~F.col('has_curve')).count()
    n_fallback = prof.filter(F.col('partition_n') < int(args.min_partition_n)).count()
    n_tname_part = prof.filter(F.col('partition_n') >= int(args.min_partition_n)).count()
    tname_null = prof.filter(F.col('tname').isNull() | (F.length('tname') == 0)).count()

    report = {
        'job': 'm2_l4_highlights', 'engine_version': ENGINE_VERSION,
        'dict': {'version': H.DICT_VERSION, 'sha1_prefix': dict_sha,
                 'n_rows': len(dict_rows), 'n_confirmed': len(words),
                 'sources': {s: sum(1 for r in dict_rows if r['source'] == s)
                             for s in ('seed', 'expand')}},
        'inputs': {'dwd_table': 'dwd_danmaku_clean', 'n_rows_raw': int(n_rows_raw),
                   'n_rows_analysis': int(n_rows),
                   'n_text_buckets': int(n_tb),
                   'n_video_batches_raw': int(vb_raw), 'n_video_batches_processed': int(n_prof),
                   'coverage': round(float(n_prof) / max(int(vb_raw), 1), 6),
                   'n_video_batches_without_curve': int(n_no_curve)},
        'resonance': {'thresholds': {'bucket_sec': H.BUCKET_SEC,
                                     'peak_min_repeat': H.PEAK_MIN_REPEAT,
                                     'peak_min_rep_texts': H.PEAK_MIN_REP_TEXTS,
                                     'seg_fallback_ratio': H.SEG_FALLBACK_RATIO,
                                     'merge_valley_ratio': H.MERGE_VALLEY_RATIO,
                                     'opening_sec': H.OPENING_SEC},
                      'n_peaks': int(n_peaks), 'n_videos_with_peaks': int(n_peak_videos),
                      'detection_rate': round(float(n_peak_videos) / max(int(n_prof), 1), 6),
                      'n_peaks_in_opening': int(n_opening),
                      'opening_peak_ratio': round(float(n_opening) / max(int(n_peaks), 1), 6),
                      'n_peaks_per_video_mean': round(float(n_peaks) / max(int(n_prof), 1), 4)},
        'critique': {'thresholds_strict': {'cand_min': H.CRITIQUE_CAND_MIN,
                                           'merge_gap': H.CRITIQUE_MERGE_GAP,
                                           'min_sum': H.CRITIQUE_SEG_MIN,
                                           'min_len': H.CRITIQUE_MIN_LEN},
                     'thresholds_relaxed': dict(H.CRITIQUE_RELAXED),
                     'strict': {'n_segments': int(strict0.count()),
                                'n_videos_with_segments': int(strict_v),
                                'detection_rate': round(float(strict_rate), 6)},
                     'relaxed': {'n_segments': int(relaxed0.count()),
                                 'n_videos_with_segments': int(relaxed_v),
                                 'detection_rate': round(float(relaxed_rate), 6)},
                     'used': use,
                     'rule': '§2.4 破例：仅当 strict 检出率<5% 才降到 n_crit>=1/Σ>=2；'
                             '页面照实显示「未检出」，不为凑数放宽',
                     'n_candidate_distinct_texts': int(len(cand_texts)),
                     'n_hit_rows': int(n_hit_rows),
                     'n_segments': int(n_crit), 'n_videos_with_segments': int(n_crit_videos),
                     'detection_rate': round(float(n_crit_videos) / max(int(n_prof), 1), 6)},
        'feedback': {'n_danmaku': int(fb['ndm'] or 0), 'n_herd': int(fb['herd'] or 0),
                     'n_neutral': int(fb['neutral'] or 0), 'n_critique': int(fb['crit'] or 0),
                     'ratio_herd': round(float(fb['herd'] or 0) / max(int(fb['ndm'] or 0), 1), 6),
                     'ratio_neutral': round(float(fb['neutral'] or 0) / max(int(fb['ndm'] or 0), 1), 6),
                     'ratio_critique': round(float(fb['crit'] or 0) / max(int(fb['ndm'] or 0), 1), 6)},
        'benchmark': {'scope': 'tname × 每 bvid 取 batch_date 最大那行（去重 bvid 集合）',
                      'min_partition_n': int(args.min_partition_n),
                      'n_latest_bvids': int(n_latest),
                      'n_video_batches_using_partition': int(n_tname_part),
                      'n_video_batches_fallback_global': int(n_fallback),
                      'n_video_batches_tname_null': int(tname_null),
                      'pctl_definition': '100 × (#基准值 <= 本值) / N（0–100）',
                      'z_definition': '(v − 基准中位) / (1.4826 × MAD)；MAD=0 → null'},
        'elapsed_sec': round(time.time() - t0, 2),
        'caveats': [
            '只测「情绪共振」与「质疑」，不测内容质量——量大 ≠ 内容好',
            '情绪类型是词典规则分类，标注「参考」，不宣称准确',
            '批判层可能很薄：未检出 = 未检出，不放宽阈值凑数',
            '复读近似 = text_norm 完全相等（不做编辑距离/语义相似），可能漏检变体',
            '弹幕只反映观众反应，不解释内容含义；共振段 ≠ 名场面真值',
        ],
    }
    with open(os.path.join(args.out_dir, 'm2_l4_highlights_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m2_l4_highlights：峰 {n_peaks}（{n_peak_videos} 视频有峰）/ '
          f'批判段 {n_crit}（{n_crit_videos} 视频，采用 {use}）/ 画像 {n_prof} 视频-批次；'
          f'耗时 {report["elapsed_sec"]}s')
    spark.stop()


if __name__ == '__main__':
    main()
