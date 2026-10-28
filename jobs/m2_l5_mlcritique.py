#!/usr/bin/env python3
"""
m2_l5_mlcritique.py — 弱监督「质疑」泛化（G2；2026-10-20）

定位（写死，不许手选参数）：
  词典版批判段精度高但召回低（只认 79 个 confirmed 词的子串）。本作业训练一个
  **弱监督**文本分类器，把「质疑语义」从词典表面词泛化出去，产出 method='ml' 的质疑段，
  与 method='dict' 并存（不替换词典版）。前端用虚线 +「可能误报」标注。

弱标签（复用现有判定，不重写）：
  正样本 = 命中 train 词 + len(text_norm) >= 8 + sent_score < 0（= highlight_core.is_critique_danmaku 同一套）
  负样本 = 不含任何 confirmed 质疑词、且未被判批判的弹幕（按分区分层随机抽样，seed=42）
  正负比 = 1:K，K ∈ {2,5,10}，由 5 折 CV 的 F1 决定（**不许手选**）

特征（写死）：
  char 1-2 gram（中文无词边界）→ HashingTF(numFeatures=2^16=65536)，raw count，不做 TF-IDF。

模型（写死）：
  pyspark.ml.classification.LogisticRegression(family='binomial', maxIter=200)
  网格 elasticNetParam ∈ {0.0,0.5,1.0} × regParam ∈ {1e-3,1e-2,1e-1,1e0}

阈值（写死程序）：
  在选定模型上取「对词典命中正样本召回率 ≥ 0.95」的**边界阈值**（分数阈值再高就掉到 0.95 以下），
  由验证集计算并写进报告，不手填。

跨词典泛化（本任务能否成立的关键）：
  79 个 confirmed 词按 9:1 拆 train词/held-out词（seed=42）。只用 train 词命中的正样本训练，
  用 held-out 词命中的正样本测召回。held-out 召回不高于「全判负」零基线（=0）就判失败、如实写报告。

落表：
  dws_critique_segments 里追加 method='ml' 行（dict 行由 m2_l4 写为 method='dict'），
  用与 dict 版**同一套** find_critique_segments 参数聚合。

用法：
  make m2-l5-mlcritique
  docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 \
    --py-files /opt/work/jobs/highlight_core.py /opt/work/jobs/m2_l5_mlcritique.py
  本地：~/danmaku-venv/bin/python jobs/m2_l5_mlcritique.py --local --wh-root file:data-out/dev-wh
"""
import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF
from pyspark.sql import SparkSession, Window

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import highlight_core as H  # noqa: E402
from m5_sentiment_v0 import load_dicts, make_scorer  # noqa: E402

ML_VERSION = 'ml-critique-v1'
SEED = 42
NUM_FEATURES = 2 ** 16          # 65536，固定
N_FOLDS = 5
K_CANDIDATES = (2, 5, 10)
ELASTIC_CANDIDATES = (0.0, 0.5, 1.0)
REG_CANDIDATES = (1e-3, 1e-2, 1e-1, 1e0)
MAX_ITER = 200
RECALL_TARGET = 0.95
HOLDOUT_RATIO = 0.1             # 词典 9:1 拆 train / held-out


# ── char 1-2 gram（中文无词边界；纯 Python，无新依赖）──
def char_grams(text):
    t = text or ''
    if not t:
        return []
    out = list(t)
    out.extend(t[i:i + 2] for i in range(len(t) - 1))
    return out


def build_spark(local):
    b = (SparkSession.builder.appName('m2_l5_mlcritique')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.session.timeZone', 'Asia/Shanghai'))
    if local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    return b.getOrCreate()


def f1_from_counts(tp, fp, fn):
    if tp <= 0:
        return 0.0
    return 2.0 * tp / (2.0 * tp + fp + fn)


def eval_f1(model, df):
    """混淆矩阵算 F1（正负不平衡，禁用 accuracy）。"""
    p = model.transform(df)
    cnt = p.groupBy('label', 'prediction').count().collect()
    tp = fp = fn = 0
    for r in cnt:
        lab, pred, n = float(r['label']), float(r['prediction']), int(r['count'])
        if lab == 1.0 and pred == 1.0:
            tp = n
        elif lab == 0.0 and pred == 1.0:
            fp = n
        elif lab == 1.0 and pred == 0.0:
            fn = n
    return f1_from_counts(tp, fp, fn)


def split_words(confirmed, seed=SEED, ratio=HOLDOUT_RATIO):
    """79 个 confirmed 词按 9:1 拆 train / held-out（seed 固定，可复现）。"""
    words = sorted({w for w, _ in confirmed})
    rnd = random.Random(seed)
    sh = words[:]
    rnd.shuffle(sh)
    n_held = max(1, int(round(len(sh) * ratio)))
    return sorted(sh[n_held:]), sorted(sh[:n_held])


def hist_params(report_path, default):
    """dict 版实际用的段参数（同一套参数聚合 ml 段；读不到就用 strict 默认）。"""
    try:
        with open(report_path, encoding='utf-8') as f:
            r = json.load(f)
        used = (r.get('critique') or {}).get('used')
        key = 'thresholds_relaxed' if used == 'relaxed' else 'thresholds_strict'
        t = (r.get('critique') or {}).get(key) or {}
        return {'cand_min': int(t.get('cand_min', default['cand_min'])),
                'merge_gap': int(t.get('merge_gap', default['merge_gap'])),
                'min_sum': int(t.get('min_sum', default['min_sum']))}, used
    except (OSError, ValueError):
        return default, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config-root', default='/opt/work/config')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--batches', default='')
    ap.add_argument('--local', action='store_true')
    ap.add_argument('--explore', action='store_true', help='只统计样本量后退出（不训练）')
    ap.add_argument('--smoke', action='store_true', help='只跑 K=2 的第 1 折（验证管线与耗时）')
    args = ap.parse_args()
    wh = args.wh_root
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    default_params = {'cand_min': H.CRITIQUE_CAND_MIN, 'merge_gap': H.CRITIQUE_MERGE_GAP,
                      'min_sum': H.CRITIQUE_SEG_MIN}
    seg_params, used_params_of = hist_params(
        os.path.join(args.out_dir, 'm2_l4_highlights_report.json'), default_params)

    # ── 0) 词典：confirmed 词 9:1 拆 train / held-out ──
    dict_path = os.path.join(args.config_root, 'dicts', 'critique_words.tsv')
    dict_rows = H.load_critique_dict(dict_path)
    confirmed = H.confirmed_words(dict_rows)
    train_words, held_words = split_words(confirmed)
    print(f'词典：confirmed {len(confirmed)} 词 → train {len(train_words)} / held-out {len(held_words)}',
          flush=True)

    spark = build_spark(args.local)
    spark.sparkContext.setLogLevel('ERROR')

    # ── 1) 输入：分析口径（is_spam=false + 时间/文本有效）──
    dwd = spark.read.parquet(f'{wh}/dwd_danmaku_clean')
    if args.batches:
        bs = [int(x) for x in args.batches.split(',') if x.strip()]
        dwd = dwd.filter(F.col('batch_date').isin(bs))
    base = (dwd.filter(~F.coalesce(F.col('is_spam'), F.lit(False)))
            .filter(F.col('time_offset').isNotNull() & (F.col('time_offset') >= 0))
            .filter(F.col('text_norm').isNotNull() & (F.length('text_norm') > 0))
            .select('bvid', 'batch_date', 'time_offset',
                    F.col('text_norm').alias('text_norm'),
                    F.coalesce(F.col('tname'), F.lit('未知')).alias('tname'),
                    F.length('text_norm').alias('len')))
    n_rows = base.count()

    # distinct 文本 + 每个文本归一个分区（取字典序最小，保证可复现）
    meta = (base.groupBy('text_norm')
            .agg(F.min('tname').alias('tname'), F.max('len').alias('len')))
    n_texts = meta.count()

    pat_all = '|'.join(re.escape(w) for w in train_words + held_words)
    pat_train = '|'.join(re.escape(w) for w in train_words)
    pat_held = '|'.join(re.escape(w) for w in held_words)

    # ── 2) 候选文本打分（只打含质疑词的 distinct 文本，与 m2_l4 同口径）──
    cand = meta.filter(F.col('text_norm').rlike(pat_all))
    cand_rows = cand.collect()
    g, h, e, anchors, neg_words, deg = load_dicts(args.config_root)
    scorer = make_scorer(g, h, e, anchors, neg_words, deg)

    pos_train, pos_held = [], []
    hit_train_words = [w for w in train_words]
    hit_held_words = [w for w in held_words]
    for r in cand_rows:
        text = r['text_norm']
        tname = r['tname'] or '未知'
        has_train = any(w in text for w in hit_train_words)
        if has_train:
            if len(text) < H.CRITIQUE_MIN_LEN:
                continue
            if float(scorer(text)[0]) < 0:
                pos_train.append((text, tname))
            continue
        # 只被 held-out 词命中的文本才进泛化评估集（见过的词不算泛化）
        if any(w in text for w in hit_held_words):
            if len(text) < H.CRITIQUE_MIN_LEN:
                continue
            if float(scorer(text)[0]) < 0:
                pos_held.append((text, tname))
    n_pos, n_held_pos = len(pos_train), len(pos_held)
    print(f'分析口径 {n_rows:,} 行 / distinct 文本 {n_texts:,}；含质疑词 {len(cand_rows):,}；'
          f'正样本(train词) {n_pos:,}；held-out 正样本 {n_held_pos:,}', flush=True)
    if args.explore:
        spark.stop()
        return

    # ── 3) 负样本池：不含任何 confirmed 词的 distinct 文本 ──
    neg_pool = meta.filter(~F.col('text_norm').rlike(pat_all))
    avail = {r['tname']: int(r['count']) for r in neg_pool.groupBy('tname').count().collect()}
    # 按分区分层配额 = K × 该分区正样本数；不足则从其它分区补足到 K×n_pos
    pos_by_t = {}
    for _, t in pos_train:
        pos_by_t[t] = pos_by_t.get(t, 0) + 1

    def quotas_for(k):
        target = k * n_pos
        q = {}
        for t, p in pos_by_t.items():
            take = min(avail.get(t, 0), k * p)
            if take > 0:
                q[t] = take
        short = target - sum(q.values())
        if short > 0:                       # 小分区不够就按剩余容量补，仍然是随机抽
            for t in sorted(avail):
                if short <= 0:
                    break
                spare = avail[t] - q.get(t, 0)
                if spare <= 0:
                    continue
                add = min(spare, short)
                q[t] = q.get(t, 0) + add
                short -= add
        return q

    grams_udf = F.udf(char_grams, T.ArrayType(T.StringType()))
    prob_udf = F.udf(lambda v: float(v[1]), T.DoubleType())
    htf = HashingTF(numFeatures=NUM_FEATURES, inputCol='grams', outputCol='features')

    def featureize(df):
        """text_norm → char 1-2 gram → HashingTF(features)。手工 transform 前必须走这一步。"""
        return htf.transform(df.withColumn('grams', grams_udf('text_norm')))

    def build_dataset(k):
        quotas = quotas_for(k)
        qdf = spark.createDataFrame([(t, int(v)) for t, v in quotas.items()],
                                    'tname string, quota int')
        neg_s = (neg_pool.withColumn('r', F.rand(SEED))
                 .withColumn('rn', F.row_number().over(
                     Window.partitionBy('tname').orderBy('r')))
                 .join(qdf, 'tname')
                 .filter(F.col('rn') <= F.col('quota'))
                 .select('text_norm', 'tname'))
        neg_df = (neg_s.withColumn('label', F.lit(0.0))
                  .withColumn('fold', F.pmod(F.xxhash64('text_norm'), N_FOLDS).cast('int')))
        # 正样本折号：打乱后轮转（分层，五折人数差 ≤1）
        idx = list(range(n_pos))
        random.Random(SEED).shuffle(idx)
        pos_df = spark.createDataFrame(
            [(pos_train[i][0], pos_train[i][1], 1.0, int(j % N_FOLDS))
             for j, i in enumerate(idx)],
            'text_norm string, tname string, label double, fold int')
        ds = pos_df.unionByName(neg_df)
        return featureize(ds).select('label', 'fold', 'features').cache()

    # ── 4) 5 折 CV × 网格（F1 均值最高者胜；并列按 K、elastic、reg 升序，可复现）──
    ks = (2,) if args.smoke else K_CANDIDATES
    folds = (0,) if args.smoke else tuple(range(N_FOLDS))
    cv_path = os.path.join(args.out_dir, 'm2_l5_cv.json')
    results = []
    try:
        with open(cv_path, encoding='utf-8') as f:
            results = json.load(f)
        print(f'CV 缓存：已有 {len(results)} 个组合（{cv_path}）', flush=True)
    except (OSError, ValueError):
        results = []
    done_keys = {(r['K'], r['elasticNetParam'], r['regParam']) for r in results}
    for k in ks:
        ds = build_dataset(k)
        n_ds = ds.count()
        print(f'[K={k}] 训练集 {n_ds:,} 行（正 {n_pos:,} / 负 {n_ds - n_pos:,}）', flush=True)
        for el in ELASTIC_CANDIDATES:
            for reg in REG_CANDIDATES:
                if (k, el, reg) in done_keys:
                    continue
                f1s = []
                for f in folds:
                    tr = ds.filter(F.col('fold') != f)
                    va = ds.filter(F.col('fold') == f)
                    lr = LogisticRegression(featuresCol='features', labelCol='label',
                                            family='binomial', maxIter=MAX_ITER,
                                            elasticNetParam=el, regParam=reg)
                    m = lr.fit(tr)
                    f1s.append(eval_f1(m, va))
                mean = sum(f1s) / len(f1s)
                sd = statistics.pstdev(f1s) if len(f1s) > 1 else 0.0
                results.append({'K': k, 'elasticNetParam': el, 'regParam': reg,
                                'f1_mean': round(mean, 6), 'f1_std': round(sd, 6),
                                'f1_folds': [round(x, 6) for x in f1s]})
                print(f'  K={k} el={el} reg={reg}: F1={mean:.4f}±{sd:.4f} '
                      f'({time.time() - t0:.0f}s)', flush=True)
                with open(cv_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=1)
        ds.unpersist()

    best = sorted(results, key=lambda r: (-r['f1_mean'], r['K'],
                                          r['elasticNetParam'], r['regParam']))[0]
    print(f'选定：{best}', flush=True)

    # ── 5) 用选定配置在全量训练集重训；算阈值与跨词典召回 ──
    ds = build_dataset(best['K'])
    n_ds = ds.count()
    lr = LogisticRegression(featuresCol='features', labelCol='label', family='binomial',
                            maxIter=MAX_ITER, elasticNetParam=best['elasticNetParam'],
                            regParam=best['regParam'])
    model = lr.fit(ds)

    # 阈值必须由**验证集**算：用 5 折 out-of-fold 正样本概率（训练集内概率会偏高，把阈值顶上去）
    oof = []
    for f in range(N_FOLDS):
        mf = LogisticRegression(featuresCol='features', labelCol='label', family='binomial',
                                maxIter=MAX_ITER, elasticNetParam=best['elasticNetParam'],
                                regParam=best['regParam']).fit(ds.filter(F.col('fold') != f))
        got = (mf.transform(ds.filter((F.col('fold') == f) & (F.col('label') == 1.0)))
               .select(prob_udf('probability').alias('p')).collect())
        oof.extend(float(r['p']) for r in got)
    probs = sorted(oof, reverse=True)
    if not probs:
        raise SystemExit('❌ 没有词典命中正样本，拒绝训练（不伪造检测）')
    k_keep = max(1, int(math.ceil(RECALL_TARGET * len(probs))))
    threshold = probs[min(k_keep, len(probs)) - 1]
    threshold = min(max(threshold, 1e-6), 0.999999)
    recall_dict = sum(1 for p in probs if p >= threshold) / float(len(probs))

    held_recall, held_stats = None, None
    if pos_held:
        held_df = spark.createDataFrame([(t,) for t, _ in pos_held], 'text_norm string')
        hp = [float(r['p']) for r in (model.transform(featureize(held_df))
                                      .select(prob_udf('probability').alias('p')).collect())]
        held_recall = sum(1 for p in hp if p >= threshold) / float(len(hp))
        # 诊断：操作阈值（0.95 召回边界）很严；0.5 处的召回与负样本分布一起报，才读得出"学到没有"
        neg_probe = [float(r['p']) for r in (
            model.transform(featureize(neg_pool.select('text_norm').limit(2000)))
            .select(prob_udf('probability').alias('p')).collect())]
        hp_sorted = sorted(hp)
        npos_probe = [p for p in hp if p >= 0.5]
        nneg_probe = [p for p in neg_probe if p >= 0.5]
        held_stats = {
            'n': len(hp),
            'mean_prob': round(sum(hp) / len(hp), 6),
            'median_prob': round(hp_sorted[len(hp_sorted) // 2], 6),
            'recall_at_0.5': round(len(npos_probe) / float(len(hp)), 6),
            'neg_probe_n': len(neg_probe),
            'neg_mean_prob': round(sum(neg_probe) / max(len(neg_probe), 1), 6),
            'neg_fpr_at_0.5': round(len(nneg_probe) / max(len(neg_probe), 1), 6),
        }
    print(f'阈值 {threshold:.6f}｜词典召回 {recall_dict:.4f}｜held-out 召回 {held_recall}｜'
          f'诊断 {held_stats}', flush=True)

    # ── 6) 全库打分 → is_critique_ml → 5s 桶聚合 → 质疑段（与 dict 同参数）──
    pred = (model.transform(featureize(meta.select('text_norm')))
            .select('text_norm', prob_udf('probability').alias('ml_score')))
    pred.cache()
    n_ml_texts = pred.filter(F.col('ml_score') >= threshold).count()

    tb = (base.withColumn('b', F.floor(F.col('time_offset') / F.lit(float(H.BUCKET_SEC)))
                          .cast('int'))
          .groupBy('bvid', 'batch_date', 'b', 'text_norm')
          .agg(F.count('*').alias('n')))
    hits = (tb.join(pred, 'text_norm')
            .filter(F.col('ml_score') >= threshold))
    buck = (hits.groupBy('bvid', 'batch_date', 'b')
            .agg(F.sum('n').alias('n_ml'),
                 F.sum(F.col('ml_score') * F.col('n')).alias('s_ml'),
                 F.max_by('text_norm', 'n').alias('top_text'),
                 F.max('n').alias('top_n')))
    prof = (spark.read.parquet(f'{wh}/dws_video_feedback_profile')
            .select('bvid', 'batch_date', 'duration'))
    buck = buck.join(prof, ['bvid', 'batch_date'], 'left')

    def per_video(row):
        bvid, bd = row[0], row[1]
        duration = row[2]
        items = row[3]
        stats = [{'b': int(b), 'n_crit': int(n), 'score': float(s)}
                 for b, n, s, _t, _tn in items]
        segs = H.find_critique_segments(stats, **seg_params) if stats else []
        out = []
        by_b = {int(b): (t, int(tn)) for b, _n, _s, t, tn in items}
        for seg_id, cs in enumerate(segs):
            s_sec, e_sec = H.clip_segment_secs(cs['start_bucket'], cs['end_bucket'], duration)
            if e_sec <= s_sec:
                continue
            inner = [(b, by_b[b][1], by_b[b][0]) for b in range(cs['start_bucket'],
                                                               cs['end_bucket'] + 1) if b in by_b]
            inner.sort(key=lambda x: (-x[1], x[0]))
            texts = [t for _b, _n, t in inner[:H.CRITIQUE_TOP_DANMAKU]]
            counts = [n for _b, n, _t in inner[:H.CRITIQUE_TOP_DANMAKU]]
            seg_n = int(cs['n_critique'])
            out.append((bvid, bd, seg_id, s_sec, e_sec, round(float(cs['critique_score']), 6),
                        seg_n, texts, counts, None,
                        float(cs['critique_score']) / max(seg_n, 1)))
        return out

    ml_rows = (buck.rdd
               .map(lambda r: ((r['bvid'], r['batch_date'], r['duration']),
                               [(r['b'], r['n_ml'], r['s_ml'], r['top_text'], r['top_n'])]))
               .reduceByKey(lambda a, b: a + b)
               .flatMap(lambda kv: per_video((kv[0][0], kv[0][1], kv[0][2], kv[1])))
               .collect())
    print(f'ml 质疑段：{len(ml_rows):,}', flush=True)

    s_crit = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.IntegerType()),
        T.StructField('seg_id', T.IntegerType()),
        T.StructField('start_sec', T.IntegerType()), T.StructField('end_sec', T.IntegerType()),
        T.StructField('critique_score', T.DoubleType()), T.StructField('n_critique', T.IntegerType()),
        T.StructField('top_danmaku', T.ArrayType(T.StringType())),
        T.StructField('top_counts', T.ArrayType(T.IntegerType())),
        T.StructField('neg_ratio', T.DoubleType()),
        T.StructField('ml_score', T.DoubleType()),
    ])
    ml_df = (spark.createDataFrame(ml_rows, schema=s_crit)
             .withColumn('method', F.lit('ml'))
             .withColumn('engine_version', F.lit(ML_VERSION)))

    # ── 7) 跨词典检验决定是否入生产表 ──
    # 规格：held-out 召回不高于零基线（全判负）=0 → 判失败、**停用**，不硬上。
    enabled = bool(held_recall is not None and held_recall > 0.0)
    status = 'passed' if enabled else 'failed'
    path = f'{wh}/dws_critique_segments'
    if enabled:
        # 与 dict 行合并（dict 行由 m2_l4 写为 method='dict'）
        old = spark.read.parquet(path)
        if 'method' not in old.columns:
            old = old.withColumn('method', F.lit('dict'))
        if 'ml_score' not in old.columns:
            old = old.withColumn('ml_score', F.lit(None).cast('double'))
        cols = old.columns
        combined = old.select(*cols).unionByName(ml_df.select(*cols)).cache()
        n_total = combined.count()
        n_dict = combined.filter(F.col('method') == 'dict').count()
        n_ml = combined.filter(F.col('method') == 'ml').count()
        combined.write.mode('overwrite').partitionBy('batch_date').parquet(path)
    else:
        # 停用：ml 段只留在旁路路径（证据可追溯），不进 dws_critique_segments / PG / 页面
        rej = f'{wh}/dws_critique_segments_ml_rejected'
        ml_df.write.mode('overwrite').partitionBy('batch_date').parquet(rej)
        n_ml = ml_df.count()
        row = spark.read.parquet(path)
        n_dict = row.filter(F.col('method') == 'dict').count() if 'method' in row.columns \
            else row.count()
        n_total = n_dict
        print(f'⛔ 跨词典 held-out 召回 {held_recall} ≤ 零基线 → 判失败，ML 层停用；'
              f'{n_ml} 段只写入旁路 {rej}（不进生产表）', flush=True)

    # ── 8) 报告（全部字段是算出来的，不是手填；保证可复现）──
    report = {
        'job': 'm2_l5_mlcritique',
        'ml_version': ML_VERSION,
        'seed': SEED,
        'dict': {'confirmed': len(confirmed), 'train_words': len(train_words),
                 'heldout_words': len(held_words),
                 'heldout_list': held_words},
        'n_pos': int(n_pos),
        'n_neg': int(n_ds - n_pos),
        'K': int(best['K']),
        'numFeatures': NUM_FEATURES,
        'elasticNetParam': best['elasticNetParam'],
        'regParam': best['regParam'],
        'f1_mean': best['f1_mean'],
        'f1_std': best['f1_std'],
        'f1_folds': best['f1_folds'],
        'cv': {'folds': len(folds), 'grid': len(results),
               'criterion': 'F1（正负不平衡，禁用 accuracy）',
               'selection': 'F1 均值最高；并列按 K / elasticNet / regParam 升序'},
        'threshold': round(float(threshold), 6),
        'recall_dict_pos': round(float(recall_dict), 6),
        'recall_heldout': (None if held_recall is None else round(float(held_recall), 6)),
        'heldout_diag': held_stats,
        'recall_zero_baseline': 0.0,
        'heldout_n_pos': int(n_held_pos),
        'seg_params': seg_params,
        'seg_params_source': ('m2_l4_highlights_report.json:used=%s' % used_params_of
                              if used_params_of else 'strict 默认'),
        'status': status,
        'enabled': bool(enabled),
        'disabled_reason': (None if enabled else
                            '跨词典 held-out 召回 %.4f ≤ 零基线 0（模型背下了词典表面词，'
                            '没学到质疑语义）→ 按规格停用，不写入生产表'
                            % (held_recall or 0.0)),
        'n_ml_segments': int(n_ml),
        'n_dict_segments': int(n_dict),
        'n_segments_total': int(n_total),
        'n_ml_texts': int(n_ml_texts),
        'n_analysis_rows': int(n_rows),
        'n_distinct_texts': int(n_texts),
        'caveat': ['ML 版只做召回增强（⊇ 词典版），不替换词典版；精度未人工抽检，前端用虚线 +「可能误报」；'
                   '跨词典泛化不过关就停用，不硬上。'],
        'elapsed_sec': round(time.time() - t0, 2),
    }
    with open(os.path.join(args.out_dir, 'm2_l5_mlcritique_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f'✅ m2_l5_mlcritique：ml 段 {n_ml} / dict 段 {n_dict}；'
          f'held-out 召回 {held_recall}；耗时 {report["elapsed_sec"]}s', flush=True)
    spark.stop()


if __name__ == '__main__':
    main()
