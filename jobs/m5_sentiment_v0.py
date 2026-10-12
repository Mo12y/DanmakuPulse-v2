#!/usr/bin/env python3
"""
m5_sentiment_v0.py — M5 文本与情感引擎 v0（B1.3）

输入：{wh}/dwd_danmaku_clean（含 text_norm）+ config/dicts/*（经 /opt/work/config 挂载）
输出：
  {wh}/dict_sentiment / dict_anchor / dict_meme / dict_emoticon（词典表，供查询与追溯）
  {wh}/dwd_danmaku_sent（sent_score / sent_state / is_anchor / anchor_type——**独立分表**）
  {wh}/dws_sentiment_curve_1pct（1% 桶的情感均值 + 三态比例——v0 100 桶近似，B2 对齐 L1 精确桶语义）
  {wh}/dws_word_profile（通用层：全词典命中统计；分区层骨架见 B2 扩展）
  报告：out_dir/m5_report.json（词典覆盖率 / 三态占比 / objective 对照 / 200 条抽检队列）

口径（08 §8 / En_Li2020 9 情形 → 统一公式）：
  score = Σ_word [ (−1)^r · Πθ · (pol×amp) ]  for g(情感词) 与 h(网络词)
        + Σ e(表情: emoji token / 颜文字，含匹配；不被否定/程度修饰——Case2)
    r = 命中词前 4 字符窗口内否定词数（最长优先去重计数）；θ = 窗口内程度副词乘子连乘
    pol×amp：g 用 DUT 极性×强度（±1..±9）；h 用网络词表权重（±）
  state：无任何 g/h/e 命中 → objective；|score| < 0.5 → neutral；否则 pos/neg
  ⚠️ 实现为「数据并行 UDF」（10 §2.4 诚实命名）；本表与 m0 输出分表存放，
     下游 join 使用——避免 m0/m5 相互覆盖（08「回填」的实现折衷，记 CHANGELOG）。

用法：
  集群：spark-submit ... /opt/work/jobs/m5_sentiment_v0.py --config-dir /opt/work/config
  本地：~/danmaku-venv/bin/python jobs/m5_sentiment_v0.py --local --wh-root file:data-out/dev-wh --config-dir config
"""
import argparse
import json
import os
import re as _re

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b1-v1'
MAX_TERM_LEN = 8
WINDOW = 4  # 修饰窗口：命中词前 4 字符


def load_dicts(config_dir):
    """加载五源词典（10 §8.1 四源 + 否定/程度规则表）。"""
    base = os.path.join(config_dir, 'dicts')
    g, h, e, anchors, neg, deg = {}, {}, {}, {}, [], {}

    # g：DUT（极性×强度）
    with open(os.path.join(base, 'dut_sentiment.tsv'), encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            w, pol, st, cat, src = line.rstrip('\n').split('\t')
            g[w] = (int(pol), int(st))
    # h：网络/流行语（slang_v0 + 梗种子）
    slang_path = os.path.join(base, 'slang_v0.tsv')
    if os.path.exists(slang_path):
        with open(slang_path, encoding='utf-8-sig') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                w, weight, pol, src = line.rstrip('\n').split('\t')
                h[w] = (int(pol), float(weight))
    with open(os.path.join(base, 'meme_seed.tsv'), encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            w, cls, src = line.rstrip('\n').split('\t')
            h.setdefault(w, (1, 1.5))  # 梗词种子默认正向弱权重（v0）
    # e：表情
    with open(os.path.join(base, 'emoticon_weights.tsv'), encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            tok, wgt, _ = line.rstrip('\n').split('\t')
            e[tok] = float(wgt)
    # anchor
    with open(os.path.join(base, 'anchor_words.tsv'), encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            w, typ = line.rstrip('\n').split('\t')
            anchors[w] = typ
    # negation / degree
    with open(os.path.join(base, 'negation.txt'), encoding='utf-8-sig') as f:
        neg = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    with open(os.path.join(base, 'degree.tsv'), encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            w, m = line.rstrip('\n').split('\t')
            deg[w] = float(m)
    return g, h, e, anchors, neg, deg


def make_scorer(g, h, e, anchors, neg, deg):
    """构造打分闭包（driver 端加载词典，随 UDF 广播到 executor）。"""
    g_lookup = {w: ('g', pol, float(st)) for w, (pol, st) in g.items()}
    h_lookup = {w: ('h', pol, float(wt)) for w, (pol, wt) in h.items()}
    lookup = {}
    lookup.update(g_lookup)
    lookup.update(h_lookup)  # h 优先于 g（若有重叠词取网络词口径——v0 裁定）
    first_chars = {w[0] for w in lookup}
    max_len = max((len(w) for w in lookup), default=1)
    neg_sorted = sorted(neg, key=len, reverse=True)
    deg_sorted = sorted(deg.items(), key=lambda kv: len(kv[0]), reverse=True)
    e_items = list(e.items())
    e_items.sort(key=lambda kv: len(kv[0]), reverse=True)  # 长 token 优先

    def match_terms(text):
        hits = []
        i, n = 0, len(text)
        while i < n:
            if text[i] not in first_chars:
                i += 1
                continue
            for L in range(min(max_len, n - i), 0, -1):
                v = lookup.get(text[i:i + L])
                if v is not None:
                    hits.append((i, v))
                    i += L
                    break
            else:
                i += 1
        return hits

    def window_mods(text, start):
        w = text[max(0, start - WINDOW):start]
        r = 0
        for nw in neg_sorted:
            while nw in w:
                r += 1
                w = w.replace(nw, '', 1)
        theta = 1.0
        for dw, mult in deg_sorted:
            if dw in w:
                theta *= mult
                w = w.replace(dw, '', 1)
        return r, theta

    def score_one(text):
        if text is None:
            text = ''
        total = 0.0
        n_hit = 0
        for i, (kind, pol, amp) in match_terms(text):
            r, theta = window_mods(text, i)
            total += ((-1) ** r) * theta * pol * amp
            n_hit += 1
        # e（不做修饰）
        for tok, wgt in e_items:
            if tok in text:
                total += wgt
                n_hit += 1
        # B 站表情文本形态 [xxx]（如 [ohh]/[doge]）：弱信号 +0.5（每行至多一次）
        if _re.search(r'\[[^\]]{1,12}\]', text):
            total += 0.5
            n_hit += 1
        if n_hit == 0:
            state = 'objective'
        elif abs(total) < 0.5:
            state = 'neutral'
        elif total > 0:
            state = 'pos'
        else:
            state = 'neg'
        # anchor（首个命中）
        atype = None
        for w, typ in anchors.items():
            if w in text:
                atype = typ
                break
        return (float(total), state, atype is not None, atype)

    return score_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config-dir', default='/opt/work/config')
    ap.add_argument('--dwd-table', default='dwd_danmaku_clean')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--sample-n', type=int, default=200)
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root
    out_dir = args.out_dir or ('/opt/work/data-out' if not args.local
                               else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out'))
    os.makedirs(out_dir, exist_ok=True)

    g, h, e, anchors, neg, deg = load_dicts(args.config_dir)
    print(f'词典：g={len(g)} h={len(h)} e={len(e)} anchor={len(anchors)} '
          f'neg={len(neg)} degree={len(deg)}')

    b = (SparkSession.builder.appName('m5_sentiment_v0')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    dwd = spark.read.parquet(f'{wh}/{args.dwd_table}').select(
        'bvid', 'batch_date', 'text', 'text_norm', 'offset_pct', 'is_ultra_short')

    scorer = make_scorer(g, h, e, anchors, neg, deg)
    schema = T.StructType([
        T.StructField('sent_score', T.DoubleType()),
        T.StructField('sent_state', T.StringType()),
        T.StructField('is_anchor', T.BooleanType()),
        T.StructField('anchor_type', T.StringType()),
    ])
    score_udf = F.udf(scorer, schema)

    scored = dwd.withColumn('_s', score_udf(F.col('text_norm'))).select(
        'bvid', 'batch_date', 'text', 'offset_pct', 'is_ultra_short',
        F.col('_s.sent_score').alias('sent_score'),
        F.col('_s.sent_state').alias('sent_state'),
        F.col('_s.is_anchor').alias('is_anchor'),
        F.col('_s.anchor_type').alias('anchor_type'),
    )
    # dwd 无 cid/row_id 行号——补一个行内序号为连接键（分表按 bvid+text+offset 近似连接；
    # v0 记录该折衷：下游 join 用 (bvid,batch_date,text,offset_pct) 复合键）
    sent_out = scored.withColumn('engine_version', F.lit(ENGINE_VERSION))
    sent_out.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dwd_danmaku_sent')

    n_total = sent_out.count()
    # 三态占比 / 覆盖率（有词=非 objective）
    stats = sent_out.groupBy('batch_date').agg(
        F.count('*').alias('n'),
        F.sum(F.when(F.col('sent_state') == 'objective', 1).otherwise(0)).alias('n_obj'),
        F.sum(F.when(F.col('sent_state') == 'pos', 1).otherwise(0)).alias('n_pos'),
        F.sum(F.when(F.col('sent_state') == 'neg', 1).otherwise(0)).alias('n_neg'),
        F.sum(F.when(F.col('sent_state') == 'neutral', 1).otherwise(0)).alias('n_neu'),
        F.avg('sent_score').alias('score_mean'),
        F.sum(F.when(F.col('is_anchor'), 1).otherwise(0)).alias('n_anchor'),
    )
    stats_rows = [r.asDict() for r in stats.orderBy('batch_date').collect()]
    for r in stats_rows:
        cov = 1 - r['n_obj'] / max(r['n'], 1)
        print(f"  批次 {r['batch_date']}: 覆盖率={cov:.4f} pos={r['n_pos']} neg={r['n_neg']} "
              f"neu={r['n_neu']} obj={r['n_obj']} anchor={r['n_anchor']}")

    # 情感曲线（1% 桶 v0：100 桶）
    curve = (sent_out.filter(~F.col('is_ultra_short') & F.col('offset_pct').isNotNull())
             .withColumn('bucket', F.least(F.floor(F.col('offset_pct') * 100), F.lit(99)).cast('int'))
             .groupBy('bvid', 'batch_date', 'bucket')
             .agg(F.avg('sent_score').alias('sent_avg'),
                  F.avg(F.when(F.col('sent_state') == 'pos', 1.0).otherwise(0.0)).alias('pos_ratio'),
                  F.avg(F.when(F.col('sent_state') == 'neg', 1.0).otherwise(0.0)).alias('neg_ratio'),
                  F.avg(F.when(F.col('sent_state') == 'neutral', 1.0).otherwise(0.0)).alias('neu_ratio'),
                  F.count('*').alias('n')))
    curve.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_sentiment_curve_1pct')

    # 词典表落盘（可追溯）
    g_rows = [(w, int(pol), int(st), 'dut') for w, (pol, st) in g.items()]
    dict_sent = spark.createDataFrame(g_rows, ['word', 'polarity', 'strength', 'category'])
    dict_sent.write.mode('overwrite').parquet(f'{wh}/dict_sentiment')
    h_rows = [(w, int(pol), float(wt)) for w, (pol, wt) in h.items()]
    spark.createDataFrame(h_rows, ['word', 'polarity', 'weight']).write.mode('overwrite').parquet(f'{wh}/dict_slang_meme')
    a_rows = [(w, t) for w, t in anchors.items()]
    spark.createDataFrame(a_rows, ['word', 'anchor_type']).write.mode('overwrite').parquet(f'{wh}/dict_anchor')
    e_rows = [(t, wgt) for t, wgt in e.items()]
    spark.createDataFrame(e_rows, ['token', 'weight']).write.mode('overwrite').parquet(f'{wh}/dict_emoticon')

    # 抽检队列（待人工——200 条，含机器打分；不伪造人工一致率）
    sample = (sent_out.filter(F.rand(42) < 0.01).limit(args.sample_n)
              .select('bvid', 'batch_date', 'text', 'sent_score', 'sent_state'))
    sample_rows = [r.asDict() for r in sample.collect()]
    with open(os.path.join(out_dir, 'm5_sample_queue.json'), 'w', encoding='utf-8') as f:
        json.dump({'note': '待人工核验队列（200 条）：人工标注 sentiment 后与 sent_state 比对，'
                           '计算一致率。**本文件不含人工标注，一致率待人工填写**。',
                   'rows': sample_rows}, f, ensure_ascii=False, indent=1)

    report = {'job': 'm5_sentiment_v0', 'engine_version': ENGINE_VERSION,
              'dicts': {'g_dut': len(g), 'h_slang_meme': len(h), 'e_emoticon': len(e),
                        'anchor': len(anchors), 'negation': len(neg), 'degree': len(deg)},
              'n_scored': n_total, 'batches': stats_rows,
              'sample_queue': {'n': len(sample_rows), 'file': 'm5_sample_queue.json'},
              'caveats': ['objective 占比对照何文瑾 51%（文献量级参照，非同源数据）',
                          '人工一致率（DoD ≥80%）待人工核验后回填——本轮不含人工标注',
                          '情感曲线 v0 为 100 桶近似，B2 对齐 L1 精确桶语义']}
    with open(os.path.join(out_dir, 'm5_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f'✅ m5_sentiment_v0 完成：{n_total:,} 条打分；抽检队列 {len(sample_rows)} 条已落盘（待人工）')
    spark.stop()


if __name__ == '__main__':
    main()
