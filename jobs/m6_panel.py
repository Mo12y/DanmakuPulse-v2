#!/usr/bin/env python3
"""
m6_panel.py — M6 截断面板构建（B4.3；07 §7 v1.4「截断面板」修正 + 08 §9.9）

机制（快照机制）：弹幕采集为全量池快照 ⇒ (视频, day d ≤ 首采日) 的**截断面板**可由单一快照构建：
  特征截至第 d 天（send_time < pubdate+(d+1)d）、标签第 d+1 天弹幕量 —— 历史批次即可跑，
  无需日频复采（跨批次追踪/早采视频 final 标签待日频，本轮不做"×7~14"主张）。
输出：{wh}/ml_panel —— 每 (bvid, day_idx) 一行：
  - 特征：A（元数据）/ B（体积，截至 d）/ C（结构，截至 d）——与 m6_features 同口径（复用其中纯函数）
  - 标签：y_next = log1p(c_{d+1})；log_ret = log1p(c_{d+1}) − log1p(c_d)（T5 留存比 log 化）
  - 参考：y_final_snap = log1p(c[0..K])（仅描述性，非主任务——设计注明 final 标签待日频复采）
口径（07 §7 v1.4 统计纪律）：
  - 双 N（行数 / 唯一 bvid）在一切报告中强制；**有效样本以 bvid 数计**
  - day_idx 范围 = [2, min(K−1, 13)]（v0 上限 13，设计未指定——记录；控制面板规模）
  - 防泄漏断言：特征源 max(day) ≤ d；标签恒在 d+1（严格未来）
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from m6_features import features_one, _f, DAY, TZ8

ENGINE_VERSION = 'b2-v1'
FEAT_COLS = ['n_danmaku', 'dm_per_min', 'rhythm_entropy', 'burst_cv', 'opening_5pct',
             'ending_5pct', 'mid_ratio', 'avg_text_len', 'color_ratio', 'top_mode_ratio',
             'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio',
             'n_events', 'peak_gap_cv', 'sync_score_med', 'herd_mean']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config', default='/opt/work/config/params.yaml')
    ap.add_argument('--min-k', type=int, default=8)
    ap.add_argument('--day-max', type=int, default=13, help='day_idx 上限（v0=13；设计未指定，记录）')
    ap.add_argument('--day-min', type=int, default=2)
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root
    import yaml
    params = yaml.safe_load(open(args.config, encoding='utf-8'))

    b = (SparkSession.builder.appName('m6_panel')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    dim = (spark.read.parquet(f'{wh}/dim_video')
           .select('bvid', 'up_mid', 'tname', 'duration',
                   F.col('pubdate').cast('long').alias('pub'),
                   F.col('first_batch').cast('int').alias('first_batch')))
    dwd_all = (spark.read.parquet(f'{wh}/dwd_danmaku_clean').filter(~F.col('is_spam'))
               .select('bvid', 'batch_date', 'send_time', 'send_date', 'time_offset', 'offset_pct',
                       'duration', 'attr_weight', 'text', 'text_len', 'is_colored', 'is_special_mode'))
    snap = {r['batch_date']: r['tsnap'] for r in
            dwd_all.groupBy('batch_date')
                   .agg(F.max(F.col('send_time').cast('long')).alias('tsnap')).collect()}

    meta_rows, seen = {}, set()
    for r in dim.collect():
        ts = snap.get(r['first_batch'])
        if ts is None or r['pub'] is None or r['bvid'] in seen:
            continue
        if int((ts - r['pub']) // DAY) >= args.min_k:
            seen.add(r['bvid'])
            h = datetime.fromtimestamp(r['pub'], TZ8).hour
            meta_rows[r['bvid']] = {'bvid': r['bvid'], 'batch_date': r['first_batch'],
                                    'pub': r['pub'], 'dur': r['duration'], 'tname': r['tname'],
                                    'up_mid': r['up_mid'],
                                    'pub_hour_bin': 0 if h < 7 else (1 if h < 19 else 2)}
    bvids = list(meta_rows.keys())
    from collections import Counter
    upc = Counter(m['up_mid'] for m in meta_rows.values())

    fbatch = spark.createDataFrame([(m['bvid'], m['batch_date']) for m in meta_rows.values()],
                                   ['bvid', 'fbatch'])
    d = (dwd_all.filter(F.col('bvid').isin(bvids)).join(fbatch, 'bvid')
         .filter(F.col('batch_date') == F.col('fbatch')).collect())
    sent = (spark.read.parquet(f'{wh}/dwd_danmaku_sent')
            .filter(F.col('bvid').isin(bvids)).join(fbatch, 'bvid')
            .filter(F.col('batch_date') == F.col('fbatch'))
            .select('bvid', 'text', 'offset_pct', 'sent_state').collect())
    sent_map = {bv: {} for bv in bvids}
    for r in sent:
        key = (r['text'], round(r['offset_pct'], 6) if r['offset_pct'] is not None else None)
        sent_map[r['bvid']].setdefault(key, r['sent_state'])

    by_video = {}
    for r in d:
        by_video.setdefault(r['bvid'], []).append(r)

    out, asserts = [], {'n_feat_violation': 0, 'n_cd_zero_skip': 0, 'n_rows': 0}
    for bvid in bvids:
        m = meta_rows[bvid]
        pub = m['pub']
        rows_all = [(int(r['send_time']), r) for r in by_video.get(bvid, [])
                    if r['send_time'] is not None]
        K = int((snap[m['batch_date']] - pub) // DAY)
        c_tot = sum(1 for st, _ in rows_all if 0 <= (st - pub) // DAY)
        for dd in range(args.day_min, min(K - 2, args.day_max) + 1):
            # d+1 ≤ K−1：标签日必须完全被快照覆盖（K 日为采集日=部分观测，防部分日标签）
            up = pub + (dd + 1) * DAY
            trunc = [{'st': st, 'to': r['time_offset'], 'off': r['offset_pct'],
                      'dur': r['duration'], 'w': _f(r['attr_weight']),
                      'tl': r['text_len'], 'col': r['is_colored'], 'mode': r['is_special_mode'],
                      'text': r['text'], 'sd': r['send_date']}
                     for st, r in rows_all if pub <= st < up]
            c_d = len(trunc)
            c_d1 = sum(1 for st, r in rows_all if (st - pub) // DAY == dd + 1)
            if c_d < 1:
                asserts['n_cd_zero_skip'] += 1
                continue
            fx = features_one(trunc, pub, sent_map.get(bvid, {}), params)
            if fx['_max_di_src'] > dd:
                asserts['n_feat_violation'] += 1
                continue
            out.append((bvid, dd, m['batch_date'], K, m['tname'], int(m['dur']), m['up_mid'],
                        m['pub_hour_bin'], c_d, c_d1,
                        float(np.log1p(c_d1)), float(np.log1p(c_d1) - np.log1p(c_d)),
                        float(np.log1p(c_tot)),
                        *[fx[c] for c in FEAT_COLS], ENGINE_VERSION))
    cols = (['bvid', 'day_idx', 'batch_date', 'K_real', 'tname', 'duration', 'up_mid',
             'pub_hour_bin', 'c_d', 'c_d1', 'y_next', 'log_ret', 'y_final_snap']
            + FEAT_COLS + ['engine_version'])
    df = spark.createDataFrame(out, cols)
    from pyspark.sql import functions as F2
    upn = {m['bvid']: upc[m['up_mid']] for m in meta_rows.values()}

    @F2.udf('int')
    def upn_of(bv):
        return int(upn.get(bv, 1))
    df = df.withColumn('up_prior_n', upn_of(F2.col('bvid')))
    df.write.mode('overwrite').parquet(f'{wh}/ml_panel')
    n_rows = df.count()
    n_vid = df.select('bvid').distinct().count()
    asserts['n_rows'] = n_rows

    report = {'job': 'm6_panel', 'engine_version': ENGINE_VERSION,
              'N_rows': n_rows, 'N_bvid': n_vid,
              'statistics_discipline': '双 N 强制报告；有效样本以 bvid 计（07 §7 v1.4）',
              'day_range': [args.day_min, args.day_max],
              'rows_per_day': {str(r['day_idx']): r['count'] for r in
                               df.groupBy('day_idx').count().orderBy('day_idx').collect()},
              'asserts': asserts,
              'note': 'day_max=13 为 v0 选择（设计未指定，记录）；y_final_snap 仅描述性'}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm6_panel_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'N_rows': n_rows, 'N_bvid': n_vid, 'asserts': asserts,
                      'rows_per_day': report['rows_per_day']}, ensure_ascii=False, indent=2))
    print(f'✅ m6_panel：{n_rows} 行 / {n_vid} bvid')
    spark.stop()


if __name__ == '__main__':
    main()
