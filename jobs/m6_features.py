#!/usr/bin/env python3
"""
m6_features.py — M6 特征矩阵（预注册 14 号 §3：0..3 日窗截断口径）

输入：dwd_danmaku_clean（~spam）+ dwd_danmaku_sent + dim_video + l2_common/m2_l2_events 纯函数
输出：{wh}/ml_features —— 每 bvid 一行（A/B/C 特征 + 标签 + 元数据；provisional 池）

口径（预注册 §3/§4，防泄漏是硬约束）：
  - 池：K_real = floor((T_snap − pubdate)/86400) ≥ 8 且 c[0..3] ≥ 1；T_snap = 该批 max(send_time)
    ⚠️ batch_date ≠ 快照时间（0902 批实采 ≈9/12）——不得用 batch_date 推算 K
  - **截断**：特征只用 pubdate ≤ send_time < pubdate+4d 的弹幕（断言①：特征源 di ≤ 3）
  - 只用**首采批快照**的行（bvid 若跨批重采，其他批的行不得混入）
  - 「首采互动」不进特征（集合时点 day K ≫ 标签窗，时间泄漏——预注册 §3.1-2）；D 层排除（§3.3）
  - 标签：y_growth_v2 = log1p(c[0..7]) − log1p(c[0..3])（1% 缩尾）+ 对照 log1p(c[0..7])
实现：driver 侧逐视频（容器无 pandas，纯 numpy）；91 量级无需 RDD。
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

import l2_common          # noqa: F401  （分发依赖，m2_l2_events 闭包用）
import m2_l2_events as m2e

ENGINE_VERSION = 'b2-v1'
TZ8 = timezone(timedelta(hours=8))
DAY = 86400.0


def _f(x, default=0.0):
    """安全转 float（dwd 部分数值列以 string 存——2026-10-13 实测踩坑）"""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def entropy_norm(vals, n_bins):
    c = np.asarray([x for x in vals if x > 0], dtype=float)
    if c.size == 0 or n_bins <= 1:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(n_bins))


def features_one(rows, pub, sent_map, params):
    """单视频截断特征（rows：截断集，每行 dict）。"""
    n = len(rows)
    dur = _f(rows[0]['dur']) if rows else 0.0
    offs = np.array([_f(r['off']) for r in rows if r['off'] is not None], dtype=float)
    opening = float(np.mean(offs < 0.05)) if offs.size else 0.0
    ending = float(np.mean(offs > 0.95)) if offs.size else 0.0
    mid = float(np.mean((offs >= 0.05) & (offs <= 0.95))) if offs.size else 0.0
    b1 = np.clip(np.floor(offs * 100).astype(int), 0, 99)
    r_ent = entropy_norm(np.bincount(b1, minlength=100), 100)
    to = np.array([_f(r['to']) for r in rows if r['to'] is not None], dtype=float)
    if to.size:
        b5 = (to // 5).astype(int)
        arr = np.zeros(int(b5.max()) + 1)
        np.add.at(arr, b5, 1.0)
        burst_cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else 0.0
    else:
        burst_cv = 0.0
    avg_len = float(np.mean([_f(r['tl']) for r in rows])) if rows else 0.0
    color = float(np.mean([1.0 if r['col'] else 0.0 for r in rows])) if rows else 0.0
    topmode = float(np.mean([1.0 if r['mode'] else 0.0 for r in rows])) if rows else 0.0
    st_c = {'pos': 0, 'neg': 0, 'neutral': 0, 'objective': 0, 'missing': 0}
    for r in rows:
        s = sent_map.get((r['text'], round(r['off'], 6))) if r['off'] is not None else None
        st_c[s if s in st_c else 'missing'] += 1
    pos, neg, neu, obj = st_c['pos'], st_c['neg'], st_c['neutral'], st_c['objective']
    hit = pos + neg + neu
    pos_r = pos / hit if hit else 0.0
    neg_r = neg / hit if hit else 0.0
    neu_r = neu / hit if hit else 0.0
    no_hit_r = obj / n if n else 0.0
    d = params['detect']
    if to.size:
        wser = np.zeros(int(to.max() // 5) + 1)
        np.add.at(wser, (to // 5).astype(int),
                  np.array([_f(r['w']) for r in rows if r['to'] is not None]))
    else:
        wser = np.zeros(0)
    events = []
    if wser.size >= 3:
        try:
            events = m2e.run_detectors(wser, params)
        except Exception:
            events = []
    n_events = len(events)
    if n_events >= 2:
        starts = np.array([e['start'] for e in events], dtype=float)
        gaps = np.diff(np.sort(starts))
        peak_gap_cv = float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else 0.0
    else:
        peak_gap_cv = 0.0
    win = int(params.get('sync', {}).get('peak_win', 2))
    b_all = (to // 5).astype(int) if to.size else np.array([], dtype=int)
    B = int(wser.size)
    dates = [r['sd'] for r in rows]
    D = len(set(dates))
    syncs = []
    for e in events:
        m = (b_all >= e['start'] - win) & (b_all <= e['end'] + win)
        if m.sum() == 0:
            continue
        h_pos = entropy_norm(np.bincount(b_all[m], minlength=B), B)
        dd = {}
        for k in np.where(m)[0]:
            dd[dates[k]] = dd.get(dates[k], 0) + 1
        h_time = entropy_norm(list(dd.values()), max(D, 2))
        syncs.append((1.0 - h_pos) * h_time)
    sync_med = float(np.median(syncs)) if syncs else 0.0
    cnt_txt, seen, ws = {}, {}, []
    for r in rows:
        cnt_txt[r['text']] = cnt_txt.get(r['text'], 0) + 1
    for r in rows:
        seen[r['text']] = seen.get(r['text'], 0) + 1
        ws.append(seen[r['text']] / cnt_txt[r['text']])
    herd = float(np.mean(ws)) if ws else 0.0
    max_di_src = int(max(((r['st'] - pub) // DAY) for r in rows)) if rows else -1
    return {
        'n_danmaku': n, 'dm_per_min': n / (dur / 60.0) if dur > 0 else 0.0,
        'rhythm_entropy': r_ent, 'burst_cv': burst_cv,
        'opening_5pct': opening, 'ending_5pct': ending, 'mid_ratio': mid,
        'avg_text_len': avg_len, 'color_ratio': color, 'top_mode_ratio': topmode,
        'pos_ratio': pos_r, 'neg_ratio': neg_r, 'neu_ratio': neu_r, 'no_hit_ratio': no_hit_r,
        'n_events': n_events, 'peak_gap_cv': peak_gap_cv,
        'sync_score_med': sync_med, 'herd_mean': herd,
        '_max_di_src': max_di_src,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config', default='/opt/work/config/params.yaml')
    ap.add_argument('--min-k', type=int, default=8)
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    import yaml
    params = yaml.safe_load(open(args.config, encoding='utf-8'))

    b = (SparkSession.builder.appName('m6_features')
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

    # 池：T_snap（batch 级 max(send_time)）→ K_real ≥ min_k
    snap = {r['batch_date']: r['tsnap'] for r in
            dwd_all.groupBy('batch_date')
                   .agg(F.max(F.col('send_time').cast('long')).alias('tsnap')).collect()}
    meta_rows, seen_bv = {}, set()
    for r in dim.collect():
        ts = snap.get(r['first_batch'])
        if ts is None or r['pub'] is None or r['bvid'] in seen_bv:
            continue
        K = int((ts - r['pub']) // DAY)
        if K >= args.min_k:
            seen_bv.add(r['bvid'])
            h = datetime.fromtimestamp(r['pub'], TZ8).hour
            meta_rows[r['bvid']] = {'bvid': r['bvid'], 'batch_date': r['first_batch'],
                                    'pub': r['pub'], 'dur': r['duration'], 'tname': r['tname'],
                                    'up_mid': r['up_mid'], 'K_real': K,
                                    'pub_hour_bin': 0 if h < 7 else (1 if h < 19 else 2)}
    bvids = list(meta_rows.keys())
    n_pool = len(bvids)
    from collections import Counter
    upc = Counter(m['up_mid'] for m in meta_rows.values())

    # 首采批快照行（bvid 跨批重采时只取首采批）
    d = (dwd_all.filter(F.col('bvid').isin(bvids))
         .join(spark.createDataFrame([(m['bvid'], m['batch_date']) for m in meta_rows.values()],
                                     ['bvid', 'fbatch']), 'bvid')
         .filter(F.col('batch_date') == F.col('fbatch')).collect())
    sent = (spark.read.parquet(f'{wh}/dwd_danmaku_sent')
            .filter(F.col('bvid').isin(bvids))
            .join(spark.createDataFrame([(m['bvid'], m['batch_date']) for m in meta_rows.values()],
                                        ['bvid', 'fbatch']), 'bvid')
            .filter(F.col('batch_date') == F.col('fbatch'))
            .select('bvid', 'text', 'offset_pct', 'sent_state').collect())
    sent_map = {bv: {} for bv in bvids}
    for r in sent:
        key = (r['text'], round(r['offset_pct'], 6) if r['offset_pct'] is not None else None)
        sent_map[r['bvid']].setdefault(key, r['sent_state'])

    by_video = {}
    for r in d:
        by_video.setdefault(r['bvid'], []).append(r)

    out_rows, asserts = [], {'n_feat_src_violation': 0, 'n_c03_zero': 0}
    for bvid in bvids:
        m = meta_rows[bvid]
        pub = m['pub']
        rows_all = by_video.get(bvid, [])
        trunc, c03, c07 = [], 0, 0
        for r in rows_all:
            if r['send_time'] is None:
                continue
            st = int(r['send_time'])
            di = (st - pub) // DAY
            if di < 0:
                continue
            if di <= 3:
                c03 += 1
            if di <= 7:
                c07 += 1
            if st < pub + 4 * DAY:          # [0,4) 日 ⇒ di∈[0,3]
                trunc.append({'st': st, 'to': r['time_offset'], 'off': r['offset_pct'],
                              'dur': r['duration'], 'w': (r['attr_weight'] or 0.0),
                              'tl': r['text_len'], 'col': r['is_colored'],
                              'mode': r['is_special_mode'], 'text': r['text'],
                              'sd': r['send_date']})
        if c03 < 1:
            asserts['n_c03_zero'] += 1
            continue
        fx = features_one(trunc, pub, sent_map.get(bvid, {}), params)
        if fx['_max_di_src'] > 3:            # 断言①：特征源不得越窗
            asserts['n_feat_src_violation'] += 1
            continue
        out_rows.append((bvid, m['batch_date'], m['K_real'], m['tname'], int(m['dur']),
                         m['up_mid'], m['pub_hour_bin'],
                         fx['n_danmaku'], fx['dm_per_min'], fx['rhythm_entropy'], fx['burst_cv'],
                         fx['opening_5pct'], fx['ending_5pct'], fx['mid_ratio'],
                         fx['avg_text_len'], fx['color_ratio'], fx['top_mode_ratio'],
                         fx['pos_ratio'], fx['neg_ratio'], fx['neu_ratio'], fx['no_hit_ratio'],
                         fx['n_events'], fx['peak_gap_cv'], fx['sync_score_med'], fx['herd_mean'],
                         float(c03), float(c07),
                         float(np.log1p(c07) - np.log1p(c03)), float(np.log1p(c07)),
                         ENGINE_VERSION))
    cols = ['bvid', 'batch_date', 'K_real', 'tname', 'duration', 'up_mid', 'pub_hour_bin',
            'n_danmaku', 'dm_per_min', 'rhythm_entropy', 'burst_cv', 'opening_5pct', 'ending_5pct',
            'mid_ratio', 'avg_text_len', 'color_ratio', 'top_mode_ratio',
            'pos_ratio', 'neg_ratio', 'neu_ratio', 'no_hit_ratio',
            'n_events', 'peak_gap_cv', 'sync_score_med', 'herd_mean',
            'c03', 'c07', 'y_growth_v2_raw', 'log1p_c07', 'engine_version']
    yv = np.array([r[27] for r in out_rows])
    lo, hi = np.percentile(yv, [1, 99])
    yw_map = {r[0]: float(v) for r, v in zip(out_rows, np.clip(yv, lo, hi))}
    upn_map = {m['bvid']: upc[m['up_mid']] for m in meta_rows.values()}

    @F.udf('double')
    def yw_of(bv):
        return yw_map.get(bv)

    @F.udf('int')
    def upn_of(bv):
        return int(upn_map.get(bv, 1))

    df = spark.createDataFrame(out_rows, cols)
    df = (df.withColumn('y_growth_v2', yw_of(F.col('bvid')))
            .drop('y_growth_v2_raw')
            .withColumn('up_prior_n', upn_of(F.col('bvid'))))
    df.write.mode('overwrite').parquet(f'{wh}/ml_features')
    n_out = df.count()

    report = {'job': 'm6_features', 'engine_version': ENGINE_VERSION,
              'n_pool_k_ge': {str(args.min_k): n_pool}, 'n_samples': n_out,
              'pool_batches': {str(r['batch_date']): r['count'] for r in
                               df.groupBy('batch_date').count().collect()},
              'provisional': f'provisional_n{n_out}',
              'y_winsor': {'lo': float(lo), 'hi': float(hi)},
              'leakage_assert': asserts,
              'note': '0..3 日窗截断口径（预注册 14 §3）；首采互动不进特征；D 层排除'}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm6_features_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m6_features：{n_out} 样本（池 {n_pool}）；断言 {asserts}')
    spark.stop()


if __name__ == '__main__':
    main()
