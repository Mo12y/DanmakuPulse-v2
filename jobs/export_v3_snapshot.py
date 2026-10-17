# -*- coding: utf-8 -*-
'''v3 诊断前端供数快照（docs/05-新系统重构/33 号 §7 交付物 1/2）

性质：**只读导出**——不改数据管线、不重跑任何 ML 实验、不写 HDFS。
     读 HDFS /warehouse 现有表 + 现有 DWS 口径，落一份前端可直读的快照。

输出 -> /opt/work/data-out/v3_snapshot/{videos,curves,baselines,events,meta}.json

口径说明（与 32 号 P4 一致）：
  - 偏离 z = (v - 分区中位) / (1.4826 * MAD)，分区内稳健 z
  - 定位 pct = 同分区内 <= 本值的比例 * 100（百分位，不是 z）
  - 展示曲线 = 1% 桶 cnt 的"占比"（每桶/全片合计），形状可比、不受时长与总量混淆
'''
import json
import time
from collections import defaultdict

from pyspark.sql import SparkSession, Window, functions as F

WH = '/warehouse'
OUT = '/opt/work/data-out/v3_snapshot'

# 主视图展示维（33 号 §2 词典全覆盖；主视图只露 top3-4，其余进折叠层）
DIMS = ['dm_per_min', 'wdm_per_min', 'opening_5pct', 'early_lag_ratio', 'main_peak_pct',
        'rhythm_entropy', 'burst_cv', 'peak_cnt', 'ending_5pct', 'mid_ratio', 'avg_text_len']
# 节奏签名空间（P1 证明形状跨批稳定 within 0.992；kNN 只用形状，不用绝对强度）
SHAPE = ['opening_5pct', 'ending_5pct', 'mid_ratio', 'rhythm_entropy', 'burst_cv', 'early_lag_ratio']
MIN_PARTITION = 20


def rd(x, n=6):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float('inf'), float('-inf')):
        return None
    return round(v, n)


spark = (SparkSession.builder.appName('export_v3_snapshot')
         .config('spark.sql.shuffle.partitions', '8').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

# ── 1. 指纹 + 维度 + 结果变量 ────────────────────────────────
fp = spark.read.parquet(f'{WH}/dws_video_rhythm_features')
dim = (spark.read.parquet(f'{WH}/dim_video')
       .select('bvid', 'title', 'tname', 'up_name', 'up_mid', 'duration', 'pubdate'))
lc = (spark.read.parquet(f'{WH}/dws_lifecycle')
      .select('bvid', 'span_days', 'half_decay_days', 'hk_halflife', 'halflife_d',
              'n_days_obs', 'n_obs', 'play_last', 'engine_version'))

mt = spark.read.parquet(f'{WH}/dws_metric_track').select('bvid', 'batch_date', 'play', 'danmaku_reported')
mt_latest = (mt.withColumn('_rn', F.row_number().over(
                 Window.partitionBy('bvid').orderBy(F.desc('batch_date'))))
             .filter(F.col('_rn') == 1).drop('_rn'))

base = (fp.join(dim, 'bvid', 'left')
          .join(lc, 'bvid', 'left')
          .join(mt_latest.select('bvid', F.col('play').alias('play_latest')), 'bvid', 'left'))

# 主峰位置（进度轴 0-100）：1% 桶 wcnt 的 argmax
d1 = spark.read.parquet(f'{WH}/dws_density_1pct').select(
    'bvid', 'batch_date', 'bucket1pct', 'cnt', 'wcnt')
peak = (d1.groupBy('bvid', 'batch_date')
          .agg(F.max(F.struct('wcnt', 'bucket1pct')).alias('top'))
          .select('bvid', 'batch_date', F.col('top.bucket1pct').alias('main_peak_pct'),
                  F.col('top.wcnt').alias('peak_wcnt')))
base = base.join(peak, ['bvid', 'batch_date'], 'left')

rows = base.collect()
print('指纹行数', len(rows))

# ── 2. 曲线：1% 桶 cnt → 占比（形状可比） ────────────────────
curve_rows = d1.select('bvid', 'batch_date', 'bucket1pct', 'cnt').collect()
raw = defaultdict(dict)
for r in curve_rows:
    raw[(r['bvid'], r['batch_date'])][int(r['bucket1pct'])] = float(r['cnt'] or 0.0)

curves = {}
for k, d in raw.items():
    tot = sum(d.values()) or 1.0
    curves['%s|%s' % k] = [round(d.get(i, 0.0) / tot, 6) for i in range(101)]

# ── 3. 分区基准（展示用占比包络 + 特征 med/MAD + 百分位） ───────
def pctl(sorted_vals, q):
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

by_part = defaultdict(list)          # tname -> [idx]
recs = []
for i, r in enumerate(rows):
    d = r.asDict()
    t = d.get('tname')
    if t:
        by_part[t].append(i)
    recs.append(d)

baselines = {}
for t, idxs in by_part.items():
    n = len(idxs)
    feats = {}
    for c in DIMS:
        vals = sorted(v for v in (recs[i].get(c) for i in idxs) if v is not None)
        if not vals:
            continue
        med = pctl(vals, 0.5)
        mad = pctl(sorted(abs(v - med) for v in vals), 0.5)
        feats[c] = {'med': rd(med), 'mad': rd(mad * 1.4826, 8), 'n': len(vals)}
    # 展示包络：分区内逐桶占比的 p25 / p50 / p75
    per_bucket = [[] for _ in range(101)]
    for i in idxs:
        cur = curves.get('%s|%s' % (recs[i]['bvid'], recs[i]['batch_date']))
        if not cur:
            continue
        for b in range(101):
            per_bucket[b].append(cur[b])
    env = {}
    for name, q in (('p25', 0.25), ('med', 0.5), ('p75', 0.75)):
        env[name] = [round(pctl(sorted(per_bucket[b]), q) or 0.0, 6) for b in range(101)]
    baselines[t] = {'n_videos': n, 'low_support': n < MIN_PARTITION,
                    'feats': feats, 'curve_p25': env['p25'],
                    'curve_med': env['med'], 'curve_p75': env['p75'],
                    'n_curve': len(per_bucket[50])}

# ── 4. 百分位（先算好，前端不碰 z） ───────────────────────────
pct_of = {}
for t, idxs in by_part.items():
    for c in DIMS:
        vals = sorted(v for v in (recs[i].get(c) for i in idxs) if v is not None)
        if not vals:
            continue
        for i in idxs:
            v = recs[i].get(c)
            if v is None:
                continue
            le = sum(1 for x in vals if x <= v)
            pct_of[(i, c)] = round(100.0 * le / len(vals), 1)

# ── 5. 事件 + 窗口内 top 弹幕 ────────────────────────────────
ev = (spark.read.parquet(f'{WH}/dws_rhythm_events')
      .select('bvid', 'batch_date', 'event_id', 'start_sec', 'end_sec',
              'start_bucket', 'end_bucket', 'score_fused', 'n_methods', 'sync_p', 'n_window'))
ev_rows = ev.collect()
print('事件数', len(ev_rows))
ev_keys = {(r['bvid'], r['batch_date']) for r in ev_rows}

dm = (spark.read.parquet(f'{WH}/dwd_danmaku_clean')
      .filter(~F.col('is_spam'))
      .select('bvid', 'batch_date', 'time_offset', 'text', 'attr_weight'))
dm = dm.join(spark.createDataFrame(
    [(r['bvid'], r['batch_date'], float(r['start_sec']), float(r['end_sec']), int(r['event_id']))
     for r in ev_rows], ['bvid', 'batch_date', 'start_sec', 'end_sec', 'event_id']),
    ['bvid', 'batch_date'])
dm = dm.filter((F.col('time_offset') >= F.col('start_sec')) &
               (F.col('time_offset') <= F.col('end_sec')))
dm = dm.withColumn('_rn', F.row_number().over(
    Window.partitionBy('bvid', 'batch_date', 'event_id')
     .orderBy(F.desc(F.col('attr_weight')))))
top = dm.filter(F.col('_rn') <= 6).collect()
tops = defaultdict(list)
for r in top:
    tx = (r['text'] or '').strip()
    if tx and tx not in tops[(r['bvid'], r['batch_date'], int(r['event_id']))]:
        tops[(r['bvid'], r['batch_date'], int(r['event_id']))].append(tx)

events = defaultdict(list)
for r in ev_rows:
    key = (r['bvid'], r['batch_date'], int(r['event_id']))
    events['%s|%s' % (r['bvid'], r['batch_date'])].append({
        'id': int(r['event_id']),
        't0': rd(r['start_sec'], 1), 't1': rd(r['end_sec'], 1),
        'b0': int(r['start_bucket']) if r['start_bucket'] is not None else None,
        'b1': int(r['end_bucket']) if r['end_bucket'] is not None else None,
        'score': rd(r['score_fused'], 4), 'n_methods': int(r['n_methods'] or 0),
        'sync_p': rd(r['sync_p'], 4), 'n_window': int(r['n_window'] or 0),
        'tops': tops.get(key, [])[:6],
    })

# ── 6. 落盘 ─────────────────────────────────────────────────
videos = []
for i, d in enumerate(recs):
    t = d.get('tname')
    key = '%s|%s' % (d['bvid'], d['batch_date'])
    videos.append({
        'bvid': d['bvid'], 'batch_date': str(d['batch_date']),
        'title': d.get('title'), 'tname': t, 'up_name': d.get('up_name'),
        'duration': d.get('duration'), 'pubdate': d.get('pubdate'),
        'n_danmaku': d.get('n_danmaku'),
        'play': d.get('play_latest'),
        'danmaku_reported': d.get('danmaku_reported'),
        'f': {c: rd(d.get(c)) for c in DIMS},
        'shape': {c: rd(d.get(c)) for c in SHAPE},
        'pct': {c: pct_of.get((i, c)) for c in DIMS},
        'outcome': {
            'span_days': rd(d.get('span_days'), 3),
            'half_decay_days': rd(d.get('half_decay_days'), 3),
            'hk_halflife': rd(d.get('hk_halflife'), 3),
            'halflife_d': rd(d.get('halflife_d'), 3),
            'n_days_obs': d.get('n_days_obs'),
            'n_obs': d.get('n_obs'),
        },
        'has_curve': key in curves,
    })

meta = {
    'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'n_videos': len(videos), 'n_curve': len(curves), 'n_partitions': len(baselines),
    'n_events': len(ev_rows), 'n_events_with_tops': sum(1 for v in events.values() for e in v if e['tops']),
    'dims': DIMS, 'shape_dims': SHAPE, 'min_partition': MIN_PARTITION,
    'engine_version': 'b2-v1',
    'z_formula': 'robust z = (v - partition_median) / (1.4826 * MAD)',
    'curve_norm': 'share of total per 1% bucket (shape)',
    'source_tables': ['dws_video_rhythm_features', 'dws_density_1pct', 'dim_video',
                      'dws_lifecycle', 'dws_metric_track', 'dws_rhythm_events',
                      'dwd_danmaku_clean'],
}

import os
os.makedirs(OUT, exist_ok=True)

# ── 快照写入：先全部写进暂存目录，再逐个原子替换 ──────────────────────
# 为什么不能直接 open(OUT/xxx.json, 'w')：那是**原地截断重写**，
#   ① 写到一半失败 -> 前端读到新旧混合的快照（videos 是新的、curves 是旧的）；
#   ② 前端恰好在写的时候读 -> 读到**截断的 JSON**，json.load 直接抛异常。
# 改成「暂存 + os.replace」后：中途失败旧快照一个字节都不动；替换是原子的。
# meta.json **最后替换**，它天然是这次导出的提交标记。
FILES = (('videos', videos), ('curves', curves), ('baselines', baselines),
         ('events', dict(events)), ('meta', meta))
STAGING = os.path.join(OUT, '.staging')
os.makedirs(STAGING, exist_ok=True)
for name, obj in FILES:
    with open(os.path.join(STAGING, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
for name, _ in FILES:
    src = os.path.join(STAGING, name + '.json')
    dst = os.path.join(OUT, name + '.json')
    os.replace(src, dst)                      # 同文件系统内原子替换
    print('写出', name, os.path.getsize(dst), 'bytes')
os.rmdir(STAGING)
print('无曲线行数', sum(1 for v in videos if not v['has_curve']))
print('分区覆盖', sum(v['n_curve'] for v in baselines.values()))
spark.stop()

