#!/usr/bin/env python3
"""export_cases_b53.py — B5.3 价值案例数据导出（3 视频高能对照 + 3 UP 诊断）

输出：data-out/cases_b53.json（量小，driver 写出）
选样（2026-10-13）：
  视频：BV1HBbE6cEc5（日常，aft 锚 970）｜BV1Lxej6FEhv（网络游戏，160）｜BV1wAp76yEu1（人文历史，19）
  UP：伤心欲茄222（HIGH，6 视频/rank 12）｜拉宏桑（MID，4/229）｜异环（LOW，1/网络游戏）
"""
import json

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

VIDEOS = ['BV1HBbE6cEc5', 'BV1Lxej6FEhv', 'BV1wAp76yEu1']
UPS = ['3493260618106936', '11870568', '3546636978489848']
WH = '/warehouse'
OUT = '/opt/work/data-out/cases_b53.json'

spark = SparkSession.builder.appName('export_cases_b53').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

dwd = spark.read.parquet(f'{WH}/dwd_danmaku_clean')
meta = (dwd.filter(F.col('bvid').isin(VIDEOS)).groupBy('bvid')
        .agg(F.max('title').alias('title'), F.max('duration').alias('duration'),
             F.max('tname').alias('tname'), F.max('up_name').alias('up_name'),
             F.max('batch_date').alias('last_batch')))
meta_map = {r['bvid']: r.asDict() for r in meta.collect()}

# 2026-10-14 修复：16 号表头写明「系统事件（该视频最后一批内）」，但原实现未按 last_batch 过滤，
# 导致跨批次视频（如 BV1wAp76yEu1 在 20261012/20261013 各 3 段）的事件被合并展示，与文档口径不符。
last_batch_df = meta.select('bvid', F.col('last_batch').alias('lb'))
ev = (spark.read.parquet(f'{WH}/dws_rhythm_events')
      .join(last_batch_df, ['bvid'], 'inner')
      .filter(F.col('bvid').isin(VIDEOS) & (F.col('batch_date') == F.col('lb')))
      .drop('lb')
      .select('bvid', 'batch_date', 'event_id', 'start_sec', 'end_sec', 'score_fused',
              'n_methods', 'sync_score', 'sync_p', 'n_anchors', 'n_anchors_aft', 'delta_hat_sec')
      .orderBy('bvid', 'start_sec'))
ev_map = {}
for r in ev.collect():
    ev_map.setdefault(r['bvid'], []).append(r.asDict())

d5 = (spark.read.parquet(f'{WH}/dws_density_5s').filter(F.col('bvid').isin(VIDEOS))
      .select('bvid', 'batch_date', 'bucket5s', 'cnt', 'wcnt'))
d5_map = {}
for r in d5.collect():
    d5_map.setdefault(r['bvid'], []).append(r.asDict())

anchor = (spark.read.parquet(f'{WH}/dwd_danmaku_sent')
          .filter(F.col('bvid').isin(VIDEOS) & F.col('is_anchor'))
          .select('bvid', 'batch_date', 'offset_pct', 'anchor_type', 'text'))
anc_map = {}
for r in anchor.collect():
    anc_map.setdefault(r['bvid'], []).append(r.asDict())

videos = []
for b in VIDEOS:
    m = meta_map.get(b, {})
    videos.append({
        'bvid': b, 'title': m.get('title'), 'tname': m.get('tname'),
        'up_name': m.get('up_name'), 'duration': m.get('duration'), 'last_batch': m.get('last_batch'),
        'events': ev_map.get(b, []), 'density': d5_map.get(b, []), 'anchors': anc_map.get(b, []),
    })

prof = spark.read.parquet(f'{WH}/dws_up_profile').filter(F.col('up_mid').isin(UPS))
prof_map = {r['up_mid']: r.asDict() for r in prof.collect()}

dim = spark.read.parquet(f'{WH}/dim_video').select('bvid', 'up_mid', 'tname')
fp = spark.read.parquet(f'{WH}/dws_video_rhythm_features').select(
    'bvid', 'dm_per_min', 'rhythm_entropy', 'burst_cv', 'pos_ratio', 'no_hit_ratio', 'n_danmaku')
# 2026-10-14 修复：M4 v2「换变量」后 dws_evolution_feats 已无 n_days / n_turns 列
# （v2 列为 n_obs / span_days / g_play / g_like / g_reported / cluster_type ...）。
# 此处以 n_obs（观测批次数）作等价列；**上榜天数的权威来源是 dws_lifecycle.n_days_obs**（仍在，未变）。
evo = spark.read.parquet(f'{WH}/dws_evolution_feats').select(
    'bvid', 'cluster_type',
    F.col('n_obs').alias('n_days'), F.col('n_obs').alias('n_turns'))
lc = spark.read.parquet(f'{WH}/dws_lifecycle').select('bvid', 'n_days_obs', 'half_decay_days', 'hk_halflife')
ev_cnt = (spark.read.parquet(f'{WH}/dws_rhythm_events').groupBy('bvid')
          .agg(F.count('*').alias('n_events')))
titles = dwd.select('bvid', 'title').distinct()

upv = (dim.filter(F.col('up_mid').isin(UPS))
       .join(fp, 'bvid', 'left').join(evo, 'bvid', 'left').join(lc, 'bvid', 'left')
       .join(ev_cnt, 'bvid', 'left').join(titles, 'bvid', 'left'))
upv_map = {}
for r in upv.collect():
    upv_map.setdefault(r['up_mid'], []).append(r.asDict())

ups = [{'up_mid': u, 'profile': prof_map.get(u, {}), 'videos': upv_map.get(u, [])} for u in UPS]

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump({'videos': videos, 'ups': ups}, f, ensure_ascii=False, indent=1)

print(f'✅ 导出 {len(videos)} 视频 / {len(ups)} UP → {OUT}')
for v in videos:
    print(f"  {v['bvid']} {v['tname']}: events={len(v['events'])} density={len(v['density'])} anchors={len(v['anchors'])}")
for u in ups:
    print(f"  UP {u['profile'].get('up_name')}: videos={len(u['videos'])}")
spark.stop()
