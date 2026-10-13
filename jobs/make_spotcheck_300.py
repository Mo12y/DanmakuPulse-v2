#!/usr/bin/env python3
"""make_spotcheck_300.py — B5.2 抽检样本生成（案 B 口径：300 条抽检级）

输入：data-out/annotation/sent_sample.csv（B1.4 的 1000 条，空标签）
输出：data-out/annotation/sent_spotcheck_300.csv（按 tname 分层抽 300；列同源 + 空 label）
     data-out/annotation/progress.csv（追加 spotcheck 任务行；保留原双人设计行以备升级）
分层：每分区分配 n = round(300 × 分区占比)，最大余数法补足到 300；组内随机（seed=42）。
纪律：不 join 任何机器输出（防锚定偏差）；标注指南见 v2 docs/annotation_guides.md（五值）。
"""
import csv
import io
import os
import random

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

WH = '/warehouse'
ANN = '/opt/work/data-out/annotation'
SRC = f'{ANN}/sent_sample.csv'
DST = f'{ANN}/sent_spotcheck_300.csv'
PROG = f'{ANN}/progress.csv'

spark = SparkSession.builder.appName('make_spotcheck_300').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

rows = list(csv.DictReader(io.open(SRC, encoding='utf-8')))
print(f'source: {len(rows)} 条')

bv = [r['bvid'] for r in rows]
tn = (spark.read.parquet(f'{WH}/dim_video').filter(F.col('bvid').isin(bv))
      .select('bvid', 'tname').distinct().collect())
tmap = {r['bvid']: r['tname'] for r in tn}

# 分层分配（最大余数法）
from collections import defaultdict
by_t = defaultdict(list)
for r in rows:
    by_t[tmap.get(r['bvid'], '未知')].append(r)
total = len(rows)
alloc = {t: 300 * len(v) / total for t, v in by_t.items()}
base = {t: int(a) for t, a in alloc.items()}
rem = 300 - sum(base.values())
frac = sorted(alloc, key=lambda t: alloc[t] - base[t], reverse=True)
for t in frac[:rem]:
    base[t] += 1

random.seed(42)
picked = []
for t, n in sorted(base.items(), key=lambda kv: -kv[1]):
    pool = by_t[t][:]
    random.shuffle(pool)
    picked.extend(pool[:n])
    print(f'  {t}: {len(pool)} → {n}')

random.shuffle(picked)
with open(DST, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['sample_id', 'bvid', 'batch_date', 'text', 'label', 'unsure'])
    for i, r in enumerate(picked, 1):
        w.writerow([i, r['bvid'], r['batch_date'], r['text'], '', ''])
print(f'✅ 抽检样本 {len(picked)} 条 → {DST}')

# 进度表：追加 spotcheck 行（不动原双人设计行）
lines = io.open(PROG, encoding='utf-8').read().rstrip('\n').split('\n')
if not any('spotcheck' in l for l in lines):
    lines.append('B 情感抽检(案B 300 单人),300,0,,,sent_spotcheck_300.csv；对照脚本 tools/compare_spotcheck.py；一致率≥80%')
    io.open(PROG, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('progress.csv 追加 spotcheck 行')
spark.stop()
