#!/usr/bin/env python3
"""compare_spotcheck.py — B5.2 抽检对比（标注回传后运行）

输入：data-out/annotation/sent_spotcheck_300.csv（人工填好 label 列；unsure 列可留空）
     系统输出：dwd_danmaku_sent（按 (bvid,batch_date,text) 匹配）
口径（12 §3.1 对齐表）：
  人工五值 → 系统四值域：pos→pos ｜ neg→neg ｜ neu→neu ｜ **obj→objective** ｜ **unsure→排除**
  输出：符合率（总 + 逐类）、4×4 混淆矩阵（系统 × 人工域）、unsure 占比、缺口（未匹配数）
判据：一致率 ≥80%（案 B）；<80% → 按 11 §5 决策点处理（记录，不自行降级）
运行：集群 spark-submit（读挂载 csv）；
"""
import csv
import io

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

ANN = '/opt/work/data-out/annotation'
SRC = f'{ANN}/sent_spotcheck_300.csv'

spark = SparkSession.builder.appName('compare_spotcheck').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

rows = list(csv.DictReader(io.open(SRC, encoding='utf-8')))
labeled = [r for r in rows if r['label'].strip()]
n_unsure = sum(1 for r in labeled if r['label'].strip() == 'unsure')
evaluable = [r for r in labeled if r['label'].strip() not in ('unsure', '')]
print(f'样本 {len(rows)} ｜ 已标 {len(labeled)} ｜ unsure {n_unsure} ｜ 进入评估 {len(evaluable)}')

if not evaluable:
    print('⚠️ 尚无已标注数据——本脚本为标注回传后运行（当前仅准备就绪）')
    spark.stop()
    raise SystemExit(0)

keys = [(r['bvid'], r['batch_date'], r['text']) for r in evaluable]
kdf = spark.createDataFrame(keys, ['bvid', 'batch_date', 'text'])
sys_state = (spark.read.parquet('/warehouse/dwd_danmaku_sent')
             .select('bvid', 'batch_date', 'text', 'sent_state')
             .groupBy('bvid', 'batch_date', 'text')
             .agg(F.first('sent_state').alias('sent_state')))
m = kdf.join(sys_state, ['bvid', 'batch_date', 'text'], 'left')
smap = {(r['bvid'], r['batch_date'], r['text']): (r['sent_state'] or 'MISSING')
        for r in m.collect()}

MAP = {'pos': 'pos', 'neg': 'neg', 'neu': 'neu', 'obj': 'objective'}
labels = ['pos', 'neg', 'neu', 'objective']
mat = {a: {b: 0 for b in labels} for a in labels}
match = miss = 0
for r in evaluable:
    human = MAP[r['label'].strip()]
    sysv = smap.get((r['bvid'], r['batch_date'], r['text']), 'MISSING')
    if sysv not in labels:
        miss += 1
        continue
    mat[sysv][human] += 1
    if sysv == human:
        match += 1

n_eval = max(len(evaluable) - miss, 1)
print(f'\n匹配 {len(evaluable) - miss} ｜ 缺口（sent 未匹配）{miss}')
print(f'✅ 一致率（系统 vs 人工）= {match / n_eval:.4f}（判据 ≥0.80）')
print('\n混淆矩阵（行=系统，列=人工）：')
print('        ' + '  '.join(f'{c:>9}' for c in labels))
for a in labels:
    print(f'{a:>9} ' + '  '.join(f'{mat[a][b]:>9}' for b in labels))
recall = {b: (mat[b][b] / max(sum(mat[a][b] for a in labels), 1)) for b in labels}
print('\n逐类命中率（对角线/列和）:', {k: round(v, 3) for k, v in recall.items()})
spark.stop()
