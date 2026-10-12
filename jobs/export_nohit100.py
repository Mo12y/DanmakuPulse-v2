#!/usr/bin/env python3
"""export_nohit100.py — 导出 100 条 objective(no_hit) 样本（text + text_norm）供归因实验（集群跑）。"""
import json

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('export_nohit100').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
rows = (spark.read.parquet('/warehouse/dwd_danmaku_sent')
        .filter(F.col('sent_state') == 'objective')
        .filter((F.length('text') >= 2) & (F.length('text') <= 30))
        .filter(F.rand(7) < 0.01)
        .select('bvid', 'text')
        .limit(100).collect())
out = '/opt/work/data-out/nohit100.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump([{'bvid': r['bvid'], 'text': r['text']} for r in rows],
              f, ensure_ascii=False, indent=1)
print(f'exported {len(rows)} -> {out}')
spark.stop()
