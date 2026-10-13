#!/usr/bin/env python3
"""修 export_cases_b53.py：ev_cnt 改为全量事件计数（UP 视频也能 join 到）。"""
import io

p = 'jobs/export_cases_b53.py'
lines = io.open(p, encoding='utf-8').read().split('\n')
fixed = 0
out = []
for ln in lines:
    if ln.startswith('ev_cnt = ev.groupBy'):
        out.append("ev_cnt = (spark.read.parquet(f'{WH}/dws_rhythm_events').groupBy('bvid')")
        out.append("          .agg(F.count('*').alias('n_events')))")
        fixed += 1
    else:
        out.append(ln)
assert fixed == 1, f'expected 1, got {fixed}'
io.open(p, 'w', encoding='utf-8').write('\n'.join(out))
print('fixed')
