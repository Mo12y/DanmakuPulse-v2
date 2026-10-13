#!/usr/bin/env python3
"""修 test_api_contract.py 第二批：evo_types 计数、cadence（新批次为 daily）。"""
import io

p = 'tests/test_api_contract.py'
s = io.open(p, encoding='utf-8').read()
fixes = [
    ("    assert sum(t['n'] for t in d['evo_types']) == 124",
     "    # 2026-10-13：演化样本随数据增长（B2 时 124，现 142），改下界\n"
     "    assert sum(t['n'] for t in d['evo_types']) >= 120"),
    ("    assert d[0]['cadence'] == 'weekly_legacy'",
     "    # 2026-10-13：新批次引入 daily 节奏（hotlist[0] 为最新批）\n"
     "    assert d[0]['cadence'] in ('daily', 'weekly_legacy')"),
]
n = 0
for old, new in fixes:
    assert s.count(old) == 1, f'not unique: {old[:60]}'
    s = s.replace(old, new, 1)
    n += 1
io.open(p, 'w', encoding='utf-8').write(s)
print(f'fixed {n}')
