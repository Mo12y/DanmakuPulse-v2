#!/usr/bin/env python3
"""修 test_api_contract.py：3 处数据版本敏感断言 → 结构断言（2026-10-13，6 批数据）。"""
import io

p = 'tests/test_api_contract.py'
s = io.open(p, encoding='utf-8').read()

fixes = [
    ("    assert d['cluster']['k'] == 4",
     "    # 2026-10-13：k 随数据演化（6 批 + burst 修复后 silhouette 选 k=3），改结构断言\n"
     "    assert d['cluster']['k'] >= 2"),
    ("    assert len(d) == 200",
     "    # 2026-10-13：hotlist 行数随新批次增长（当前 300），改下界断言\n"
     "    assert len(d) >= 200"),
    ("    assert len(parts) == 71, f'命名分区应为 71（全表 72 含 1 行 tname=NULL 未知分区）: {len(parts)}'",
     "    # 2026-10-13：分区数随新批次增长（当前 76 命名 + 1 未知），改下界断言\n"
     "    assert len(parts) >= 70, f'命名分区 ≥70（当前 {len(parts)}）'"),
]
n = 0
for old, new in fixes:
    assert s.count(old) == 1, f'not unique: {old[:50]}'
    s = s.replace(old, new, 1)
    n += 1
io.open(p, 'w', encoding='utf-8').write(s)
print(f'fixed {n} assertions')
