#!/usr/bin/env python3
"""F1 修复：m0_full 读 ods 全表时开启 mergeSchema（新批次含 user_hash，历史批 null）。"""
import io

p = 'jobs/m0_full.py'
s = io.open(p, encoding='utf-8').read()
old = "    ods = spark.read.parquet(f'{wh}/ods_danmaku_full')"
new = ("    # ⚠️ 2026-10-13 修复(F1)：全表读需 mergeSchema——新批次（1012/1013）含 user_hash 列、\n"
       "    #    历史 4 批无；默认 schema 取代表性文件（老批）会导致 user_hash '不存在' 而跳过刷量检测。\n"
       "    ods = (spark.read.option('mergeSchema', 'true')\n"
       "           .parquet(f'{wh}/ods_danmaku_full'))")
assert s.count(old) == 1, f'count={s.count(old)}'
s = s.replace(old, new, 1)
io.open(p, 'w', encoding='utf-8').write(s)
print('m0_full patched')
