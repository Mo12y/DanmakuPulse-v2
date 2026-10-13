#!/usr/bin/env python3
"""F1 二修：detect_spam 的窗口函数（lag.over 顺序 + range frame 的 orderBy 类型）。"""
import io

p = 'jobs/m0_full.py'
s = io.open(p, encoding='utf-8').read()
fixes = [
    # 窗口定义：range frame 的 orderBy 必须数值（send_time 为 string）
    ("    wu = Window.partitionBy('bvid', 'batch_date', 'user_hash').orderBy('send_time')",
     "    # ⚠️ 2026-10-13 二修（F1）：range frame 的 orderBy 须数值型；此前因 user_hash 从未生效，\n"
     "    #    该代码块从未被执行，两处隐藏 bug 今日首次暴露（lag.over 顺序 + string 排序键）。\n"
     "    wu = (Window.partitionBy('bvid', 'batch_date', 'user_hash')\n"
     "          .orderBy(F.col('send_time').cast('long')))"),
    # lag：先 over 再 cast
    ("             .withColumn('gap', F.col('send_time').cast('long') -\n"
     "                         F.lag('send_time').cast('long').over(wu))",
     "             .withColumn('gap', F.col('send_time').cast('long') -\n"
     "                         F.lag('send_time').over(wu).cast('long'))"),
]
n = 0
for old, new in fixes:
    assert s.count(old) == 1, f'not unique ({s.count(old)}): {old[:60]}'
    s = s.replace(old, new, 1)
    n += 1
io.open(p, 'w', encoding='utf-8').write(s)
print(f'fixed {n} spots')
