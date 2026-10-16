#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''perf 快照导出：data-out/perf_results.jsonl → serve/v3data/perf.json

为什么要有这个工具（而不是把数字手抄进页面）：
  页面上的每个数字必须与实验产物**同源可复现**。手抄迟早会出现
  「页面写 26.9s、jsonl 写 27.1s」这类无声漂移，而这一页是答辩证据。
  这个工具让两者永远一致，并把来源与导出时间写进文件本身。

口径纪律（写进文件，供页面照实说）：
  · 墙钟是**亚线性**：数据量 ×7.99 时墙钟只 ×1.77。线性的是数据量，不是时间 ——
    两者分开写，不合并成一句"线性"。
  · 放大数据是「重放+扰动」**合成**放大，不是真实流量。

用法：cd ~/projects/danmaku-v2 && python3 tools/make_perf_snapshot.py
'''
import io
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data-out', 'perf_results.jsonl')
DST = os.path.join(ROOT, 'serve', 'v3data', 'perf.json')

LIMITS = [
    '集群只有 1 个 Spark worker，spark-submit 固定 --total-executor-cores 2（见 Makefile 的 SUBMIT）'
    ' ⇒ par=4 这一档**测不了**，加速比只能测到 2 路并行。这是记录在仓库配置里的限制，不是推断。',
    '放大数据是「重放+扰动」**合成**放大（n_out 随倍率 1/4/8 近精确线性），不是真实流量 —— '
    '所以这页证明的是「随规模与并行的行为」，不能当作真实业务量的承诺。',
    '墙钟是**亚线性**：数据量 ×7.99 时，单并行墙钟只 ×1.77（15.2s→26.9s）。'
    '线性的是数据量，不是时间 —— 两个结论分开写，不合并成一句"线性"。',
]


def main():
    recs = []
    with io.open(SRC, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    obj = {
        'source': 'data-out/perf_results.jsonl',
        'snapshot_at': time.strftime('%Y-%m-%dT%H:%M:%S+08:00'),
        'n_records': len(recs),
        'engine_version': 'b2-v1',
        'suite': 'B5.1 性能实验（make perf-suite；放大 1/4/8 网格 + Hawkes 加速比 + 倾斜对照）',
        'limits': LIMITS,
        'records': recs,
    }
    with io.open(DST, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, separators=(',', ':'))
    print('写出 %s：%d 条实测，%d bytes' % (DST, len(recs), os.path.getsize(DST)))


if __name__ == '__main__':
    main()
