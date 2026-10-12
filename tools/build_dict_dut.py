#!/usr/bin/env python3
"""build_dict_dut.py — DUTIR 情感词汇 CSV → config/dicts/dut_sentiment.tsv

输入：/tmp/dutir_emotional.csv（27,466 词，字段：词语/词性种类/词义数/词义序号/情感分类/强度/极性/…）
输出：config/dicts/dut_sentiment.tsv（word \t polarity \t strength \t category \t source）
规则：极性 1→+1、2→−1；0（中性）/3（兼有褒贬）不收录（v0 规避歧义，记 CHANGELOG）；
      同词多条记录取「强度最大」的一条。
"""
import csv
import io
import os

SRC = '/tmp/dutir_emotional.csv'
DST = 'config/dicts/dut_sentiment.tsv'
os.makedirs(os.path.dirname(DST), exist_ok=True)

rows = {}
n_total = 0
with open(SRC, encoding='utf-8') as f:
    r = csv.reader(f)
    next(r)
    for row in r:
        if len(row) < 7 or not row[0].strip():
            continue
        n_total += 1
        word = row[0].strip()
        cat = row[4].strip()
        try:
            s = float(row[5]) if row[5].strip() else 0.0
            p = float(row[6]) if row[6].strip() else 0.0
        except ValueError:
            continue
        if p in (1.0, 2.0):
            pol = 1 if p == 1.0 else -1
            if word not in rows or s > rows[word][1]:
                rows[word] = (pol, int(s), cat)

with io.open(DST, 'w', encoding='utf-8') as f:
    f.write('# word\tpolarity\tstrength\tcategory\tsource\n')
    for w, (p, s, c) in sorted(rows.items()):
        f.write(f'{w}\t{p}\t{s}\t{c}\tdut\n')

n_pos = sum(1 for v in rows.values() if v[0] == 1)
n_neg = len(rows) - n_pos
print(f'DUT 行数={n_total} → 收录={len(rows)}（正 {n_pos} / 负 {n_neg}），输出 {DST}')
