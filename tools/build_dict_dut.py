#!/usr/bin/env python3
"""build_dict_dut.py — DUTIR 情感词汇 CSV → config/dicts/dut_sentiment.tsv（v2，2026-10-12 解冻后修）

v2 变更（用户批复·案 B 配套执行）：极性 0 的词**按情感类别补方向**——
  「难过」（NB/5/**0**）等约 4,500 词曾被 v1 的"只收极性 1/2"规则整体挡掉；
  v2：负向类（NB..NL）0→−1、正向类（PA..PK）0→+1（强度保留）；3（兼有）维持排除；类别不明（prep 等）排除。
输入：config/dicts/source/dutir_emotional.csv（已归档；fallback /tmp/dutir_emotional.csv）
输出：config/dicts/dut_sentiment.tsv + config/dicts/source/dut_skipped.tsv（被挡清单，含原因）
"""
import csv
import hashlib
import io
import os

SRC_CANDIDATES = ['config/dicts/source/dutir_emotional.csv', '/tmp/dutir_emotional.csv']
DST = 'config/dicts/dut_sentiment.tsv'
SKIPPED = 'config/dicts/source/dut_skipped.tsv'

NEG_CATS = {'NB', 'NE', 'ND', 'NN', 'NJ', 'NH', 'NK', 'NL'}
POS_CATS = {'PA', 'PE', 'PD', 'PH', 'PG', 'PB', 'PK', 'PC'}

SRC = next(p for p in SRC_CANDIDATES if os.path.exists(p))
md5 = hashlib.md5(open(SRC, 'rb').read()).hexdigest()

rows = {}
skipped = []
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
            skipped.append((word, cat, row[6].strip(), 'parse_error'))
            continue
        if p == 1.0:
            pol = 1
        elif p == 2.0:
            pol = -1
        elif p == 0.0 and cat in NEG_CATS:
            pol = -1
        elif p == 0.0 and cat in POS_CATS:
            pol = 1
        else:
            reason = 'dual(3)' if p == 3.0 else (
                'polarity0_cat_unknown' if p == 0.0 else f'polarity_{p}')
            skipped.append((word, cat, row[6].strip(), reason))
            continue
        if word not in rows or s > rows[word][1]:
            rows[word] = (pol, int(s), cat)

with io.open(DST, 'w', encoding='utf-8') as f:
    f.write('# word\tpolarity\tstrength\tcategory\tsource\n')
    for w, (p, s, c) in sorted(rows.items()):
        f.write(f'{w}\t{p}\t{s}\t{c}\tdut\n')

with io.open(SKIPPED, 'w', encoding='utf-8') as f:
    f.write('# word\tcategory\tpolarity_raw\treason（v2 被挡清单——一手证据）\n')
    for w, c, praw, reason in skipped:
        f.write(f'{w}\t{c}\t{praw}\t{reason}\n')

n_pos = sum(1 for v in rows.values() if v[0] == 1)
n_neg = len(rows) - n_pos
print(f'[v2] src={SRC} md5={md5}')
print(f'[v2] DUT 行数={n_total} → 收录={len(rows)}（正 {n_pos} / 负 {n_neg}）｜被挡={len(skipped)} → {SKIPPED}')
print(f'[v2] 输出 {DST}')
