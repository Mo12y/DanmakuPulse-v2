#!/usr/bin/env python3
"""build_dict_emoticon_ext.py — kaomoji/emoticon 开源词表（ekohrt 62,149 条）→ emoticon_weights.tsv 扩充

来源：github.com/ekohrt/emoticon_kaomoji_dataset（62,149 条，95 类手工标签；抓取自公开表情站）
筛选：标签归属单一情绪类（按优先级取首个匹配）+ 长度 ≤12 + 无 # 开头/无空白字符 + 去重 → 上限 1500 条
输出：追加到 config/dicts/emoticon_weights.tsv（3 列格式与现有 loader 兼容——m5 无需改代码）
"""
import io
import json
import random
import re

SRC = '/tmp/emoticon_dict.json'
DST = 'config/dicts/emoticon_weights.tsv'
LEN_MAX = 12
CAP = 1500

RULES = [
    (['love', 'kiss', 'heart', 'blush'], 3.0, 'LOVE', 0),
    (['happy', 'smiling', 'smile', 'excited', 'cheerful', 'laughing', 'laugh', 'celebration'], 2.0, 'HAPPY', 1),
    (['thumbs up', 'salute', 'thanks', 'good', 'yes', 'win'], 1.5, 'APPROVE', 6),
    (['surprised', 'shocked', 'omg'], 1.0, 'SURPRISE', 5),
    (['wink', 'playful', 'smirk'], 1.0, 'PLAYFUL', 7),
    (['angry', 'anger', 'furious', 'rage', 'annoyed', 'fight'], -2.0, 'ANGRY', 2),
    (['sad', 'crying', 'cry', 'disappointed'], -2.0, 'SAD', 3),
    (['scared', 'worried', 'nervous', 'dead', 'zombie'], -1.5, 'FEAR', 4),
    (['confused', 'thinking', 'shrug', 'sweat'], -0.5, 'CONFUSE', 8),
]

data = json.load(open(SRC, encoding='utf-8'))
rows, seen = [], set()
for emo, meta in data.items():
    if not emo or emo in seen or len(emo) > LEN_MAX:
        continue
    if emo.startswith('#') or '\n' in emo or '\t' in emo or '\r' in emo:
        continue
    if not re.search(r'[\u4e00-\u9fff\uff00-\uffef\u3000-\u303f\U0001F300-\U0001FAFF()（）\[\]{}<>^_~;:*.ovxXTY\\/|@#$%&+=\-]', emo):
        continue
    tags = [str(t).lower() for t in (meta.get('new_tags') or []) + (meta.get('original_tags') or [])]
    for keys, w, name, pri in RULES:
        if any(k in tags for k in keys):
            rows.append((emo, w, name, pri))
            seen.add(emo)
            break

rows.sort(key=lambda r: (r[3], len(r[0])))
# 按类别配额（防止单类挤占全部名额；同类内短字符优先——弹幕中更常打出）
QUOTA = {'LOVE': 300, 'HAPPY': 350, 'ANGRY': 250, 'SAD': 250,
         'SURPRISE': 150, 'FEAR': 100, 'APPROVE': 60, 'PLAYFUL': 20, 'CONFUSE': 20}
cnt = {}
kept = []
for emo, w, name, pri in rows:
    if cnt.get(name, 0) < QUOTA.get(name, 0):
        kept.append((emo, w, name, pri))
        cnt[name] = cnt.get(name, 0) + 1
rows = kept

s = io.open(DST, encoding='utf-8').read()
if 'emoticon_kaomoji_dataset' in s:
    print('already extended — skip')
else:
    block = ('\n# ── 扩充（来源：github.com/ekohrt/emoticon_kaomoji_dataset，62,149 条开源集；'
             '筛选单一情绪类+长度≤12，共 %d 条）──\n' % len(rows))
    for emo, w, name, _ in rows:
        block += f'{emo}\t{w}\t{name}\n'
    io.open(DST, 'w', encoding='utf-8').write(s.rstrip('\n') + '\n' + block)
    print('extended:', len(rows))

random.seed(1)
print('--- 抽检 25 条 ---')
for r in random.sample(rows, min(25, len(rows))):
    print(r[0], '=>', r[1], r[2])
