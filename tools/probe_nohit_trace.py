#!/usr/bin/env python3
"""probe_nohit_trace.py — no_hit 归因实验（宿主机 venv 跑，不依赖 Spark）

输入：data-out/nohit100.json（text + text_norm）
方法：镜像 m5_sentiment_v0.make_scorer 的匹配逻辑（只读用途），逐条输出中间态：
  - text_norm 归一化影响（内容丢失比例）
  - 命中明细（应为空——no_hit 定义）
  - 前缀近邻（"是某词典词真前缀(>=2字)却不构成完整命中"的子串）→ 规则/边界/分词线索
分类（每条归一类）：
  no_cjk       纯符号/数字/英文（无汉字）——真·无词可命中
  normalize_loss 归一化后内容丢失 >40%（text_norm 与 text 汉字数对比）
  prefix_near  存在前缀近邻——**最值得查的线索**（词在表里"差一点"匹配上）
  pure_gap     无任何线索——词表真缺口
输出：stdout 明细 + data-out/nohit_attribution.json（分类计数 + 明细）
"""
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'jobs'))
from m0_full import EMOJI_RULES  # noqa: E402
from m5_sentiment_v0 import load_dicts  # noqa: E402

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
IN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out', 'nohit100.json')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out', 'nohit_attribution.json')


def _j2p(pat):
    """Java regex \\x{XXXX} → Python \\UXXXXXXXX（供本地重算 text_norm）。"""
    return re.sub(r'\\x\{([0-9A-Fa-f]+)\}',
                  lambda m: '\\U%08X' % int(m.group(1), 16), pat)


_PY_RULES = [(_j2p(p), t) for p, t in EMOJI_RULES]


def norm_text_py(t):
    for p, tok in _PY_RULES:
        t = re.sub(p, tok, t)
    return t

g, h, e, anchors, neg, deg = load_dicts(CONFIG)
lookup = {}
for w, (pol, st) in g.items():
    lookup[w] = ('g', pol, st)
for w, (pol, wt) in h.items():
    lookup[w] = ('h', pol, wt)
first = {w[0] for w in lookup}
maxlen = max(len(w) for w in lookup)
prefixes = set()
for w in lookup:
    for L in range(2, len(w)):
        prefixes.add(w[:L])

CJK = re.compile(r'[\u4e00-\u9fff]')


def scan(text):
    hits, near = [], []
    i, n = 0, len(text)
    while i < n:
        if text[i] not in first:
            i += 1
            continue
        matched = False
        for L in range(min(maxlen, n - i), 0, -1):
            piece = text[i:i + L]
            if piece in lookup:
                hits.append((i, piece, lookup[piece]))
                i += L
                matched = True
                break
        if not matched:
            for L in range(min(maxlen - 1, n - i), 1, -1):
                if text[i:i + L] in prefixes:
                    near.append((i, text[i:i + L]))
                    break
            i += 1
    return hits, near


def main():
    rows = json.load(open(IN, encoding='utf-8'))
    detail, kinds = [], Counter()
    for r in rows:
        text = r['text']
        tn = norm_text_py(text)  # sent 表未存 text_norm（本轮发现）——本地按同一规则重算
        hits, near = scan(tn)
        hits_text, _ = scan(text)
        c_cjk = len(CJK.findall(text))
        tn_cjk = len(CJK.findall(tn))
        if c_cjk == 0:
            kind = 'no_cjk'
        elif tn_cjk == 0 or (c_cjk - tn_cjk) / c_cjk > 0.4:
            kind = 'normalize_loss'
        elif near:
            kind = 'prefix_near'
        else:
            kind = 'pure_gap'
        kinds[kind] += 1
        detail.append({'kind': kind, 'text': text, 'text_norm': tn,
                       'hits_on_norm': hits, 'hits_on_text': hits_text,
                       'prefix_near': near[:5]})
    summary = {
        'n': len(rows), 'kinds': dict(kinds),
        'note': 'no_hit 归因：no_cjk=无汉字真无词；normalize_loss=归一化丢内容；'
                'prefix_near=前缀近邻(词表有"半截"——规则/边界线索)；pure_gap=词表真缺口',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'detail': detail}, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    # 明细打印（紧凑）
    for d in detail:
        line = f"[{d['kind']}] {d['text'][:24]!r}"
        if d['prefix_near']:
            line += f"  NEAR={[(i, s) for i, s in d['prefix_near']]}"
        print(line)


if __name__ == '__main__':
    main()
