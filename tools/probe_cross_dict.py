#!/usr/bin/env python3
"""probe_cross_dict.py — 跨域词典合并实验（用户指引：参考通用领域成熟资源）

输入：/tmp/sadict（ppzhenghua/SentimentAnalysisDictionary：NTUSD/清华/HowNet）
      现词典（config/dicts/*）+ 100 条 no_hit 样本（data-out/nohit100.json）
方法：三源合并去重（仅收长度≥2 的正面/负面词）→ 作为扩展 h 类并入扫描 → 对 no_hit 样本试算命中提升
输出：stdout 摘要 + data-out/cross_dict_exp.json
⚠️ 实验专用，不写生产词典；是否正式合并由裁决决定。
"""
import io
import json
import os
import re
import sys

SADIR = '/tmp/sadict/SentimentAnalysisDictionary-main'
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, '..', 'config')
NOHIT = os.path.join(HERE, '..', 'data-out', 'nohit100.json')
OUT = os.path.join(HERE, '..', 'data-out', 'cross_dict_exp.json')

sys.path.insert(0, os.path.join(HERE, '..', 'jobs'))
from m5_sentiment_v0 import load_dicts  # noqa: E402

CJK = re.compile(r'[\u4e00-\u9fff]')


def decode_smart(path):
    b = open(path, 'rb').read()
    best = None
    for enc in ('utf-16', 'gbk', 'utf-8-sig', 'utf-8'):
        try:
            t = b.decode(enc)
        except Exception:
            continue
        n = max(len(t), 1)
        cjk = len(CJK.findall(t))
        score = cjk / n
        if cjk > 50 and score > 0.3:
            return t
        if best is None or score > best[1]:
            best = (t, score)
    return best[0] if best else b.decode('utf-8', errors='ignore')


def load_words(path):
    t = decode_smart(path)
    words = []
    for ln in t.splitlines():
        w = ln.strip()
        if not w or w.startswith('#') or '\t' in ln:
            continue  # 表头行（如 '中文正面评价词语\t3730'）
        words.append(w)
    return words


def main():
    sources = {}
    files = {
        'ntusd_pos': f'{SADIR}/台湾大学NTUSD简体中文情感词典/NTUSD_positive_simplified.txt',
        'ntusd_neg': f'{SADIR}/台湾大学NTUSD简体中文情感词典/NTUSD_negative_simplified.txt',
        'ths_pos': f'{SADIR}/清华大学李军中文褒贬义词典/tsinghua_positive_gb.txt',
        'ths_neg': f'{SADIR}/清华大学李军中文褒贬义词典/tsinghua.negative.gb.txt',
        'hn_pos_eval': f'{SADIR}/知网Hownet情感词典/正面评价词语（中文）.txt',
        'hn_neg_eval': f'{SADIR}/知网Hownet情感词典/负面评价词语（中文）.txt',
        'hn_pos_emo': f'{SADIR}/知网Hownet情感词典/正面情感词语（中文）.txt',
        'hn_neg_emo': f'{SADIR}/知网Hownet情感词典/负面情感词语（中文）.txt',
    }
    for k, p in files.items():
        if os.path.exists(p):
            ws = load_words(p)
            sources[k] = ws
            print(f'  {k}: {len(ws)} 词（样例: {ws[:3]}）')
        else:
            print(f'  {k}: MISSING {p}')

    # 合并（±1）
    ext = {}
    for k, ws in sources.items():
        pol = 1 if ('pos' in k) else -1
        for w in ws:
            if len(w) < 2 or CJK.search(w) is None:
                continue
            ext.setdefault(w, pol)
    print(f'三源合并（≥2 字）: {len(ext)} 词')

    # 排除现有词典已有的
    g, h, e, anchors, neg, deg = load_dicts(CONFIG)
    existing = set(g) | set(h) | set(e)
    ext_new = {w: p for w, p in ext.items() if w not in existing}
    print(f'去重后新增: {len(ext_new)} 词（现有词典 {len(existing)} 词）')

    # 扫描器（镜像 m5 逻辑 + ext 词表）
    lookup = {}
    for w, (pol, st) in g.items():
        lookup[w] = ('g', pol, st)
    for w, (pol, wt) in h.items():
        lookup[w] = ('h', pol, wt)
    for w in ext_new:
        lookup.setdefault(w, ('ext', ext_new[w], 3.0))
    first = {w[0] for w in lookup}
    maxlen = max(len(w) for w in lookup)

    def scan(text):
        hits = []
        i, n = 0, len(text)
        while i < n:
            if text[i] not in first:
                i += 1
                continue
            for L in range(min(maxlen, n - i), 0, -1):
                v = lookup.get(text[i:i + L])
                if v is not None:
                    hits.append((text[i:i + L], v[0]))
                    i += L
                    break
            else:
                i += 1
        return hits

    rows = json.load(open(NOHIT, encoding='utf-8'))
    n_hit_new, n_hit_ext, examples = 0, 0, []
    for r in rows:
        text = r['text']
        base_hits = [x for x in scan(text) if x[1] != 'ext']
        ext_hits = [x for x in scan(text) if x[1] == 'ext']
        if base_hits:
            continue  # 意外（应为 0）：不算
        if ext_hits:
            n_hit_new += 1
            examples.append({'text': text, 'hits': ext_hits[:4]})
        n_hit_ext += len(ext_hits)

    summary = {
        'n_sources_total': sum(len(v) for v in sources.values()),
        'n_merged_ge2': len(ext), 'n_new_after_dedup': len(ext_new),
        'n_nohit_samples': len(rows),
        'n_samples_newly_hit': n_hit_new,
        'hit_lift_pct': round(100 * n_hit_new / max(len(rows), 1), 1),
        'examples': examples[:60],
        'note': '跨域合并实验（NTUSD+清华+HowNet）——若正式合并是否执行由裁决决定',
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'examples'},
                     ensure_ascii=False, indent=1))
    print('例子：')
    for x in examples[:60]:
        print(' ', x['text'][:20], '->', x['hits'])


if __name__ == '__main__':
    main()
