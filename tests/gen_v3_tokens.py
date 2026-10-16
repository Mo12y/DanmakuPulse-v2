#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_v3_tokens.py — 从 DESIGN.md §2 生成 v3.css 的 token 区块

为什么用生成而不是手抄：DESIGN.md §2 有 48 个 token，手抄必然漂移，
而「手写规范 + 手抄实现」正是 test_design_tokens.py 存在的原因。
这里让两边**同源**：改 DESIGN.md → 重跑本脚本 → 测试自然绿。

用法：~/danmaku-venv/bin/python tests/gen_v3_tokens.py
"""
import io
import os
import re
import sys

V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN = os.path.join(os.path.dirname(V2), 'danmaku', 'DESIGN.md')
CSS = os.path.join(V2, 'serve', 'static', 'v3.css')
BEGIN = '/* @tokens-begin */'
END = '/* @tokens-end */'


def design_path():
    p = os.environ.get('DESIGN_MD', DESIGN)
    if not os.path.exists(p):
        sys.exit('找不到 DESIGN.md：%s（可用 DESIGN_MD= 指定）' % p)
    return p


def parse(path):
    """返回 (两值 token 列表[(名, 浅, 深)], 单值 token 列表[(名, 值)])。"""
    md = io.open(path, encoding='utf-8').read()
    m = re.search(r'## 2\. 颜色 token.*?(?=\n## 3\.)', md, re.S)
    assert m, 'DESIGN.md 找不到 §2 颜色 token 段'
    body = m.group(0)
    # 按小节切：2.1–2.6 是「token|浅|深|用途」，2.7 是「token|值|用途」
    chunks = re.split(r'\n### 2\.(\d)', body)
    pairs, singles = [], []
    for i in range(1, len(chunks), 2):
        sec, text = chunks[i], chunks[i + 1]
        for line in text.splitlines():
            cells = re.findall(r'\x60([^\x60]+)\x60', line)
            if len(cells) < 2 or not cells[0].startswith('--'):
                continue
            if sec == '7':
                singles.append((cells[0], cells[1]))
            else:
                pairs.append((cells[0], cells[1], cells[2] if len(cells) > 2 else cells[1]))
    return pairs, singles


def render(pairs, singles):
    out = [BEGIN]
    out.append('/* 由 tests/gen_v3_tokens.py 从 DESIGN.md §2 生成 —— 请勿手改，改 DESIGN.md 后重跑 */')
    out.append(':root {')
    for n, light, _dark in pairs:
        out.append('  %s: %s;' % (n, light))
    for n, v in singles:
        out.append('  %s: %s;' % (n, v))
    out.append('}')
    out.append('')
    out.append('[data-theme="dark"] {')
    for n, _light, dark in pairs:
        out.append('  %s: %s;' % (n, dark))
    out.append('}')
    out.append(END)
    return '\n'.join(out)


def main():
    pairs, singles = parse(design_path())
    css = io.open(CSS, encoding='utf-8').read()
    i, j = css.index(BEGIN), css.index(END) + len(END)
    new = css[:i] + render(pairs, singles) + css[j:]
    io.open(CSS, 'w', encoding='utf-8').write(new)
    print('token：两值 %d 个 + 单值 %d 个 = %d 个' % (len(pairs), len(singles), len(pairs) + len(singles)))
    print('已写入 %s' % CSS)


if __name__ == '__main__':
    main()
