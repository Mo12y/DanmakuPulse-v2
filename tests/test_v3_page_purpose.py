#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''test_v3_page_purpose.py — 每个页面都必须写清「这一页解决什么」

本轮唯一判据（压过奥卡姆剃刀）：每个模块必须能一句话回答
**"它帮谁，做成了哪件事（一个动作）？"** —— 答不出就是没达到"有效"，重做或不做。
⛔ 不许用"它提供了信息 / 它有数据支撑"代替这个答案。
（v2 的毛病不是模块多，是模块答不出这句话 —— 一堆可视化大屏。）

为了让这句话**不是纸面合规**，本轮把它做成两件事：
  ① 每个页面上放一句可见的副标题（`<p class="pagesub">`）；
  ② 这条守卫 —— 缺了就红。于是它在页面上看得见、在机器上查得到。

判据刻意不止"有没有这句话"，还要"像不像真话"：
  · 每个 serve/static/*.html 都有 `<p class="pagesub">`；
  · 文案要含「这一页解决什么」；
  · 后半句（去掉「这一页解决什么」标点、以及「给 X：」这类受众前缀后）**至少 8 个字** ——
    否则有人写一句「这一页解决什么：待补充」就能凑过守卫。

用法：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python -m pytest tests/ -q
'''
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, 'serve', 'static')

_TAG = re.compile(r'<[^>]+>')
# 受众前缀写成「给 UP 主：」「给 MCN/运营：」这类；去掉它再判断答案够不够长
_AUDIENCE = re.compile(r'给[^：:\n]{1,14}[：:]')
_SEP = re.compile(r'^[：:\s·\-—>＞]+')


def _pages():
    return [n for n in sorted(os.listdir(STATIC)) if n.endswith('.html')]


def _pagesub(txt):
    m = re.search(r'<p[^>]*class="pagesub"[^>]*>(.*?)</p>', txt, re.S)
    return _TAG.sub('', m.group(1)).strip() if m else None


def test_every_page_states_what_it_solves():
    bad = []
    pages = _pages()
    assert pages, 'serve/static 下一个 .html 都没有？'
    for n in pages:
        txt = io.open(os.path.join(STATIC, n), encoding='utf-8').read()
        sub = _pagesub(txt)
        if sub is None:
            bad.append('%s：没有 <p class="pagesub"> 副标题' % n)
            continue
        if '这一页解决什么' not in sub:
            bad.append('%s：副标题没写「这一页解决什么」（实际："%s"）' % (n, sub[:36]))
            continue
        rest = sub.split('这一页解决什么', 1)[1]
        rest = _SEP.sub('', rest)
        rest = _AUDIENCE.sub('', rest).strip()
        if len(rest) < 8:
            bad.append('%s：答案太短（"%s"）—— 那不像"帮谁做成了哪件事"' % (n, rest))
    assert not bad, ('这些页面没写清「这一页解决什么」（本轮判据：答不出 = 重做或不做的那个答案）：\n  '
                     + '\n  '.join(bad))
