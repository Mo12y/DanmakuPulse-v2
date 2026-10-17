#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''test_v3_page_purpose.py — 页面功能标题 + 受众标签 + 措辞规范

判据（本轮唯一，压过奥卡姆剃刀）：每个模块必须答得出"它帮谁，做成了哪件事"。
为了让这句话不是纸面合规，它落到页面上：每个页面一个 p.pagesub
写「名词化的功能标题 + 受众标签」，缺了就红。

⛔ 措辞规范（2026-10-16 用户指出后新增）：禁止自指与问句。
  第一版把副标题写成「这一页解决什么 · 给 UP 主：把一支视频的节奏跟同类比一遍…」——
  那是"关于本页的说明"，不是"这个功能的名称"。
  系统界面的措辞应该是名词化的功能标题，不复述自己、不向用户提问。
  所以 _BANNED 里的词在可见文案与内联脚本里一律不许出现。

用法：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python -m pytest tests/ -q
'''
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, 'serve', 'static')

_TAG = re.compile(r'<[^>]+>')

# 自指 + 问句类措辞：界面上不该出现（要的是"功能名称"，不是"本页说明"）
_BANNED = ('这一页', '本页', '这个页面', '该页面',
           '解决什么', '做什么', '怎么做', '说明了什么', '说明什么',
           '帮谁', '做成了什么', '干什么', '用来做什么', '有什么用')

# 不带受众标签的页面（登录页对所有人开放，谈不上受众）
_NO_AUD = {'login.html'}


def _pages():
    return [n for n in sorted(os.listdir(STATIC)) if n.endswith('.html')]


def _pagesub_raw(txt):
    m = re.search(r'<p[^>]*class="pagesub"[^>]*>(.*?)</p>', txt, re.S)
    return m.group(1) if m else None


def _strip(txt):
    return _TAG.sub('', txt).strip()


def test_every_page_has_noun_form_title():
    """每个页面都有 p.pagesub：名词化功能标题 + 受众标签。"""
    bad = []
    for n in _pages():
        txt = io.open(os.path.join(STATIC, n), encoding='utf-8').read()
        raw = _pagesub_raw(txt)
        if raw is None:
            bad.append('%s：没有 p.pagesub 功能标题' % n)
            continue
        title = _strip(raw)
        if len(title) < 6:
            bad.append('%s：标题太短（"%s"）—— 那不像一个功能名称' % (n, title))
        if n not in _NO_AUD and '面向' not in raw:
            bad.append('%s：缺受众标签 <span class="aud">面向 X</span>' % n)
        if not re.search(r'<b>.+?</b>', raw, re.S):
            bad.append('%s：标题没用 <b> 包起来（样式依赖它）' % n)
    assert not bad, ('这些页面的功能标题不合规：' + chr(10) + '  ' + (chr(10) + '  ').join(bad))


def test_no_self_referential_or_question_wording():
    """界面文案不许自指、不许向用户提问（"这一页解决什么"这类）。"""
    bad = []
    for n in _pages():
        for ln, line in enumerate(io.open(os.path.join(STATIC, n), encoding='utf-8').read().split(chr(10)), 1):
            code = line.split('//')[0]          # 跳过 JS 行注释（给开发者看的）
            code = re.sub(r'/\*.*?\*/', '', code)   # 跳过块注释
            if not code.strip() or code.strip() in ('/*', '*', '*/'):
                continue
            for w in _BANNED:
                if w in code:
                    bad.append('%s:%d 用了「%s」：%s' % (n, ln, w, code.strip()[:70]))
    assert not bad, ('界面文案里出现了自指/问句类措辞 —— 改成名词化的功能标题，' + chr(10) +
                     '受众用 span.aud 标注「面向 X」，不要复述"这一页在做什么"：' + chr(10) + '  '
                     + (chr(10) + '  ').join(bad[:20]))


def test_no_self_referential_wording_in_backend_copy():
    """后端生成的界面文案也不许自指/提问。

    2026-10-16 实测发现：页面副标题改完之后，spark.html / track.html 的说明文字里
    仍有「这页回答的不是…」「所以这一页给的是…」—— 因为**界面文案有一半是在
    serve/*.py 里拼出来再发给前端的**，只扫 .html 抓不到。所以这条扫 .py 的
    **字符串字面量**（跳过注释与 docstring，那些是给开发者看的）。
    """
    bad = []
    for n in sorted(os.listdir(os.path.join(ROOT, 'serve'))):
        if not n.endswith('.py'):
            continue
        path = os.path.join(ROOT, 'serve', n)
        in_doc = False
        for ln, line in enumerate(io.open(path, encoding='utf-8').read().split(chr(10)), 1):
            s = line.strip()
            # docstring 开/关（同一行成对出现不算切换）
            toggles = s.count('"""') + s.count("'''")
            if toggles % 2 == 1:
                in_doc = not in_doc
                continue
            if in_doc:
                continue
            code = line.split('#')[0]
            if '#' in line and code.strip() == '':
                continue
            for w in _BANNED:
                if w in code:
                    bad.append('%s:%d 用了「%s」：%s' % (n, ln, w, code.strip()[:70]))
    assert not bad, ('后端拼出来的界面文案里出现了自指/问句类措辞：' + chr(10) + '  '
                     + (chr(10) + '  ').join(bad[:20]))