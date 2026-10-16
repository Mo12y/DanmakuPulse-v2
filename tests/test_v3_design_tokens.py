#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_v3_design_tokens.py — v3 前端的设计规范守卫

和 danmaku 仓库的 tests/test_design_tokens.py 同源同思路，对象换成 v3 这套静态页。

钉住四件事：
  1. v3.css 的 @tokens 区块 == DESIGN.md §2（48 个 token，逐字一致，不多不少）
  2. serve/static 下的 html/js 里不得出现裸 hex 色（dpVar fallback 与 token 定义除外）
  3. dpVar('--x', '#fallback') 的 fallback 必须等于 --x 的浅色值
     （否则 CSS 未加载时会闪一个错色）
  4. DESIGN.md §1.2 反 AI 味硬约束的可判定项：
     C1 无 transition:all ｜ C5 有 prefers-reduced-motion 降级
     §1#11 无「outline:none 而无替代」

用法：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python -m pytest tests/ -q
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, 'serve', 'static')
DESIGN = os.environ.get('DESIGN_MD', os.path.join(os.path.dirname(ROOT), 'danmaku', 'DESIGN.md'))


def _read(p):
    with io.open(p, encoding='utf-8') as f:
        return f.read()


def _design_tokens():
    """DESIGN.md §2 -> (两值 token {名:(浅,深)}, 单值 token {名:值})"""
    md = _read(DESIGN)
    m = re.search(r'## 2\. 颜色 token.*?(?=\n## 3\.)', md, re.S)
    assert m, 'DESIGN.md 找不到 §2 颜色 token 段'
    chunks = re.split(r'\n### 2\.(\d)', m.group(0))
    pairs, singles = {}, {}
    for i in range(1, len(chunks), 2):
        sec, text = chunks[i], chunks[i + 1]
        for line in text.splitlines():
            cells = re.findall(r'\x60([^\x60]+)\x60', line)
            if len(cells) < 2 or not cells[0].startswith('--'):
                continue
            if sec == '7':
                singles[cells[0]] = cells[1]
            else:
                pairs[cells[0]] = (cells[1], cells[2] if len(cells) > 2 else cells[1])
    return pairs, singles


def _css_tokens():
    css = _read(os.path.join(STATIC, 'v3.css'))
    blk = css[css.index('/* @tokens-begin */'):css.index('/* @tokens-end */')]
    m_light = re.search(r':root\s*\{(.*?)\}', blk, re.S)
    m_dark = re.search(r'\[data-theme="dark"\]\s*\{(.*?)\}', blk, re.S)
    assert m_light and m_dark, 'v3.css token 区块缺 :root 或 [data-theme="dark"]'

    def parse(b):
        return {k: re.sub(r'\s+', ' ', v.strip())
                for k, v in re.findall(r'(--[a-z0-9-]+)\s*:\s*([^;]+);', b)}
    return parse(m_light.group(1)), parse(m_dark.group(1))


def _static_files(exts):
    return [n for n in sorted(os.listdir(STATIC)) if os.path.splitext(n)[1] in exts]


_CSS_COMMENT = re.compile(r'/\*.*?\*/', re.S)
_HTML_COMMENT = re.compile(r'<!--.*?-->', re.S)
# 只剥「整行」的 // 注释：行内的 // 会撞上 URL（http://），不能盲剥
_LINE_COMMENT = re.compile(r'(?m)^[ \t]*//.*$')


def _strip_comments(txt, ext):
    """剥注释后再扫 —— 否则文档里写的示例（dpVar('--token', ...) / transition:all）
    会被当成真代码误判。测试要拦的是代码，不是说明。"""
    if ext == '.html':
        txt = _HTML_COMMENT.sub('', txt)
        return _CSS_COMMENT.sub('', txt)
    if ext == '.css':
        return _CSS_COMMENT.sub('', txt)
    return _LINE_COMMENT.sub('', _CSS_COMMENT.sub('', txt))


# ── 1. token 与 DESIGN.md 一致 ───────────────────────────────

def test_tokens_match_design_md_exactly():
    pairs, singles = _design_tokens()
    light, dark = _css_tokens()
    want_light = dict({k: v[0] for k, v in pairs.items()}, **singles)
    want_dark = {k: v[1] for k, v in pairs.items()}
    assert set(light) == set(want_light), (
        'v3.css 浅色 token 集合与 DESIGN.md 不一致：多 %s / 少 %s'
        % (sorted(set(light) - set(want_light)), sorted(set(want_light) - set(light))))
    bad = {k: (light[k], want_light[k]) for k in want_light if light[k].lower() != want_light[k].lower()}
    assert not bad, 'v3.css 浅色 token 值不符（css -> design）：%s' % bad
    bad_d = {k: (dark.get(k), v[1]) for k, v in pairs.items() if dark.get(k, '').lower() != v[1].lower()}
    assert not bad_d, 'v3.css 深色 token 值不符（css -> design）：%s' % bad_d


def test_token_count_is_48():
    pairs, singles = _design_tokens()
    assert len(pairs) + len(singles) == 48, 'DESIGN.md §2 的 token 数变了（现 %d）' % (len(pairs) + len(singles))


# ── 2/3. 裸色与 dpVar fallback ───────────────────────────────

_DPVAR = re.compile(r"dpVar\(\s*'(--[a-z0-9-]+)'\s*,\s*'([^']*)'\s*\)")
_HEX = re.compile(r'#[0-9a-fA-F]{3,8}\b')
HEX_ALLOW = {'v3chart.js'}   # dpVar 的 fallback（DESIGN.md §1#1 明确允许，第 3 条核对它）


def test_no_raw_hex_outside_allowlist():
    light, _ = _css_tokens()
    allowed = {v.lower() for v in light.values()}
    problems = []
    for n in _static_files({'.html', '.js'}):
        txt = _strip_comments(_read(os.path.join(STATIC, n)), os.path.splitext(n)[1])
        hits = [h for h in _HEX.findall(txt) if h.lower() not in allowed]
        if hits and n not in HEX_ALLOW:
            problems.append('%s: %s' % (n, sorted(set(hits))))
    assert not problems, '出现裸 hex 色（应改用 var(--token) 或 dpVar）：\n  ' + '\n  '.join(problems)


def test_dpvar_fallbacks_equal_light_values():
    light, _ = _css_tokens()
    wrong = []
    for n in _static_files({'.js'}):
        txt = _strip_comments(_read(os.path.join(STATIC, n)), os.path.splitext(n)[1])
        for name, fb in _DPVAR.findall(txt):
            if name not in light:
                wrong.append('%s: dpVar 用了不存在的 token %s' % (n, name))
            elif fb.lower() != light[name].lower():
                wrong.append('%s: dpVar(%s) fallback=%s 但 token 浅色值=%s' % (n, name, fb, light[name]))
    assert not wrong, '\n  '.join(wrong)


# ── 4. 反 AI 味可判定项 ──────────────────────────────────────

def test_c1_no_transition_all():
    bad = []
    for n in _static_files({'.css', '.html'}):
        txt = _strip_comments(_read(os.path.join(STATIC, n)), os.path.splitext(n)[1])
        for m in re.finditer(r'transition\s*:\s*all\b', txt):
            bad.append('%s@%d' % (n, txt[:m.start()].count('\n') + 1))
    assert not bad, 'DESIGN.md §1.2.3 C1 禁止 transition: all —— 命中：%s' % bad


def test_c5_has_reduced_motion():
    css = _read(os.path.join(STATIC, 'v3.css'))
    assert re.search(r'@media\s*\(prefers-reduced-motion:\s*reduce\)', css), \
        'DESIGN.md §1.2.3 C5：v3.css 缺 reduced-motion 降级块'


def test_no_outline_none_without_focus_visible():
    css = _read(os.path.join(STATIC, 'v3.css'))
    assert ':focus-visible' in css, 'DESIGN.md §1#11：缺 :focus-visible 焦点样式'
    # 允许「:focus:not(:focus-visible){outline:none}」这一种写法（鼠标点击不闪轮廓的标准做法）
    assert ':focus:not(:focus-visible)' in css
    for m in re.finditer(r'outline\s*:\s*none', css):
        seg = css[max(0, m.start() - 140):m.start()]
        assert ':focus:not(:focus-visible)' in seg, 'DESIGN.md §1#11：出现无替代的 outline:none'


def test_no_pure_black_canvas():
    """§1#2：禁止纯黑底 / 冷灰白画布。允许 #FFFFFF 只作为 --on-accent（强调底上的文字）。"""
    light, dark = _css_tokens()
    for name, v in list(light.items()) + list(dark.items()):
        if v.upper() in ('#000', '#000000'):
            raise AssertionError('§1#2：token %s 是纯黑' % name)
        if v.upper() in ('#FFFFFF', '#FFF') and name != '--on-accent':
            raise AssertionError('§1#2：token %s 是纯白（只允许 --on-accent）' % name)


# ── 5. 图注不得描述已经不存在的颜色 ─────────────────────────

_STALE_COLOR_WORDS = ('橙色', '橘色', '橙黄')


def test_no_stale_color_words_in_copy():
    """配色改过之后，图注里"上图橙色虚线"这类描述会**骗人** —— 比没有图注更糟。
    现行数据色只有：--primary 蓝 / --accent 粉（该看的地方）/ --chart-axis 灰（参照系）。
    所以用户可见文案里不该再出现橙色系词。"""
    bad = []
    for n in _static_files({'.html'}):
        txt = _strip_comments(_read(os.path.join(STATIC, n)), '.html')
        for w in _STALE_COLOR_WORDS:
            if w in txt:
                bad.append('%s: %s' % (n, w))
    assert not bad, '图注/文案里出现旧配色词（现行配色无橙色系）：%s' % bad
