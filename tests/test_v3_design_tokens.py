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


def test_c1_no_transition_shorthand():
    """transition 简写同样是 all。

    CSS 的 transition 简写里，第一个时间值之前的那个词才是属性名；
    只写 transition:.15s 时属性缺省为 all —— 与 transition: all .15s 完全等价。

    这条是补盲区：原守卫只匹配字面量 transition: all，而本项目实测有 7 处写成
    transition:.15s / .12s（诊断卡 / 总览 / 收藏页的内联样式），守卫全绿却真实违规
    （2026-10-16 发现）。教训与 A1 的 S>1% 那次同型：判据有多个条件时只查一个。
    """
    bad = []
    for n in _static_files({'.css', '.html'}):
        txt = _strip_comments(_read(os.path.join(STATIC, n)), os.path.splitext(n)[1])
        for m in re.finditer(r'transition\s*:\s*[.0-9]', txt):
            bad.append('%s@%d' % (n, txt[:m.start()].count(chr(10)) + 1))
    assert not bad, ('DESIGN.md §1.2.3 C1：transition 简写（首值是时间）等价于 transition:all，'
                     '必须写明具体属性 —— 命中：%s' % bad)


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


def test_dockerfile_copies_every_serve_module():
    """Dockerfile 必须把所有 serve/*.py 拷进镜像。

    2026-10-16 实测踩过：新增 live_diag.py 但 Dockerfile 用显式文件清单、漏登记 ->
    容器起来就 ModuleNotFoundError 反复重启。这类漏项不会自己暴露（本地 import 一切正常），
    所以钉一条守卫：serve/ 下每个 .py 都要么被通配符覆盖，要么出现在 COPY 行里。
    """
    import glob
    import re
    df = io.open(os.path.join(ROOT, 'serve', 'Dockerfile'), encoding='utf-8').read()
    copy_lines = [l for l in df.split(chr(10)) if l.strip().upper().startswith('COPY')]
    wildcard = any(re.search(r'COPY\s+\*\.py', l) for l in copy_lines)
    mods = [os.path.basename(p) for p in glob.glob(os.path.join(ROOT, 'serve', '*.py'))]
    if wildcard:
        return
    joined = ' '.join(copy_lines)
    missing = [m for m in mods if m not in joined]
    assert not missing, ('serve/Dockerfile 没把这些模块拷进镜像：%s —— 容器会 ModuleNotFoundError。'
                         '要么加进 COPY 行，要么改用 COPY *.py ./' % missing)


def test_compose_fastapi_has_collector_mounts():
    """采集控制台的三个挂载必须在 **fastapi** 服务里。

    2026-10-16 实测踩过：用『mem_limit + restart』这种通用锚点改 compose，
    匹配到了文件里更早的 **postgres** 段 —— 结果内存上限和 user 加到了数据库上，
    而 fastapi 原封不动。postgres 当时没被重建所以没炸，下次 make up 就会坏。
    教训：改 compose 必须**按服务块定位**，不能用跨服务都出现的行当锚点。
    """
    c = io.open(os.path.join(ROOT, 'compose', 'docker-compose.b0.yml'), encoding='utf-8').read()
    ls = c.split(chr(10))
    starts = [(i, l.split(':')[0].strip()) for i, l in enumerate(ls)
              if re.match(r'^  [a-z][a-z0-9-]*:$', l)]
    for k, (i, name) in enumerate(starts):
        if name != 'fastapi':
            continue
        j = starts[k + 1][0] if k + 1 < len(starts) else len(ls)
        blk = chr(10).join(ls[i:j])
        for want in ('danmaku/collect:/collect', 'data/cli_collected:/data-out', 'danmaku/logs:/logs'):
            assert want in blk, 'fastapi 服务块缺少挂载 %s —— 采集控制台会报脚本没挂进来' % want
        return
    raise AssertionError('compose 里找不到 fastapi 服务')


def test_all_inline_scripts_compile_as_js():
    """每个页面内联的 <script> 必须能被真正的 JS 引擎**编译通过**。

    2026-10-16 实测踩过**两次**，两次都只表现为「页面一直显示正在加载」：
      ① 按行区间 splice 删函数，留下一个孤立的 }  → SyntaxError
      ② 插入代码时又写了一遍 const b（和上面已有的重名）
         → SyntaxError: invalid redefinition of lexical identifier
    而 pytest 全绿、HTTP 200、curl 拿到的 HTML 完全正常 —— **只有人眼看截图才发现**。

    为什么用 quickjs 而不是 esprima：重复声明是 **JS 早期错误**，esprima 是解析器、不查
    （已实测确认它放行）。quickjs 是真引擎，两类都拦得住。
    判据：**只有 SyntaxError 算失败** —— 页面脚本必然会摸 DOM，缺 document 抛的
    ReferenceError 恰恰证明「编译过了、开始执行了」。

    注：曾试图用 Python 扫「同层重复 const」补 esprima 的缺口，但花括号计数处理不了
    对象字面量与 for...of 作用域，在**干净的代码上就误报 7 处** —— 会喊狼来了的守卫
    等于没有，故弃用，改真引擎。
    """
    import re
    import quickjs
    bad = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'serve', 'static'))):
        if not name.endswith('.html'):
            continue
        src = io.open(os.path.join(ROOT, 'serve', 'static', name), encoding='utf-8').read()
        for i, blk in enumerate(re.findall(r'<script>(.*?)</script>', src, re.S)):
            if not blk.strip():
                continue
            # ⚠️ 每块**新建上下文**：共用一个 Context 会让上一块的 $ / var 泄漏到下一块，
            # 于是报出假的 redeclaration（实测踩过）。
            try:
                quickjs.Context().eval(blk)
            except Exception as e:
                # ⚠️ quickjs 抛的是 JSException，消息**以 SyntaxError 开头**，
                # 不能写 except SyntaxError（那是 Python 的类型，匹配不上，会被下面的兜底吞掉）。
                msg = str(e).split(chr(10))[0]
                if msg.startswith('SyntaxError'):
                    bad.append('%s 第 %d 个 <script>: %s' % (name, i + 1, msg[:110]))
                # 其余（ReferenceError/TypeError）= 编译通过、开始执行了，正常
    assert not bad, ('这些页面的内联 JS 编译不过 —— 整段脚本会失败、页面只表现为停在「正在加载」：'
                     + chr(10) + chr(10).join(bad))



def test_static_js_files_compile():
    """独立的 .js 文件也要能编译通过。

    2026-10-16 实测踩过：删 v3nav.js 里一个导航条目时，正则把这一组的 ] }, 一起删了，
    **整个 GROUPS 数组语法坏掉 -> 侧栏完全不渲染**（页面其它部分正常，所以只看内容看不出来）。
    而 test_all_inline_scripts_compile_as_js 只扫 .html 的内联脚本，**扫不到独立 .js** ——
    这就是盲区。补上这条。
    """
    import quickjs
    bad = []
    for name in sorted(os.listdir(os.path.join(ROOT, 'serve', 'static'))):
        if not name.endswith('.js'):
            continue
        src = io.open(os.path.join(ROOT, 'serve', 'static', name), encoding='utf-8').read()
        try:
            quickjs.Context().eval(src)
        except Exception as e:
            msg = str(e).split(chr(10))[0]
            if msg.startswith('SyntaxError'):
                bad.append('%s: %s' % (name, msg[:110]))
    assert not bad, ('这些 .js 文件编译不过（不会报错，只会让页面静默少一块）：' + chr(10) + chr(10).join(bad))
