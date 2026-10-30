#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''test_v3_ai_highlights.py — AI 解读切到「精彩时刻」数据源（2026-10-20）

为什么钉这条：
  旧实现把**旧诊断卡**（verdict / position / deviations / benchmarks）喂给模型，
  页面主视图早已换成精彩时刻检测 —— 模型会写出页面上根本看不到的结论。
  切源之后最容易犯的三个错，都钉成断言：
    ① 忘了换源（还在 diagnose()）或忘了 404（退回旧诊断）；
    ② 把 highlights 自己的 `benchmark`（单数）当成旧卡的 `benchmarks`（复数）一起删掉；
    ③ 缓存键用 meta.generated_at —— 它每次请求 now() 生成，缓存永不命中（等于没有缓存）。

判据全部对着**源码**查（宿主 venv 没有 psycopg，不能 import serve.ai_report）。

用法：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python -m pytest tests/ -q
'''
import ast
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI = os.path.join(ROOT, 'serve', 'ai_report.py')
APP = os.path.join(ROOT, 'serve', 'app.py')


def _src(path):
    return io.open(path, encoding='utf-8').read()


def _func_body(txt, name, stop):
    """截出 def name(...) 到下一个 def stop( 之间的源码（不含签名前的注释）。"""
    i = txt.index('def %s(' % name)
    j = txt.index('def %s(' % stop)
    return txt[i:j]


def test_prompt_version_exists():
    t = _src(AI)
    assert "PROMPT_VERSION = 'hl-ai-v4'" in t, \
        '需要 PROMPT_VERSION（换口径/换提示词就换版本，旧缓存不能被命中）'


def test_system_forbids_field_names_and_numbers():
    """v3 约束：只许引时间点，禁字段名（pctl/score/n_repeat）与统计数字。
    来源：2026-10-20 真实冒烟发现模型把「pctl 82.98」念进正文（用户拍板改提示词）。"""
    t = _src(AI)
    assert '禁止出现任何字段名' in t and 'pctl' in t and 'n_repeat' in t, \
        'SYSTEM 第 9 条必须点名禁字段名（pctl/score/n_repeat）'
    assert '统计数字' in t and '不要念原始数值' in t, '必须写明禁统计数字'
    assert 'mm:ss' in t and '不要写「第 1350–1360 秒」' in t, \
        '时间必须要求 mm:ss（2026-10-22：不许让用户自己除 60）'


def test_payload_is_highlights_not_old_card():
    t = _src(AI)
    body = _func_body(t, 'highlights_payload', 'make_snap_id')
    for old in ("'verdict'", "'position'", "'deviations'", "'benchmarks'"):
        assert old not in body, 'payload 里不应再出现旧诊断字段 %s' % old
    assert "'benchmark'" in body, \
        'highlights 自己的 benchmark（单数，分区共振对标）必须保留 —— 别和旧卡 benchmarks 删混'
    assert 'diag_payload' not in t, 'diag_payload 已改名 highlights_payload'


def test_report_route_uses_highlights_and_404():
    t = _src(APP)
    i = t.index("@app.post('/api/v3/ai/report')")
    j = t.index("@app.get('/api/v3/deviation')")
    body = t[i:j]
    assert 'hl.card(' in body, 'AI 解读的数据源必须是 highlights.card()'
    assert 'diagnosis.diagnose' not in body, '不得退回旧诊断卡'
    assert '不在精彩时刻检测快照里' in body, '不在快照里必须 404 并说清，不静默换数据源'


def test_cache_key_not_generated_at():
    t = _src(AI)
    body = _func_body(t, 'make_snap_id', 'build_messages')
    ret = [ln for ln in body.splitlines() if ln.strip().startswith('return')]
    assert ret, 'make_snap_id 必须有显式 return'
    for ln in ret:
        assert 'generated_at' not in ln, \
            '缓存键不能用 meta.generated_at（每次请求 now()，永不命中）'
    for k in ('PROMPT_VERSION', 'batch_date', 'engine_version', 'dict_version'):
        assert k in body, '缓存键应含 %s' % k


def test_system_prompt_titles():
    t = _src(AI)
    assert '【可参考的观察点】' in t, '输出标题必须是【可参考的观察点】'
    assert '【可执行建议】' not in t, '旧标题【可执行建议】应已删除'
    assert '不判定情绪' in t and '不评价内容质量' in t, 'SYSTEM 必须写明两条边界'


def test_split_uses_new_title():
    t = _src(AI)
    body = _func_body(t, '_split', 'generate')
    assert '【可参考的观察点】' in body, '_split 要按新标题切分'
    assert '可执行建议' not in body, '_split 不应再认旧标题'


def test_banned_words_specific_assertions_only():
    t = _src(AI)
    tree = ast.parse(t)
    words = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'BANNED_WORDS':
            words = tuple(ast.literal_eval(node.value))
    assert words, 'BANNED_WORDS 必须仍是显式常量'
    for w in ('感人', '好笑', '愤怒', '燃', '节奏健康', '没有明显短板',
              '内容最好', '名场面', '爆款', '一定会火'):
        assert w in words, '禁用词缺少具体断言词「%s」' % w
    # 边界名词不能进禁用词表：模型复述"不判定情绪 / 不评价内容质量"时会被误杀
    for border in ('内容质量', '情绪类型', '情绪判断', '情绪'):
        assert border not in words, '「%s」是边界名词，不该进禁用词表' % border


def test_error_copy_has_no_cost():
    t = _src(AI)
    assert '本次费用' not in t, '报错文案不得包含费用金额（费用只写服务端日志）'
    assert re.search(r"raise (ValueError|RuntimeError)\('模型输出命中禁用词", t), \
        '命中禁用词仍需拒绝展示（文案可改，行为不变）'


def test_state_route_crops_for_normal_user():
    t = _src(APP)
    i = t.index("@app.get('/api/v3/ai/state')")
    j = t.index("@app.post('/api/v3/ai/limits')")
    body = t[i:j]
    assert "role') != 'admin'" in body, 'ai/state 必须按角色裁剪（普通用户不拿用量/费用/模型名）'
    assert "'enabled'" in body and "'configured'" in body, '普通用户只回 {enabled, configured}'


def test_frontend_has_no_ai_usage_surface():
    """2026-10-28 诊断页重建：AI 按钮与面板已删除。

    后端 AI 端点与按角色裁剪仍由 test_state_route_crops_for_normal_user 守住；
    这里反过来钉住前端不再出现任何用量/费用入口，避免旧面板被顺手带回来。
    """
    html = io.open(os.path.join(ROOT, 'serve', 'static', 'diagnosis.html'),
                   encoding='utf-8').read()
    assert 'aiUsageLine' not in html
    assert 'AI 解读' not in html
    assert 'spent_cny' not in html
