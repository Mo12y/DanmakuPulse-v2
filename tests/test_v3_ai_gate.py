#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''test_v3_ai_gate.py — AI 报告解读的付费闸：**有配置且默认关闭**

为什么钉这条（铁律·付费服务先设闸）：
  2026-09-22 实测失控过一次，约 **¥30**（3 小时 56 分、1024+ 次 LLM 调用）。
  "默认关闭"和"三条硬上限"如果只写在文档里就等于没有 —— 所以钉成守卫。
  触发条件不是"看起来很多"，而是"**会计费**"。

判据（全部对着源码查；不需要数据库）：
  ① 三条闸有常量且值不超规格：单日 <= 200、累计 <= ¥10、单次 max_tokens <= 600
  ② 默认关闭：DEFAULT_ENABLED 为 False，且建表语句里 enabled 的初始值就是 false
  ③ 密钥只从 env 读：不得出现疑似密钥字面量
  ④ 缓存键含"快照"：同一支视频在**不同快照**下是不同的解读（规格："同 bvid+快照只解读一次"）
  ⑤ 超限会**自动停**：源码里有"把开关置回 false"的路径
  ⑥ 输出过禁用词校验，命中**拒绝展示**（不许静默换成别的文案）

用法：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python -m pytest tests/ -q
'''
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'serve', 'ai_report.py')


def _src():
    return io.open(SRC, encoding='utf-8').read()


def _const(txt, name):
    m = re.search(r'^%s\s*=\s*([0-9.]+)' % re.escape(name), txt, re.M)
    assert m, 'serve/ai_report.py 里找不到常量 %s（三条闸必须是显式常量，别散在代码里）' % name
    return float(m.group(1))


def test_three_gates_in_spec():
    t = _src()
    assert _const(t, 'DAY_LIMIT') <= 200, '单日上限必须 <= 200 次'
    assert _const(t, 'BUDGET_CNY') <= 10.0, '累计上限必须 <= ¥10'
    assert _const(t, 'MAX_TOKENS') <= 600, '单次输出上限必须 <= 600 token'


def test_default_disabled():
    t = _src()
    assert re.search(r'^DEFAULT_ENABLED\s*=\s*False\b', t, re.M), '默认必须是关闭'
    assert re.search(r'enabled\s+boolean\s+NOT NULL\s+DEFAULT\s+false', t), \
        '建表语句里 enabled 的初始值必须是 false（不能靠"记得别开"）'


def test_key_from_env_never_hardcoded():
    t = _src()
    assert "os.environ.get('LLM_API_KEY')" in t, '密钥必须从 env 读（不硬编码）'
    bad = re.findall(r"['\"](?:sk-[A-Za-z0-9_\-]{8,}|[A-Za-z0-9]{32,})['\"]", t)
    assert not bad, '源码里出现疑似密钥字面量：%s' % bad


def test_cache_key_includes_snapshot():
    t = _src()
    assert 'snap_id' in t and re.search(r'PRIMARY KEY \(bvid, snap_id\)', t), \
        '缓存必须按 (bvid, 快照) 唯一 —— 规格原文："同 bvid+快照只解读一次"'


def test_autostop_on_limit():
    t = _src()
    assert 'set_enabled_off' in t, '需要一个"把开关关掉"的函数（超限自动停）'
    assert re.search(r'UPDATE ai_config SET enabled = false', t), '超限必须真的把开关置回关闭'


def test_banned_words_refuse_to_display():
    t = _src()
    assert 'BANNED_WORDS' in t and 'check_output' in t, '必须有禁用词校验'
    assert '拒绝展示' in t, '命中禁用词必须拒绝展示（不许静默换成别的文案）'
