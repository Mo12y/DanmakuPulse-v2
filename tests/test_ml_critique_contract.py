#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''test_ml_critique_contract.py — 弱监督质疑泛化（G2）的契约守卫

为什么钉（2026-10-20）：
  G2 的产物会直接进页面（虚线质疑段），一旦某处静默退回"只有词典版"，
  页面会显示"两条线一样"，看起来像功能没做 —— 所以把链路钉成断言：
    ① 作业参数不许手填：K/elasticNet/regParam 走网格 + 5 折 CV；阈值由正样本分位数算
    ② 落表带 method/ml_score：DWS → ADS → API 三处列名一致
    ③ 前端必须标注"虚线 = 弱监督泛化（召回更高、可能误报）"（诚实边界不能只写在报告里）

判据全部对着源码查（宿主 venv 不装 pyspark/psycopg，不能 import 作业）。
用法：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python -m pytest tests/ -q
'''
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(*parts):
    return io.open(os.path.join(ROOT, *parts), encoding='utf-8').read()


def test_job_exists_and_grid_is_fixed():
    t = _src('jobs', 'm2_l5_mlcritique.py')
    assert 'SEED = 42' in t
    assert 'NUM_FEATURES = 2 ** 16' in t, 'HashingTF 维度必须固定 2^16'
    assert 'K_CANDIDATES = (2, 5, 10)' in t, '正负比 K 必须是候选集合，由 CV 决定'
    assert 'ELASTIC_CANDIDATES' in t and 'REG_CANDIDATES' in t, '缺少 LR 网格'
    assert re.search(r'LogisticRegression\([^)]*maxIter=MAX_ITER', t, re.S), \
        '必须用 pyspark.ml LogisticRegression(maxIter=200)'
    assert 'family=\'binomial\'' in t
    assert 'N_FOLDS = 5' in t, '必须 5 折交叉验证'
    assert 'accuracy' in t and 'F1' in t, '评估指标必须是 F1（不许 accuracy）'


def test_threshold_and_holdout_are_computed():
    t = _src('jobs', 'm2_l5_mlcritique.py')
    assert 'RECALL_TARGET = 0.95' in t
    assert 'k_keep' in t and 'probs[' in t, '阈值必须由正样本分位数算，不许手填'
    assert 'recall_heldout' in t and 'pos_held' in t, '必须有跨词典 held-out 召回'
    assert 'recall_zero_baseline' in t, '报告必须同时给零基线（全判负）'


def test_table_columns_carry_method():
    ads = _src('jobs', 'ads_highlights.py')
    assert "'method'" in ads and "'ml_score'" in ads, 'ADS 导出必须带 method / ml_score'
    m4 = _src('jobs', 'm2_l4_highlights.py')
    assert "F.lit('dict')" in m4, '词典行必须显式写 method=\'dict\''
    hl = _src('serve', 'highlights.py')
    assert "'method'" in hl and 'n_ml_segments' in hl and 'n_dict_segments' in hl, \
        'API 载荷必须给 method 与两种段数'


def test_frontend_marks_weak_supervision():
    html = _src('serve', 'static', 'diagnosis.html')
    assert '虚线 = 弱监督泛化（召回更高、可能误报）' in html, '图例必须写清可能误报'
    assert "s.method === 'ml'" in html, '前端要按 method 区分实线/虚线'
    assert '（弱监督）' in html and '（词典）' in html, 'chip / 面板要标来源'
