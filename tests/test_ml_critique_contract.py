#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''test_ml_critique_contract.py — 弱监督质疑泛化（G2）的契约守卫

为什么钉（2026-10-20）：
  G2 的产物会直接进页面（虚线质疑段），一旦某处静默退回"只有词典版"，
  页面会显示"两条线一样"，看起来像功能没做 —— 所以把链路钉成断言：
    ① 作业参数不许手填：K/elasticNet/regParam 走网格 + 5 折 CV；阈值由正样本分位数算
    ② 落表带 method/ml_score：DWS → ADS → API 三处列名一致
    ③ 2026-10-28 诊断页重建后主视图不再展开质疑层；API 契约不变，
       这里反向钉住旧虚线图例没有被带回前端。

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


def test_frontend_does_not_reintroduce_weak_supervision_layer():
    """2026-10-28 诊断页重建：主视图不再展开质疑段。

    API 仍必须带 method / ml_score（由 test_table_columns_carry_method 守住）；
    这里反向钉住旧虚线图例与来源标签没有随旧页面被带回来。
    """
    html = _src('serve', 'static', 'diagnosis.html')
    assert '弱监督' not in html
    assert '质疑段' not in html
    assert "s.method === 'ml'" not in html
