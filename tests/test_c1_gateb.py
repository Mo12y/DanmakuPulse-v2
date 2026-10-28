#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_c1_gateb.py — C1 链路测试：Gate B 纯函数 + Spark 产物报告 + 协议冻结守卫

运行：~/danmaku-venv/bin/python -m pytest tests/test_c1_gateb.py -q
纯函数部分不依赖任何数据；产物部分读 repo 内的运行报告 JSON（由 Spark 作业落盘）。
"""
import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


gb = _load("exp49_gateb", "experiments/exp49_c1_gateb.py")

# ---------------- 边界匹配 ----------------


def test_match_exact():
    r = gb.match_boundaries([0, 10, 20], [0, 10, 20])
    assert r['hits'] == 3 and r['precision'] == 1.0 and r['recall'] == 1.0 and r['f1'] == 1.0
    assert r['errors'] == [0, 0, 0]


def test_match_tolerance_edge():
    r = gb.match_boundaries([10], [11])          # Δ=1 桶 = 60s 容差内
    assert r['hits'] == 1 and r['errors'] == [1]
    r2 = gb.match_boundaries([10], [12])         # Δ=2 桶 → 不命中
    assert r2['hits'] == 0 and r2['recall'] == 0.0 and r2['precision'] == 0.0


def test_match_one_to_one():
    """两个真值边界挤在同一个预测边界旁：只允许命中一次（1:1 约束）"""
    r = gb.match_boundaries([10, 11], [11])
    assert r['hits'] == 1 and r['n_truth'] == 2 and r['n_pred'] == 1
    assert r['recall'] == 0.5 and r['precision'] == 1.0


def test_match_empty():
    assert gb.match_boundaries([], [1, 2])['f1'] == 0.0
    assert gb.match_boundaries([1, 2], [])['f1'] == 0.0


def test_boundaries_of_segments():
    assert gb.boundaries_of_segments([(0, 4), (5, 9)], 10) == [0, 5, 10]
    assert gb.boundaries_of_segments([(0, 9)], 10) == [0, 10]  # 协议 §6：含视频右边界（末段 end+1）
    assert gb.boundaries_of_segments([], 10) == []


# ---------------- 基线 1：等频 ----------------


def test_equal_freq():
    assert gb.seg_equal_freq(10, 3) == [(0, 2), (3, 6), (7, 9)]
    assert gb.seg_equal_freq(10, 1) == [(0, 9)]
    assert gb.seg_equal_freq(3, 5) == [(0, 0), (1, 1), (2, 2)]      # k>n 时收缩到 k=n
    assert gb.seg_equal_freq(10, 0) == []
    assert gb.seg_equal_freq(0, 3) == []


# ---------------- 基线 2：单变点 ----------------


def test_single_changepoint_clear_split():
    cp = gb.seg_single_changepoint([1, 1, 1, 9, 9, 9])
    assert cp[0][1] == 2 and cp[1][0] == 3


def test_single_changepoint_short_and_flat():
    assert gb.seg_single_changepoint([5]) == [(0, 0)]
    assert gb.seg_single_changepoint([]) == []
    cp = gb.seg_single_changepoint([4, 4, 4, 4])       # 无差异也要返回合法两段/整段
    assert cp and cp[0][0] == 0 and cp[-1][1] == 3


# ---------------- 候选：2σ 突起 ----------------


def test_event_candidates_spike():
    ev = gb.seg_event_candidates([0, 0, 0, 50, 0, 0, 0, 0])
    assert ev == [(3, 3)]


def test_event_candidates_flat_no_event():
    assert gb.seg_event_candidates([2, 2, 2, 2, 2, 2]) == []
    assert gb.seg_event_candidates([]) == []


def test_event_candidates_two_peaks_merged_or_split():
    # 两峰相距 3 桶 → 两个独立事件（合并条件是间隔 ≤2 桶内的连片突起）
    ev = gb.seg_event_candidates([0, 0, 40, 0, 0, 0, 40, 0, 0, 0])
    assert len(ev) == 2 and ev[0][0] == 2 and ev[1][0] == 6


# ---------------- 汇总 ----------------


def test_evaluate_micro():
    res = gb.evaluate({'A': [0, 10], 'B': [5]}, {'A': [0, 10], 'B': [5]})
    assert res['micro']['f1'] == 1.0 and res['micro']['TP'] == 3
    res2 = gb.evaluate({'A': [0, 10]}, {'A': [0]})
    assert res2['micro']['TP'] == 1 and res2['micro']['FN'] == 1 and res2['micro']['FP'] == 0
    assert res2['micro']['recall'] == 0.5 and res2['micro']['precision'] == 1.0


# ---------------- Spark 产物报告（任务 A 验收） ----------------


def test_rhythm60s_report():
    p = os.path.join(REPO, 'data-out/rhythm60s_report.json')
    if not os.path.exists(p):
        pytest.skip('rhythm60s_report.json 未生成（先跑 jobs/m2_l1_rhythm60s.py）')
    r = json.load(open(p, encoding='utf-8'))
    assert r['input']['dup_groups_actual_key'] == 0            # 去重键 (batch,bvid,cid,row_id) 零重复
    assert r['input']['rows'] > 3_000_000                      # 与 DWD 行数对账
    assert r['output']['meta_units'] == 2948
    assert r['output']['c1_eligible_units'] > 0
    assert r['output']['c1_eligible_videos'] >= 200            # 40 号线索：216 支（现为 254，批次数变多）
    assert r['engine_version'] == 'c1-v1'
    assert 'fetched_at' in r['rules']['post_snapshot']        # 口径写死：缺失≠零


def test_c1_annotation_materials():
    d = os.path.join(REPO, 'data-out/c1')
    if not os.path.exists(os.path.join(d, 'manifest.json')):
        pytest.skip('c1 标注材料未生成（先跑 jobs/c1_prepare_annotation.py）')
    m = json.load(open(os.path.join(d, 'manifest.json'), encoding='utf-8'))
    assert m['sampled'] == 30 and m['seed'] == 20261028
    assert m['category_set'] == ['L', 'R', 'H', 'B', 'D', 'X']
    hdr = m['sheet_columns']
    assert hdr == ['bvid', 'seg_id', 'start_bucket', 'end_bucket', 'category', 'confidence', 'note']


def test_c1_protocol_frozen():
    """协议守卫：v1.0 冻结稿必须存在，且含诚实披露与门槛——防止被静默改掉"""
    p = os.path.join(REPO, 'docs/c1_annotation_protocol.md')
    assert os.path.exists(p), '协议文件缺失'
    txt = open(p, encoding='utf-8').read()
    assert 'v1.0' in txt and '冻结' in txt
    assert '不得根据算法分段结果倒推' in txt or '按算法结果倒推' in txt
    assert '0.70' in txt          # Gate A 门槛
    assert '15s' in txt and '60%' in txt   # Gate B 预设门槛
