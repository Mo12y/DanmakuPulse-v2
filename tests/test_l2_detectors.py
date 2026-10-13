#!/usr/bin/env python3
"""
test_l2_detectors.py — L2 合成信号单测（08 §13 算法单元测试；B3.1 验收「各检测器合成信号单测过」）

运行：~/danmaku-venv/bin/python -m pytest tests/test_l2_detectors.py -q
说明：全部为纯函数级测试（不依赖 Spark 集群）；参数取 config/params.yaml 的现场值。
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, '..', 'jobs')
sys.path.insert(0, JOBS)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(JOBS, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import l2_common  # noqa: E402  （jobs/ 已在 path）

m2e = _load('m2_l2_events', 'm2_l2_events.py')
m2lag = _load('m2_l2_lag', 'm2_l2_lag.py')
m2nov = _load('m2_l2_novelty', 'm2_l2_novelty.py')
m2sync = _load('m2_l2_sync', 'm2_l2_sync.py')

P = {'bucket_sec': 5, 'd1_quantile': 0.70, 'd1_slope': 20, 'd2_k': 2.0, 'd3_gap_min': 10,
     'flux_win': 3, 'flux_delta': 1.5, 'fusion_w': [0.3, 0.3, 0.2, 0.2], 'fusion_theta': 0.5,
     'merge_gap': 3}


def spike_series(L=61, base=5.0):
    c = np.full(L, base)
    c[30:34] = [80.0, 150.0, 160.0, 90.0]
    return c


# ── platform_peaks ──

def test_platform_peaks_basic():
    assert l2_common.platform_peaks([0, 1, 0]) == [1]
    assert l2_common.platform_peaks([0, 2, 2, 0]) == [1]      # 平台取中点
    assert l2_common.platform_peaks([2, 1, 0]) == []          # 端点不算峰
    assert l2_common.platform_peaks([0, 0, 0]) == []


# ── 检测器：平线无事件 / 尖峰有事件 ──

def test_flat_series_no_events():
    flat = np.full(50, 7.0)
    assert m2e.detect_d1(flat, P) == []
    assert m2e.detect_d2(flat, P) == []
    assert m2e.detect_d3(flat, P) == []
    assert m2e.detect_d4(flat, P) == []


def test_d1_spike():
    segs = m2e.detect_d1(spike_series(), P)
    assert segs, 'D1 应命中尖峰'
    assert all(s['start'] <= 33 and s['end'] >= 30 for s in segs)


def test_d2_gate_and_slope():
    segs = m2e.detect_d2(spike_series(), P)
    assert segs and min(s['start'] for s in segs) <= 31 <= max(s['end'] for s in segs)
    # 密度门限：整体 5 的序列（med·3=15）不命中
    assert m2e.detect_d2(np.full(40, 5.0), P) == []


def test_d3_constraints():
    segs = m2e.detect_d3(spike_series(), P)
    assert len(segs) == 1 and 30 <= segs[0]['start'] <= 33
    assert segs[0]['raw'] >= 2.0  # 突出度
    # 峰距：两峰相距 5 < gap 10 → 只保留高者
    c = np.full(60, 1.0)
    c[20], c[25] = 100.0, 200.0
    segs2 = m2e.detect_d3(c, P)
    assert len(segs2) == 1 and segs2[0]['start'] == 25


def test_d4_onset():
    c = np.zeros(60)
    c[30:] = 20.0
    segs = m2e.detect_d4(c, P)
    assert segs and any(abs(s['start'] - 30) <= 2 for s in segs)


# ── 融合 ──

def test_fusion_multi_method_merge_and_theta():
    a = [{'start': 10, 'end': 12, 'raw': 30.0, 'score': 1.0}]
    b = [{'start': 11, 'end': 13, 'raw': 9.0, 'score': 1.0}]
    evs = m2e.fuse_video({'d1': a, 'd2': b, 'd3': [], 'd4': []},
                         {'d1': 0.3, 'd2': 0.3, 'd3': 0.2, 'd4': 0.2}, 0.5)
    assert len(evs) == 1 and evs[0]['start'] == 10 and evs[0]['end'] == 13
    assert abs(evs[0]['score_fused'] - 0.6) < 1e-12 and evs[0]['n_methods'] == 2
    # 单法最高 0.3 < θ=0.5 → 截断
    assert m2e.fuse_video({'d1': a, 'd2': [], 'd3': [], 'd4': []},
                          {'d1': 0.3, 'd2': 0.3, 'd3': 0.2, 'd4': 0.2}, 0.5) == []


def test_run_detectors_end_to_end():
    evs = m2e.run_detectors(spike_series(), {'detect': P})
    assert evs, '合成尖峰应产出 ≥1 融合事件'
    assert all(e['score_fused'] >= 0.5 for e in evs)
    assert [e['event_id'] for e in evs] == list(range(len(evs)))


# ── 同步性（纯函数部分：抽样与熵；确定性=同种子同结果）──

def test_sync_metrics_basic_and_deterministic():
    bc = np.array([5] * 10 + [6] * 10 + [0] * 5, dtype=np.int64)   # 25 弹幕，窗口 [5,6] → N=20
    dc = np.array([0] * 15 + [1] * 10, dtype=np.int64)
    r1 = m2sync.sync_metrics(bc, dc, 5, 6, 7, 2, np.random.RandomState(42), 200)
    r2 = m2sync.sync_metrics(bc, dc, 5, 6, 7, 2, np.random.RandomState(42), 200)
    sync, p, hp, ht, N = r1
    assert N == 20
    assert 0.0 <= sync <= 1.0 and 0.0 <= p <= 1.0
    assert r1 == r2, '同种子必须同结果（幂等）'


def test_sync_metrics_full_window_p_is_one():
    # 窗口覆盖全部弹幕（N==M）→ 置换集合同观测，p=1.0
    bc = np.array([3, 3, 4, 4, 5], dtype=np.int64)
    dc = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    sync, p, hp, ht, N = m2sync.sync_metrics(bc, dc, 0, 6, 7, 2, np.random.RandomState(7), 100)
    assert N == 5 and p == 1.0


# ── Novelty 分段 ──

def test_foote_segments_boundary():
    c = np.r_[np.full(40, 2.0), np.full(40, 50.0)]
    segs = m2nov.foote_segments(c, 10, 8)
    # 阶跃处应检出边界；窗口化 SSM 下边界可能成对出现（Step 两侧各一峰），落点 ±10 内即算检出
    assert len(segs) >= 2, f'{segs}'
    assert any(abs(s[0] - 40) <= 10 for s in segs[1:]), f'边界应贴近 40：{segs}'
    assert segs[0][0] == 0 and segs[-1][1] == 79


def test_foote_short_series_single_segment():
    assert m2nov.foote_segments(np.full(6, 1.0), 10, 8) == [[0, 5]]
    assert m2nov.foote_segments(np.array([]), 10, 8) == []


# ── 滞后校正确认 ──

def test_per_video_lag_delta():
    evs = [(0, 5, 10)]                       # 桶 5..10 → center=(5+10+1)/2*5=40s
    ans = [('aft', 40.0, 8), ('aft', 50.0, 10), ('aft', 60.0, 12),   # 偏移 0/10/20
           ('fore', 20.0, 4)]
    out = m2lag.per_video_lag((('bv', 'd'), (evs, ans)), win=6, bucket_sec=5)
    assert len(out) == 1
    _, _, _, delta, n, n_aft, n_fore, center = out[0]
    assert abs(center - 40.0) < 1e-9 and n == 4 and n_aft == 3 and n_fore == 1
    assert abs(delta - 10.0) < 1e-9
    # 无 aft 锚 → δ NULL
    out2 = m2lag.per_video_lag((('bv', 'd'), (evs, [('fore', 20.0, 4)])), win=6, bucket_sec=5)
    assert out2[0][3] is None and out2[0][4] == 1
    # 窗外锚不计入
    out3 = m2lag.per_video_lag((('bv', 'd'), (evs, [('aft', 999.0, 200)])), win=6, bucket_sec=5)
    assert out3[0][3] is None and out3[0][4] == 0

# ── 作业级幂等守卫（2026-10-13 实测踩坑：finalize 二次运行 COLUMN_ALREADY_EXISTS）──

def test_finalize_zcols_dedup_on_rerun():
    """m2_l2_finalize 必须对已含 *_z_part 的输入表可重跑（08 §0.1 幂等）。

    缺陷形态：src_cols 取自 fp.columns，若 fp 已含上一轮写入的 *_z_part 列，
    再 select(*src_cols, *z_cols) 会造出重名列 → COLUMN_ALREADY_EXISTS。
    本测试静态断言「组装 select 前已把 z 列从 src_cols 剔除」。
    """
    import re
    p = os.path.join(JOBS, 'm2_l2_finalize.py')
    with open(p, encoding='utf-8') as f:
        src = f.read()
    m = re.search(r'^\s*src_cols\s*=\s*\[(.*?)\]', src, re.M | re.S)
    assert m, 'finalize 未找到 src_cols 定义'
    expr = m.group(1)
    assert 'z_cols' in expr, (
        f'src_cols 未排除 z_cols（重跑会重名列）：{expr!r}')
