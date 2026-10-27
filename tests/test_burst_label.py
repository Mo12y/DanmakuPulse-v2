"""exp35 标签函数单测（H=6；T1–T7 移植自协议 §4.4，边界值按 H=6 重算）。

运行：~/danmaku-venv/bin/python -m pytest tests/test_burst_label.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jobs"))

import m2_l6_burst_risk as M  # noqa: E402


def seq(c, bvid="BV1test", batch="20260902", duration=None):
    c = np.asarray(c, dtype=np.int64)
    dur = len(c) * M.B if duration is None else duration
    return M.Sequence(bvid=bvid, batch=batch, duration=float(dur), c=c)


def labels(c, **kw):
    return M.label_points(seq(c), **kw)


# ── T1–T7：阈值语义 ───────────────────────────────────────────────
def test_t1_lambda_zero_threshold_three_inclusive():
    p = labels([0] * 12 + [3] + [0] * 11)[0]
    assert p.lam == 0.0
    assert p.thr == 3.0          # max(3, 0+2√0) = 3
    assert p.y == 1              # 3 >= 3 命中（含等号）


def test_t2_lambda_zero_below_threshold():
    p = labels([0] * 12 + [2] + [0] * 11)[0]
    assert p.y == 0


def test_t3_threshold_equals_three_for_lambda_one():
    p = labels([1] * 12 + [10] + [0] * 11)[0]
    assert p.lam == 1.0
    assert p.thr == pytest.approx(3.0)
    assert p.y == 1


def test_t4_exactly_at_threshold():
    p = labels([1] * 12 + [3] + [0] * 11)[0]
    assert p.thr == pytest.approx(3.0)
    assert p.y == 1


def test_t5_last_future_bucket_only():
    # 协议原式 [1]*12+[2]*11+[3] 的尖峰落在下标 23，那是 H=12 的第 12 个未来桶；
    # H=6 下"最后一个未来桶"是下标 17，故按同构重排（12 历史 + 5 平 + 1 尖 + 6 尾）。
    p = labels([1] * 12 + [2] * 5 + [3] + [0] * 6)[0]
    assert p.win_end == 18
    assert p.y == 1


def test_t6_last_window_bucket_in_range():
    pts = labels([1] * 12 + [3] + [0] * 10 + [3])
    assert pts[0].y == 1
    assert pts[0].win_start == 12 and pts[0].win_end == 18
    assert [p.t for p in pts] == [12, 18]     # K=24


def test_t7_high_then_collapse_is_negative():
    p = labels([5] * 12 + [0] * 12)[0]
    assert p.thr == pytest.approx(5 + 2 * math.sqrt(5))
    assert p.y == 0


def test_all_zero_sequence_is_negative():
    assert labels([0] * 24)[0].y == 0


# ── 分桶：floor + 左闭右开 ─────────────────────────────────────────
def test_floor_bucketing_left_closed_right_open():
    s = M.build_sequence("BV1a", "20260902",
                         [0.0, 4.999, 5.0, 9.999, 10.0],
                         ["d1", "d2", "d3", "d4", "d5"], duration=30)
    assert s.c[0] == 2 and s.c[1] == 2 and s.c[2] == 1
    assert s.K == 6


def test_no_np_histogram_used_in_source():
    src = open(os.path.join(HERE, "..", "jobs", "m2_l6_burst_risk.py"), encoding="utf-8").read()
    assert "np.histogram" not in src and "numpy.histogram" not in src


# ── 决策点：间距、边界、最后一点 ───────────────────────────────────
def test_decision_points_spacing():
    assert list(M.decision_times(24)) == [12, 18]
    assert list(M.decision_times(30)) == [12, 18, 24]


def test_no_legal_point_when_too_short():
    assert M.decision_times(17).size == 0        # 12+6=18 > 17
    assert labels([0] * 17) == []


def test_exactly_one_legal_point():
    ts = M.decision_times(18)
    assert list(ts) == [12]


def test_last_legal_point_kept():
    for K in (18, 23, 24, 29, 30, 41):
        ts = M.decision_times(K)
        assert ts.size > 0
        last = int(ts[-1])
        assert last + M.H <= K                  # 闭区间上界
        assert last + M.H + M.H > K             # 再走一步就越界


def test_threshold_computed_once_at_decision_time():
    p = labels([2] * 12 + [9, 9, 9, 9, 9, 9])[0]
    assert p.thr == pytest.approx(M.threshold(2.0))
    assert p.win_start == p.t and p.win_end == p.t + M.H


# ── 非法时间 / 视频时长边界 ────────────────────────────────────────
def test_invalid_times_dropped_with_counts():
    s = M.build_sequence("BV1a", "20260902",
                         [np.nan, -1.0, 999.0, 1.0],
                         ["d1", "d2", "d3", "d4"], duration=120)
    assert s.audit["dropped_nan"] == 1
    assert s.audit["dropped_negative"] == 1
    assert s.audit["dropped_beyond_duration"] == 1
    assert s.audit["rows_kept"] == 1


def test_offset_equal_duration_clipped_to_last_bucket():
    s = M.build_sequence("BV1a", "20260902", [120.0], ["d1"], duration=120)
    assert s.K == 24 and s.c[23] == 1 and s.audit["clipped_to_last_bucket"] == 1


def test_usable_requires_min_duration():
    short = M.build_sequence("BV1a", "20260902", [1.0], ["d1"], duration=100)
    ok = M.build_sequence("BV1b", "20260902", [1.0], ["d1"], duration=120)
    assert short.usable is False and ok.usable is True


def test_illegal_duration_rejected():
    with pytest.raises(M.ProtocolError):
        M.build_sequence("BV1a", "20260902", [1.0], ["d1"], duration=0)


# ── dmid 去重：真复读必须保留 ──────────────────────────────────────
def test_dmid_dedup_keeps_first_copy():
    s = M.build_sequence("BV1a", "20260902",
                         [10.0, 10.0], ["same", "same"], duration=120)
    assert s.audit["dropped_duplicate_dmid"] == 1
    assert int(s.c.sum()) == 1


def test_true_repeat_same_text_same_time_different_dmid_kept():
    s = M.build_sequence("BV1a", "20260902",
                         [10.0, 10.0], ["d1", "d2"], duration=120)
    assert s.audit["dropped_duplicate_dmid"] == 0
    assert int(s.c.sum()) == 2


# ── 性质测试 ───────────────────────────────────────────────────────
def test_property_matches_any_semantics():
    rng = np.random.default_rng(0)
    for _ in range(200):
        K = int(rng.integers(18, 60))
        c = rng.integers(0, 12, size=K)
        s = M.Sequence("BV1x", "20260902", K * 5.0, c)
        for p in M.label_points(s):
            thr = M.threshold(float(c[p.t - M.PRE:p.t].mean()))
            expect = int(any(int(c[k]) >= thr for k in range(p.t, p.t + M.H)))
            assert p.y == expect and p.thr == pytest.approx(thr)


def test_property_future_independence():
    rng = np.random.default_rng(1)
    K = 30
    c = rng.integers(0, 8, size=K)
    base = [p.y for p in M.label_points(M.Sequence("BV1x", "20260902", K * 5.0, c))]
    c2 = c.copy()
    c2[12 + M.H:] = 99                      # 只改 t+H 之后
    after = [p.y for p in M.label_points(M.Sequence("BV1x", "20260902", K * 5.0, c2))]
    assert base[0] == after[0]


def test_labels_do_not_use_whole_video_future():
    """长序列尾部任意改动，不得改变前面决策点的标签。"""
    K = 60
    c = np.zeros(K, dtype=np.int64)
    c[12:18] = 5
    a = M.label_points(M.Sequence("BV1x", "20260902", K * 5.0, c))
    c2 = c.copy()
    c2[24:] = 50
    b = M.label_points(M.Sequence("BV1x", "20260902", K * 5.0, c2))
    assert [p.y for p in a if p.t == 12] == [p.y for p in b if p.t == 12]
