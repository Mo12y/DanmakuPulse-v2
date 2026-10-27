"""exp35 特征单测：lambda / slope / since / censored / prog / duration / ent + 白名单。

运行：~/danmaku-venv/bin/python -m pytest tests/test_burst_features.py -q
"""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jobs"))

import m2_l6_burst_risk as M  # noqa: E402


def seq(c, bvid="BV1f", batch="20260902"):
    c = np.asarray(c, dtype=np.int64)
    return M.Sequence(bvid=bvid, batch=batch, duration=len(c) * M.B, c=c)


def test_lambda_is_history_mean():
    c = [0] * 12 + [0] * 20
    c[0:12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    f = M.feature_row(seq(c), 12)
    assert f["lam"] == pytest.approx(np.mean(c[0:12]))
    assert f["log1p_lam"] == pytest.approx(math.log1p(np.mean(c[0:12])))


def test_slope_is_log_difference_of_halves():
    c = [0] * 40
    c[0:6] = [2] * 6
    c[6:12] = [8] * 6
    f = M.feature_row(seq(c), 12)
    assert f["slope"] == pytest.approx(math.log1p(8) - math.log1p(2))


def test_slope_has_no_epsilon_explosion():
    c = [0] * 40
    c[6:12] = [1000] * 6    # 前半段为 0
    f = M.feature_row(seq(c), 12)
    assert math.isfinite(f["slope"])
    assert abs(f["slope"]) < 50


def test_slope_bounded_on_random_sequences():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(300):
        K = int(rng.integers(24, 80))
        c = rng.integers(0, 5000, size=K)
        s = M.Sequence("BV1x", "20260902", K * 5.0, c)
        for t in M.decision_times(K):
            worst = max(worst, abs(M.feature_row(s, int(t))["slope"]))
    assert worst < 50


def test_source_has_no_epsilon_in_slope():
    """slope 的计算行不得出现 epsilon（否则分母趋零时会产生 1e6~1e9 的伪值）。"""
    src = open(os.path.join(HERE, "..", "jobs", "m2_l6_burst_risk.py"), encoding="utf-8").read()
    lines = [ln for ln in src.splitlines() if "slope=" in ln]
    assert lines, "未找到 slope 计算行"
    for ln in lines:
        assert "1e-" not in ln, ln


# ── ent ────────────────────────────────────────────────────────────
def test_ent_uniform_is_one():
    assert M.entropy12([1] * 12) == pytest.approx(1.0)


def test_ent_single_bucket_is_zero():
    assert M.entropy12([5] + [0] * 11) == 0.0


def test_ent_all_zero_is_zero():
    assert M.entropy12([0] * 12) == 0.0


def test_ent_two_buckets_normalised_by_12():
    # [1,1,0*10] → ln2/ln12 = 0.2790（按 12 桶归一，不是按非零桶数）
    assert M.entropy12([1, 1] + [0] * 10) == pytest.approx(0.2790, abs=1e-3)


def test_ent_in_unit_interval():
    rng = np.random.default_rng(4)
    for _ in range(100):
        w = rng.integers(0, 50, size=12)
        assert 0.0 <= M.entropy12(w) <= 1.0 + 1e-12


# ── since / censored ───────────────────────────────────────────────
def test_since_one_when_previous_bucket_burst():
    c = np.zeros(36, dtype=np.int64)
    c[17] = 5                      # λ_17 = 0 → thr 3 → 5 >= 3
    since, cens = M.since_censored(seq(c), 18)
    assert (since, cens) == (1.0, 0.0)


def test_since_counts_distance():
    c = np.zeros(36, dtype=np.int64)
    c[14] = 9
    since, cens = M.since_censored(seq(c), 18)
    assert (since, cens) == (4.0, 0.0)


def test_first_decision_point_is_censored():
    c = np.zeros(30, dtype=np.int64)
    c[11] = 9                      # 决策点 t=12 之前没有可观察事件
    assert M.since_censored(seq(c), 12) == (12.0, 1.0)


def test_events_outside_observable_window_are_censored():
    c = np.zeros(36, dtype=np.int64)
    c[5] = 9                       # t=18 的可观察窗口是 k ∈ [12,18)
    assert M.since_censored(seq(c), 18) == (12.0, 1.0)


def test_since_uses_same_event_form_as_label():
    """e_k 用各自前 60 s 基线；高强度段之后 ρ 高也不算事件。"""
    c = np.zeros(40, dtype=np.int64)
    c[0:24] = 20                   # 前 24 桶恒定 20 → 未来看历史时 thr=20+2√20>20
    since, cens = M.since_censored(seq(c), 30)
    assert cens == 1.0 and since == 12.0


# ── prog / duration ────────────────────────────────────────────────
def test_prog_is_t_over_K():
    s = seq(np.zeros(40, dtype=np.int64))
    for t in M.decision_times(40):
        assert M.feature_row(s, int(t))["prog"] == pytest.approx(int(t) / s.K)


def test_duration_feature_is_log1p():
    s = M.Sequence("BV1f", "20260902", 600.0, np.zeros(120, dtype=np.int64))
    assert M.feature_row(s, 12)["log1p_duration"] == pytest.approx(math.log1p(600.0))


# ── 特征不使用未来 ─────────────────────────────────────────────────
def test_features_do_not_read_future_buckets():
    rng = np.random.default_rng(5)
    K = 40
    c = rng.integers(0, 10, size=K)
    s1 = M.Sequence("BV1x", "20260902", K * 5.0, c)
    c2 = c.copy()
    c2[12:] = 777
    s2 = M.Sequence("BV1x", "20260902", K * 5.0, c2)
    assert M.feature_row(s1, 12) == M.feature_row(s2, 12)


def test_all_features_causal_on_every_point():
    rng = np.random.default_rng(6)
    K = 48
    c = rng.integers(0, 9, size=K)
    s = M.Sequence("BV1x", "20260902", K * 5.0, c)
    for t in M.decision_times(K):
        t = int(t)
        a = M.feature_row(s, t)
        cc = c.copy()
        cc[t:] = 12345
        b = M.feature_row(M.Sequence("BV1x", "20260902", K * 5.0, cc), t)
        assert a == b, f"t={t} 的特征依赖了 t 之后的数据"


# ── 白名单 / 黑名单 ────────────────────────────────────────────────
def test_m2_feature_set_is_exact():
    assert set(M.FEATURES_M2) == {"log1p_lam", "slope", "since", "censored", "prog",
                                  "log1p_duration"}
    assert len(M.FEATURES_M2) == 6


def test_m3_is_m2_plus_ent_only():
    assert set(M.FEATURES_M3) - set(M.FEATURES_M2) == {"ent"}


def test_unapproved_models_keep_interface_only():
    for name in ("M0", "M4"):
        with pytest.raises(M.ProtocolError):
            M.model_features(name)


def test_banned_feature_rejected():
    for bad in ("n_dm", "y", "thr", "future_max", "play"):
        with pytest.raises(M.ProtocolError):
            M.assert_whitelist((bad,))


def test_inject_ndm_must_fail():
    rows = [dict(log1p_lam=0.0, slope=0.0, since=12.0, censored=1.0, prog=0.1,
                 log1p_duration=5.0, ent=0.0, n_dm=100.0)]
    with pytest.raises(M.ProtocolError):
        M.select_features(rows, ("log1p_lam", "n_dm"))


def test_unknown_feature_rejected():
    with pytest.raises(M.ProtocolError):
        M.assert_whitelist(("log1p_lam", "my_new_feature"))


def test_non_finite_feature_rejected():
    rows = [dict(log1p_lam=float("nan"), slope=0.0, since=12.0, censored=1.0,
                 prog=0.1, log1p_duration=5.0)]
    with pytest.raises(M.ProtocolError):
        M.select_features(rows, M.FEATURES_M2)


# ── M1 已批准定义（2026-10-27）的契约测试 ───────────────────────────
def test_m1_feature_set_is_exact():
    """M1 = 仅 log1p_lam（研究者 2026-10-27 批准）；不得悄悄加列。"""
    assert M.FEATURES_M1 == ("log1p_lam",)
    assert len(M.FEATURES_M1) == 1
    assert M.model_features("M1") == ("log1p_lam",)


def test_m1_matrix_has_exactly_one_column():
    rows = [dict(log1p_lam=0.5, slope=0.1, since=12.0, censored=1.0, prog=0.2,
                 log1p_duration=5.0, ent=0.3) for _ in range(3)]
    X = M.select_features(rows, M.model_features("M1"))
    assert X.shape == (3, 1)


def test_m1_cannot_smuggle_extra_columns():
    """M1 的列数由 model_features 固定；给 fit_model 多传一列必须因形状不符失败。"""
    feats = M.model_features("M1")
    with pytest.raises(M.ProtocolError):
        M.fit_model(np.ones((4, 2)), np.array([0, 1, 0, 1]), feats)
