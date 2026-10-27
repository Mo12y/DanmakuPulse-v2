"""exp35 评估单测：视频宏平均视频内 AUC、配对 Bootstrap、负对照。

运行：~/danmaku-venv/bin/python -m pytest tests/test_burst_eval.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jobs"))

import m2_l6_burst_risk as M  # noqa: E402


# ── 视频宏平均视频内 AUC ───────────────────────────────────────────
def test_macro_auc_hand_computed():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.1, 0.9])
    vid = np.array(["A", "A", "B", "B"])
    r = M.macro_within_video_auc(y, p, vid)
    assert r["per_video"]["A"] == pytest.approx(1.0)
    assert r["per_video"]["B"] == pytest.approx(0.0)
    assert r["macro"] == pytest.approx(0.5)     # (1.0 + 0.0) / 2，不是 pooled
    assert r["n_eligible"] == 2 and r["n_videos"] == 2


def test_macro_differs_from_pooled():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.1, 0.9])
    vid = np.array(["A", "A", "B", "B"])
    macro = M.macro_within_video_auc(y, p, vid)["macro"]
    pooled = M.pooled_auc(y, p)
    assert macro == pytest.approx(0.5)
    assert pooled == pytest.approx(0.5)          # 本例巧合相等
    y2 = np.array([1, 0, 1, 0])
    p2 = np.array([0.95, 0.9, 0.2, 0.1])
    v2 = np.array(["A", "A", "B", "B"])
    # A: 1>0 正确；B: 0.2 > 0.1 但标签 1/0 → 也正确 → macro 1.0
    assert M.macro_within_video_auc(y2, p2, v2)["macro"] == pytest.approx(1.0)
    assert M.pooled_auc(y2, p2) < 1.0


def test_single_class_video_excluded_not_filled():
    y = np.array([1, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    vid = np.array(["A", "A", "B", "B", "C", "C"])   # C 全 0
    r = M.macro_within_video_auc(y, p, vid)
    assert r["n_videos"] == 3
    assert r["n_eligible"] == 2
    assert r["n_excluded_single_class"] == 1
    assert "C" not in r["per_video"]
    assert r["macro"] == pytest.approx((1.0 + 1.0) / 2)
    assert r["eligibility_share"] == pytest.approx(2 / 3)


def test_auc_requires_two_classes():
    with pytest.raises(M.ProtocolError):
        M.auc([0, 0, 0], [0.1, 0.2, 0.3])


def test_duration_only_gives_exactly_half():
    """同一视频内 duration 恒定 ⇒ 预测值全并列 ⇒ 视频内 AUC 必然 0.5。"""
    rng = np.random.default_rng(3)
    ys, ps, vs = [], [], []
    for i in range(30):
        n = 8
        y = rng.integers(0, 2, size=n)
        if y.min() == y.max():
            y[0] = 1 - y[0]
        ys += list(y)
        ps += [5.0] * n          # 同一视频内常数
        vs += [f"BV{i:02d}"] * n
    r = M.macro_within_video_auc(np.array(ys), np.array(ps), np.array(vs))
    assert r["macro"] == pytest.approx(0.5, abs=1e-12)


# ── 配对 Bootstrap ─────────────────────────────────────────────────
def _boot_data(n_videos=60, n_points=8, seed=5):
    rng = np.random.default_rng(seed)
    y, vid = [], []
    for i in range(n_videos):
        yy = rng.integers(0, 2, size=n_points)
        if yy.min() == yy.max():
            yy[0] = 1 - yy[0]
        y += list(yy)
        vid += [f"BV{i:03d}"] * n_points
    y = np.array(y); vid = np.array(vid)
    pa = np.clip(y * 0.5 + rng.normal(0.25, 0.1, size=y.size), 0.001, 0.999)
    pb = np.clip(rng.random(y.size), 0.001, 0.999)
    return y, pa, pb, vid


def test_bootstrap_unit_is_video_and_uses_same_resample():
    y, pa, pb, vid = _boot_data()
    r = M.paired_bootstrap_delta(y, pa, pb, vid, B=500, seed=1)
    assert r["unit"] == "bvid"
    assert r["n_eligible"] == len(set(vid))
    assert r["n_valid_reps"] + r["n_discarded"] == 500


def test_bootstrap_identical_models_gives_zero():
    y, pa, _, vid = _boot_data()
    r = M.paired_bootstrap_delta(y, pa, pa, vid, B=500, seed=2)
    assert r["delta"] == pytest.approx(0.0)
    assert r["ci_lo"] == pytest.approx(0.0) and r["ci_hi"] == pytest.approx(0.0)


def test_bootstrap_reproducible():
    y, pa, pb, vid = _boot_data()
    a = M.paired_bootstrap_delta(y, pa, pb, vid, B=300, seed=42)
    b = M.paired_bootstrap_delta(y, pa, pb, vid, B=300, seed=42)
    assert (a["delta"], a["ci_lo"], a["ci_hi"]) == (b["delta"], b["ci_lo"], b["ci_hi"])


def test_bootstrap_delta_equals_mean_paired_video_delta():
    y, pa, pb, vid = _boot_data()
    r = M.paired_bootstrap_delta(y, pa, pb, vid, B=200, seed=7)
    ds = []
    for v in sorted(set(vid)):
        m = vid == v
        ds.append(M.auc(y[m], pa[m]) - M.auc(y[m], pb[m]))
    assert r["delta"] == pytest.approx(float(np.mean(ds)))


# ── 负对照 ─────────────────────────────────────────────────────────
def _seqs(n=10, K=60, seed=9):
    rng = np.random.default_rng(seed)
    return [M.Sequence(f"BV{i:02d}", "20260902", K * M.B, rng.integers(0, 9, size=K))
            for i in range(n)]


def test_shuffle_labels_reproducible_and_keeps_counts():
    rows = M.build_rows(_seqs())
    y = np.array([r["y"] for r in rows])
    vid = np.array([r["bvid"] for r in rows])
    a = M.shuffle_labels_within_video(y, vid, seed=7)
    b = M.shuffle_labels_within_video(y, vid, seed=7)
    assert np.array_equal(a, b)
    for v in np.unique(vid):
        m = vid == v
        assert a[m].sum() == y[m].sum()          # 每视频正例数不变
    assert a.sum() == y.sum()                    # 总体基率不变


def test_shift_features_keeps_labels_changes_features():
    seqs = _seqs()
    base = M.build_rows(seqs)
    shifted = M.build_rows(seqs, feature_shift=6)
    assert len(shifted) < len(base)               # 前移后早于可观察窗的点被丢弃
    key = lambda r: (r["bvid"], r["t"])
    yb = {key(r): r["y"] for r in base}
    assert all(yb[key(r)] == r["y"] for r in shifted)   # 标签不变
    fb = {key(r): r["prog"] for r in base}
    assert any(fb[key(r)] != r["prog"] for r in shifted)  # 特征变了


def test_label_permutation_eval_side_is_at_chance():
    """标准置换检验：真标签拟合的预测，对置乱标签评估 → 主指标应≈0.5。"""
    rows = M.build_rows(_seqs(n=40))
    y = np.array([r["y"] for r in rows]); vid = np.array([r["bvid"] for r in rows])
    X = M.select_features(rows, M.FEATURES_M2)
    y_perm = M.shuffle_labels_within_video(y, vid, seed=13)
    m = M.fit_model(X, y, M.FEATURES_M2)
    r = M.macro_within_video_auc(y_perm, m.predict_proba(X), vid)
    assert 0.4 <= r["macro"] <= 0.6


# ── 功效公式（设计估计，不是结果） ─────────────────────────────────
def test_mde_matches_normal_approximation():
    """MDE = (z.975 + z.80)·SD/√n；用精确分位复算（与旧产物用 1.96/0.8416 的差约 3e-7）。"""
    sd = 0.22204870655293082
    assert M.Z_975 == pytest.approx(1.9599639845400545, rel=1e-12)
    assert M.Z_80 == pytest.approx(0.8416212335729143, rel=1e-12)
    assert M.mde_80(sd, 150) == pytest.approx(0.0507933, abs=1e-6)
    assert M.mde_80(sd, 134) == pytest.approx(0.0537405, abs=1e-6)
    assert M.mde_80(sd, 200) == pytest.approx(0.0439885, abs=1e-6)
    # 与旧产物（用四舍五入常量 2.8016）的偏差必须 < 1e-5，否则说明公式变了
    assert abs(M.mde_80(sd, 150) - 0.05079357103752174) < 1e-5


def test_n_for_effect_02_is_968():
    """检出 0.02 所需视频数 = ceil(((z.975+z.80)*SD/0.02)^2) = 968。"""
    assert M.n_for_effect(0.22204870655293082, 0.02) == 968


def test_n_for_effect_rejects_nonpositive_delta():
    with pytest.raises(M.ProtocolError):
        M.n_for_effect(0.22, 0.0)
