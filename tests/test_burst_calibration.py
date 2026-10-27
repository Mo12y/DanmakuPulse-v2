"""exp35 校准单测：分组交叉拟合、折外分数、只用 D-train、D-conf 拒绝。

运行：~/danmaku-venv/bin/python -m pytest tests/test_burst_calibration.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jobs"))

import m2_l6_burst_risk as M  # noqa: E402


def make_dataset(n_videos=40, K=60, seed=11):
    rng = np.random.default_rng(seed)
    seqs = []
    for i in range(n_videos):
        c = rng.integers(0, 8, size=K)
        seqs.append(M.Sequence(f"BV{i:03d}", "20260902", K * M.B, c))
    return M.build_rows(seqs)


@pytest.fixture(scope="module")
def data():
    rows = make_dataset()
    y = np.array([r["y"] for r in rows], dtype=int)
    vid = np.array([r["bvid"] for r in rows], dtype=str)
    X = M.select_features(rows, M.FEATURES_M2)
    return rows, X, y, vid


def test_folds_are_grouped_by_video(data):
    rows, X, y, vid = data
    cf = M.cross_fit_scores(X, y, vid, M.FEATURES_M2, folds=5, seed=1)
    for a in cf["assignments"]:
        assert a["video_overlap"] == 0, "折外预测不得包含本折视频"
        assert a["n_train_videos"] + a["n_test_videos"] == len(set(vid))


def test_oof_is_complete_and_probabilistic(data):
    rows, X, y, vid = data
    oof = M.cross_fit_scores(X, y, vid, M.FEATURES_M2, folds=5, seed=1)["oof"]
    assert oof.shape == y.shape
    assert np.isfinite(oof).all()
    assert (oof > 0).all() and (oof < 1).all()


def test_oof_is_out_of_fold_not_in_sample(data):
    """同一视频在训练内的分数应比折外分数更"自信"（AUC 更高）。"""
    rows, X, y, vid = data
    cf = M.cross_fit_scores(X, y, vid, M.FEATURES_M2, folds=5, seed=1)
    oof_auc = M.macro_within_video_auc(y, cf["oof"], vid)["macro"]
    full = M.fit_model(X, y, M.FEATURES_M2, fit_videos=set(vid))
    in_auc = M.macro_within_video_auc(y, full.predict_proba(X), vid)["macro"]
    assert in_auc >= oof_auc - 0.02, "训练内分数不应显著低于折外分数"


def test_dev_rows_cannot_enter_fit(data):
    rows, X, y, vid = data
    bad = [dict(split="dev")]
    with pytest.raises(M.ProtocolError):
        M.assert_train_only(bad)
    M.assert_train_only([dict(split="train")] * 3)


def test_dconf_paths_rejected():
    for p in ("/data/warehouse/ods/dt=20261027/x.parquet",
              "/data/dconf/x.csv",
              "/home/molly/projects/danmaku-v2/data/cli_collected/hotlist_top200_20261027.csv"):
        with pytest.raises(M.DataIsolationError):
            M.guard_path(p)


def test_platt_fit_is_monotone_and_flagged_as_oof(data):
    rows, X, y, vid = data
    oof = M.cross_fit_scores(X, y, vid, M.FEATURES_M2, folds=5, seed=1)["oof"]
    calib = M.fit_platt(oof, y)
    assert calib["fit_split"] == "train_inner_oof"
    assert calib["a"] > 0
    grid = np.linspace(0.02, 0.98, 50)
    out = M.apply_platt(calib, grid)
    assert np.all(np.diff(out) > 0)


def test_calibration_recovers_slope_near_one():
    rng = np.random.default_rng(21)
    z = rng.normal(0, 1.2, size=20000)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(p.size) < p).astype(int)
    rep = M.calibration_report(p, y)
    assert 0.85 <= rep["slope"] <= 1.15
    assert abs(rep["intercept"]) < 0.1
    assert rep["brier_skill"] > 0


def test_reliability_curve_bins_sum_to_n():
    rng = np.random.default_rng(22)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    rep = M.calibration_report(p, y, bins=10)
    assert sum(b["n"] for b in rep["reliability"]) == 1000
    assert len(rep["reliability"]) == 10


def test_single_class_fit_rejected(data):
    rows, X, y, vid = data
    with pytest.raises(M.ProtocolError):
        M.fit_model(X, np.zeros(len(y), dtype=int), M.FEATURES_M2)
