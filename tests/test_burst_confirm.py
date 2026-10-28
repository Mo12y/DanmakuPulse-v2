"""exp35 confirm 端到端（合成数据）单测：训练/评估隔离、指标、Bootstrap、校准、输出结构。

运行：~/danmaku-venv/bin/python -m pytest tests/test_burst_confirm.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jobs"))

import m2_l6_burst_risk as M  # noqa: E402


def mkseqs(n, batch, prefix, seed, K=60):
    rng = np.random.default_rng(seed)
    return [M.Sequence(f"BV{prefix}{i:03d}", batch, K * M.B,
                       rng.integers(0, 7, size=K)) for i in range(n)]


@pytest.fixture(scope="module")
def split_data():
    train = mkseqs(40, "20260902", "tr", 101)
    ev = mkseqs(30, "20260914", "ev", 202)
    return train, ev


def test_confirm_end_to_end(split_data):
    train, ev = split_data
    out = M.run_confirm(train, ev, eval_split="dev", reps=200, seed=7)

    assert out["stage"] == "confirm" and out["eval_split"] == "dev"
    assert out["protocol_version"] == M.PROTOCOL_VERSION
    c = out["counts"]
    assert c["train_points"] > 0 and c["eval_points"] > 0
    assert c["eval_videos_eligible"] <= c["eval_videos_with_points"]

    # 两个模型必须用同一评估单位
    assert out["models"]["M1"]["n_eligible"] == out["models"]["M2"]["n_eligible"]
    assert out["models"]["M1"]["features"] == ["log1p_lam"]
    assert len(out["models"]["M2"]["features"]) == 6
    for name in ("M1", "M2"):
        auc = out["models"][name]["macro_auc"]
        assert 0.0 <= auc <= 1.0
        assert out["models"][name]["n_iter"] > 0

    d = out["delta"]
    assert d["ci_lo"] <= d["delta"] <= d["ci_hi"]
    assert d["unit"] == "bvid" and d["B"] == 200
    assert d["n_valid_reps"] + d["n_discarded"] == 200

    cal = out["calibration"]
    assert cal["fit_split"] == "train_inner_oof"
    assert cal["eval_on"] == "dev"
    rep = cal["report"]
    assert np.isfinite(rep["slope"]) and np.isfinite(rep["brier"])
    assert len(rep["reliability"]) > 0

    # 每视频计数覆盖全部合格评估视频
    assert len(out["per_video"]) == out["models"]["M2"]["n_eligible"]
    for v, d2 in out["per_video"].items():
        assert d2["pos"] + d2["neg"] == d2["points"]
        assert d2["pos"] >= 2 and d2["neg"] >= 2


def test_confirm_rejects_dconf(split_data):
    train, ev = split_data
    with pytest.raises(M.DataIsolationError):
        M.run_confirm(train, ev, eval_split="dconf")


def test_confirm_rejects_unknown_split(split_data):
    train, ev = split_data
    with pytest.raises(M.ProtocolError):
        M.run_confirm(train, ev, eval_split="train")


def test_confirm_rejects_video_overlap(split_data):
    train, _ = split_data
    leaked = [M.Sequence(train[0].bvid, "20260914", 60 * M.B,
                         np.random.default_rng(3).integers(0, 7, size=60))]
    with pytest.raises(M.ProtocolError):
        M.run_confirm(train, leaked, eval_split="dev")


def test_confirm_rejects_wrong_batches(split_data):
    train, ev = split_data
    # 训练集混入 dev 批次
    with pytest.raises(M.ProtocolError):
        M.run_confirm(train + ev, ev, eval_split="dev")
    # 评估集混入 train 批次
    with pytest.raises(M.ProtocolError):
        M.run_confirm(train, train, eval_split="dev")


def test_confirm_is_deterministic(split_data):
    train, ev = split_data
    a = M.run_confirm(train, ev, eval_split="dev", reps=300, seed=11)
    b = M.run_confirm(train, ev, eval_split="dev", reps=300, seed=11)
    assert a["delta"]["delta"] == b["delta"]["delta"]
    assert a["models"]["M2"]["macro_auc"] == b["models"]["M2"]["macro_auc"]
