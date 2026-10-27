"""exp35 切分单测：批次归属、交集为零、first_seen、可复现。

运行：~/danmaku-venv/bin/python -m pytest tests/test_burst_split.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jobs"))

import m2_l6_burst_risk as M  # noqa: E402


def mkseq(c, bvid, batch):
    c = np.asarray(c, dtype=np.int64)
    return M.Sequence(bvid=bvid, batch=batch, duration=len(c) * M.B, c=c)


def test_split_of_known_batches():
    for b in ("20260902", "20260907"):
        assert M.split_of(b) == "train"
    for b in ("20260914", "20260917"):
        assert M.split_of(b) == "dev"


def test_train_batches_are_first_two():
    assert M.D_TRAIN_BATCHES == ("20260902", "20260907")
    assert max(M.D_TRAIN_BATCHES) == "20260907"


def test_dconf_batches_rejected():
    for b in ("20261027", "20261101", "20261231"):
        with pytest.raises(M.DataIsolationError):
            M.split_of(b)


def test_unknown_batch_rejected_not_defaulted_to_train():
    for b in ("20260101", "20261026", "whatever"):
        with pytest.raises(M.DataIsolationError):
            M.split_of(b)


def test_intersection_zero_by_construction():
    seqs = [mkseq([1] * 24, "BV1a", "20260902"),
            mkseq([1] * 24, "BV1b", "20260917")]
    audit = M._split_audit(seqs)
    assert audit["train_videos"] == 1 and audit["dev_videos"] == 1
    assert audit["n_intersection"] == 0
    assert audit["intersection"] == []


def test_intersection_detected_when_video_crosses_splits():
    seqs = [mkseq([1] * 24, "BV1dup", "20260902"),
            mkseq([1] * 24, "BV1dup", "20260914")]
    audit = M._split_audit(seqs)
    assert audit["n_intersection"] == 1
    assert audit["intersection"] == ["BV1dup"]


def test_every_point_inherits_video_split():
    seqs = [mkseq([2] * 30, "BV1a", "20260902"), mkseq([2] * 30, "BV1b", "20260914")]
    rows = M.build_rows(seqs)
    assert {r["bvid"]: r["split"] for r in rows}["BV1a"] == "train"
    assert {r["bvid"]: r["split"] for r in rows}["BV1b"] == "dev"
    assert all(r["split"] == ("train" if r["bvid"] == "BV1a" else "dev") for r in rows)


def test_first_seen_must_use_raw_table():
    raw = [("BV1x", "20260902"), ("BV1x", "20260914"), ("BV1y", "20260914")]
    filtered = [("BV1x", "20260914"), ("BV1y", "20260914")]   # 0902 那天不足 200 条被滤掉
    assert M.first_seen(raw)["BV1x"] == "20260902"
    assert M.first_seen(filtered)["BV1x"] == "20260914"       # 用筛后表会算错
    res = M.split_leak_check(raw, assigned_split={"BV1x": "train", "BV1y": "dev"})
    assert res["n_violations"] == 0                           # 原始表口径：未违规
    res2 = M.split_leak_check(filtered, assigned_split={"BV1x": "train", "BV1y": "dev"})
    assert res2["n_violations"] == 1                          # 用筛后表复算会误判


def test_split_is_reproducible():
    seqs = [mkseq(np.arange(30) % 4, "BV1a", "20260902")]
    a = M.build_rows(seqs)
    b = M.build_rows(seqs)
    assert a == b


def test_train_rows_only_guard():
    rows = [dict(split="train"), dict(split="dev")]
    with pytest.raises(M.ProtocolError):
        M.assert_train_only(rows)
    M.assert_train_only([dict(split="train")])


def test_eligibility_gate_counts():
    rows = ([dict(bvid="A", y=1)] * 2 + [dict(bvid="A", y=0)] * 3 +      # 5 点 2 正 3 负 → 合格
            [dict(bvid="B", y=1)] * 2 + [dict(bvid="B", y=0)] * 2 +      # 4 点 → 不合格
            [dict(bvid="C", y=1)] * 1 + [dict(bvid="C", y=0)] * 6 +      # 只有 1 正 → 不合格
            [dict(bvid="D", y=1)] * 6 + [dict(bvid="D", y=0)] * 1)       # 只有 1 负 → 不合格
    assert M.eligible_video_ids(rows) == ["A"]


# ── A3（2026-10-27 批准）：多 P 口径 ────────────────────────────────
def _seq_pc(c, bvid, pc):
    c = np.asarray(c, dtype=np.int64)
    return M.Sequence(bvid=bvid, batch="20260902", duration=len(c) * M.B, c=c, part_count=pc)


def test_known_multip_only_counts_positive_evidence():
    rows = M.build_rows([_seq_pc([2] * 30, "BVmulti", 4),
                         _seq_pc([2] * 30, "BVsingle", 1),
                         _seq_pc([2] * 30, "BVunknown", None)])
    assert M.known_multip_bvids(rows) == {"BVmulti"}


def test_multip_policy_keep_keeps_everything():
    rows = M.build_rows([_seq_pc([2] * 30, "BVmulti", 2),
                         _seq_pc([2] * 30, "BVunknown", None)])
    assert len(M.filter_multip(rows, "keep")) == len(rows)


def test_multip_policy_exclude_known_drops_only_known():
    rows = M.build_rows([_seq_pc([2] * 30, "BVmulti", 2),
                         _seq_pc([2] * 30, "BVunknown", None),
                         _seq_pc([2] * 30, "BVsingle", 1)])
    kept = M.filter_multip(rows, "exclude_known")
    assert {r["bvid"] for r in kept} == {"BVunknown", "BVsingle"}
    assert len({r["bvid"] for r in rows}) - len({r["bvid"] for r in kept}) == 1


def test_multip_policy_rejects_unknown_value():
    with pytest.raises(M.ProtocolError):
        M.filter_multip([], "drop_all_unknown")


def test_multip_rows_carry_part_count():
    rows = M.build_rows([_seq_pc([2] * 30, "BVmulti", 5)])
    assert all(r["part_count"] == 5 for r in rows)
