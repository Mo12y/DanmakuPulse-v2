"""MF-1：协议版本一致性（代码 ↔ rc5 文档 ↔ 产物）。

运行：~/danmaku-venv/bin/python -m pytest tests/test_burst_protocol_version.py -q
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "jobs"))

import m2_l6_burst_risk as M  # noqa: E402

RC5 = os.path.join(ROOT, "docs", "exp35_508", "35_formal_spec_rc5_candidate.md")


def _rc5_text():
    with open(RC5, encoding="utf-8") as fh:
        return fh.read()


def test_version_matches_protocol_doc():
    """代码常量必须等于 rc5 头部声明的版本串（避免两边各自自洽）。"""
    m = re.search(r"协议版本串：`([^`]+)`", _rc5_text())
    assert m, "rc5 头部未找到 `协议版本串：` 声明"
    assert M.PROTOCOL_VERSION == m.group(1), f"code={M.PROTOCOL_VERSION} doc={m.group(1)}"


def test_version_matches_freeze_template():
    """rc5 §11 的冻结模板也必须用同一版本串。"""
    text = _rc5_text()
    assert re.search(r"protocol_version\s*:\s*" + re.escape(M.PROTOCOL_VERSION), text), \
        "rc5 冻结模板的 protocol_version 与代码常量不一致"


def test_no_stale_rc4_literal_in_source():
    src = open(os.path.join(ROOT, "jobs", "m2_l6_burst_risk.py"), encoding="utf-8").read()
    assert "exp35-formal-rc4" not in src
    assert src.count("exp35-formal-rc5-candidate") == 1


def test_all_artifact_paths_stamp_the_constant():
    """产物不得硬编码版本串：必须统一引用常量。"""
    src = open(os.path.join(ROOT, "jobs", "m2_l6_burst_risk.py"), encoding="utf-8").read()
    assert src.count("protocol_version=PROTOCOL_VERSION") >= 5, \
        "有产物构造点未打版本戳（power/confirm/audit/build/gate-sim/negctl）"
    assert '"protocol_version"' not in src.replace("protocol_version=PROTOCOL_VERSION", "")


def test_gate_simulation_result_carries_version():
    rows = [dict(bvid=f"BV{i:02d}", y=i % 2) for i in range(12) for _ in range(6)]
    out = M.gate_simulation(rows, n_grid=(6,), reps=5)
    assert out["protocol_version"] == M.PROTOCOL_VERSION


def test_confirm_result_carries_version():
    import numpy as np
    rng = np.random.default_rng(1)
    K = 60
    tr = [M.Sequence(f"BVt{i:02d}", "20260902", K * M.B, rng.integers(0, 7, size=K)) for i in range(20)]
    ev = [M.Sequence(f"BVe{i:02d}", "20260914", K * M.B, rng.integers(0, 7, size=K)) for i in range(14)]
    out = M.run_confirm(tr, ev, eval_split="dev", reps=200, seed=3)
    assert out["protocol_version"] == M.PROTOCOL_VERSION
