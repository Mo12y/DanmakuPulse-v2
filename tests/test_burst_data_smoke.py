"""exp35 数据级 smoke（**默认跳过**，需显式 opt-in）。

只读：仅核对 `time_offset == duration` 的边界计数，不修改任何数据。

运行（显式开启）：
  EXP35_DATA_SMOKE=1 ~/danmaku-venv/bin/python -m pytest tests/test_burst_data_smoke.py -q
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jobs"))

import m2_l6_burst_risk as M  # noqa: E402

DATA_ROOT = os.environ.get("EXP35_DATA_ROOT", "/home/molly/projects/danmaku")
MANIFEST = os.path.join(HERE, "..", "docs", "exp35_508", "file_manifest.csv")
ENABLED = os.environ.get("EXP35_DATA_SMOKE") == "1"


pytestmark = pytest.mark.skipif(
    not ENABLED, reason="opt-in：设置 EXP35_DATA_SMOKE=1 才运行（需真实 ODS 数据）")


def _load_offsets():
    import pandas as pd
    frames = []
    for dt in M.KNOWN_BATCHES:
        d = os.path.join(DATA_ROOT, "data", "warehouse", "ods", f"dt={dt}")
        for f in sorted(os.listdir(d)):
            if f.endswith(".parquet"):
                t = pd.read_parquet(os.path.join(d, f),
                                    columns=["bvid", "duration", "time_offset"])
                t["batch"] = dt
                frames.append(t)
    return pd.concat(frames, ignore_index=True)


def test_duration_boundary_counts():
    df = _load_offsets()
    eq = np.isclose(df["time_offset"], df["duration"])
    n_eq = int(eq.sum())
    n_need_clip = int((eq & (df["duration"] % M.B == 0)).sum())
    n_natural = int((eq & (df["duration"] % M.B != 0)).sum())

    assert int((df["time_offset"] > df["duration"]).sum()) == 0
    assert int((df["time_offset"] < 0).sum()) == 0
    assert int(df["time_offset"].isna().sum()) == 0

    assert n_eq == 36, f"offset==duration 应为 36 行，实得 {n_eq}"
    assert n_need_clip == 4, f"需夹取应为 4 行，实得 {n_need_clip}"
    assert n_natural == 32, f"自然落末桶应为 32 行，实得 {n_natural}"


def test_clip_counter_matches_boundary_rows():
    """build_sequence 的 clipped_to_last_bucket 合计必须等于"需夹取"的行数。"""
    seqs, _ = M.load_sequences(MANIFEST, DATA_ROOT, batches=M.KNOWN_BATCHES, layer="ods")
    total_clip = int(sum(s.audit.get("clipped_to_last_bucket", 0) for s in seqs))
    assert total_clip == 4, f"夹取计数应为 4，实得 {total_clip}"
    assert len(seqs) == 509
