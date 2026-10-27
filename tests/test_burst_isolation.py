"""exp35 数据隔离单测：ODS/DWD 分离、v2 不进正式路径、D-conf 拒绝、输入可追溯。

运行：~/danmaku-venv/bin/python -m pytest tests/test_burst_isolation.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "jobs"))

import m2_l6_burst_risk as M  # noqa: E402


def test_output_dirs_are_independent():
    ods = M.output_dir_for("ods")
    dwd = M.output_dir_for("dwd")
    assert ods != dwd
    assert "dwd" not in ods and "sensitivity" in dwd


def test_layer_mixing_rejected():
    with pytest.raises(M.ProtocolError):
        M.assert_single_layer(["ods", "dwd"])
    assert M.assert_single_layer(["ods", "ODS"]) == "ods"
    with pytest.raises(M.ProtocolError):
        M.assert_single_layer(["v2"])


def test_v2_exploration_data_rejected():
    with pytest.raises(M.DataIsolationError):
        M.guard_path("/home/molly/projects/danmaku-v2/data/cli_collected/hotlist_top200_20261012.csv")
    # 代码仓库本身可以访问
    assert M.guard_path("/home/molly/projects/danmaku-v2/jobs/m2_l6_burst_risk.py")


def test_dconf_batch_rejected_before_reading(tmp_path):
    man = tmp_path / "file_manifest.csv"
    man.write_text("layer,batch,file,bytes,sha256,rows\n", encoding="utf-8")
    with pytest.raises(M.DataIsolationError):
        M.load_sequences(man, str(tmp_path), batches=("20261027",))


def test_dconf_path_in_manifest_rejected(tmp_path):
    (tmp_path / "data").mkdir()
    man = tmp_path / "file_manifest.csv"
    man.write_text(
        "layer,batch,file,bytes,sha256,rows\n"
        "ods_parquet,20260902,data/warehouse/ods/dt=20261027/part-0.parquet,1,deadbeef,\n",
        encoding="utf-8")
    with pytest.raises(M.DataIsolationError):
        M.load_sequences(man, str(tmp_path), batches=("20260902",))


def _write_batch(root, batch, bvid, K=60):
    d = root / "data" / "warehouse" / "ods" / f"dt={batch}"
    d.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(dict(
        bvid=[bvid] * K,
        duration=[float(K * M.B)] * K,
        time_offset=[float(i * M.B) for i in range(K)],
        row_id=[f"{batch}-{i}" for i in range(K)],
        part_count=[None] * K,
    ))
    f = d / "part-00000.parquet"
    df.to_parquet(f, index=False)
    return f


def test_manifest_whitelist_and_provenance(tmp_path):
    f = _write_batch(tmp_path, "20260902", "BV1a")
    sha = M._sha256(f)
    man = tmp_path / "file_manifest.csv"
    man.write_text(
        "layer,batch,file,bytes,sha256,rows\n"
        f"ods_parquet,20260902,data/warehouse/ods/dt=20260902/part-00000.parquet,"
        f"{f.stat().st_size},{sha},\n", encoding="utf-8")
    seqs, prov = M.load_sequences(man, str(tmp_path), batches=("20260902",))
    assert len(seqs) == 1 and seqs[0].bvid == "BV1a" and seqs[0].K == 60
    assert prov[0]["sha256"] == sha
    assert [p["file"] for p in prov] == ["data/warehouse/ods/dt=20260902/part-00000.parquet"]


def test_loader_reads_only_listed_files(tmp_path):
    _write_batch(tmp_path, "20260902", "BV1a")
    # 另有一个未登记文件（模拟"通配符会读到"的东西）
    extra = tmp_path / "data" / "warehouse" / "ods" / "dt=20260907"
    extra.mkdir(parents=True)
    _write_batch(tmp_path, "20260907", "BV1b")
    man = tmp_path / "file_manifest.csv"
    f1 = tmp_path / "data/warehouse/ods/dt=20260902/part-00000.parquet"
    man.write_text(
        "layer,batch,file,bytes,sha256,rows\n"
        f"ods_parquet,20260902,data/warehouse/ods/dt=20260902/part-00000.parquet,"
        f"{f1.stat().st_size},{M._sha256(f1)},\n", encoding="utf-8")
    seqs, prov = M.load_sequences(man, str(tmp_path), batches=("20260902",))
    assert [s.bvid for s in seqs] == ["BV1a"]
    assert len(prov) == 1


def test_manifest_row_missing_file_is_reported(tmp_path):
    man = tmp_path / "file_manifest.csv"
    man.write_text(
        "layer,batch,file,bytes,sha256,rows\n"
        "ods_parquet,20260902,data/nope.parquet,10,abc,\n", encoding="utf-8")
    res = M.verify_manifest(man, str(tmp_path))
    assert res["all_ok"] is False and res["rows"][0]["status"] == "missing"


def test_manifest_sha_mismatch_detected(tmp_path):
    f = _write_batch(tmp_path, "20260902", "BV1a")
    man = tmp_path / "file_manifest.csv"
    man.write_text(
        "layer,batch,file,bytes,sha256,rows\n"
        f"ods_parquet,20260902,data/warehouse/ods/dt=20260902/part-00000.parquet,"
        f"{f.stat().st_size},0000,\n", encoding="utf-8")
    res = M.verify_manifest(man, str(tmp_path))
    assert res["rows"][0]["status"] == "sha_mismatch"
