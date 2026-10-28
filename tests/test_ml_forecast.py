#!/usr/bin/env python3
"""test_ml_forecast.py — 任务38 ML 预测卡：单元（特征/载荷/口径）+ 契约（服务在线时）。

运行：~/danmaku-venv/bin/python -m pytest tests/test_ml_forecast.py -q
单元部分不依赖服务与真实快照；契约部分复用 test_api_contract 的登录模式，服务未起则 skip。
"""
import importlib.util
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


try:
    mf = _load("ml_forecast", "serve/ml_forecast.py")
except ImportError:
    mf = None  # 宿主 venv 无 fastapi：serve 模块仅容器内完整可用，载荷单测跳过

needs_mf = pytest.mark.skipif(mf is None, reason="宿主 venv 无 fastapi")
jb = _load("ml_forecast_build", "jobs/ml_forecast_build.py")

# ---------------- 特征/预测纯函数 ----------------


def test_hour_bins_boundaries():
    rel = np.array([0.0, 0.99, 1.0, 2.99, 3.0, 5.99, 6.0, -0.5])
    n01, n13, n36 = jb.hour_bins(rel)
    assert (n01, n13, n36) == (2.0, 2.0, 2.0)  # 6.0 与 -0.5 均不计入前 6h


def test_d1_predict_roundtrip():
    art = {"coef": [1.0, 0.2, -0.1, 0.05, 0.3, 0.5], "intercept": 2.0, "features": jb.D1_FEATS}
    p = jb.d1_predict(art, n01=10, n13=20, n36=30, duration=600)
    x = np.array([np.log1p(60), np.log1p(30), np.log1p(20), np.log1p(10),
                  np.log1p(30) - np.log1p(20), np.log1p(600)])
    assert p == pytest.approx(float(np.exp(np.dot(x, art["coef"]) + 2.0) - 1), rel=1e-9)
    assert p >= 0


def test_d1_feature_contract():
    """生产特征列序必须与 exp38 训练口径（OLS-6，6 特征）一致。"""
    assert jb.D1_FEATS == ["c0", "n36", "n13", "n01", "slope", "dur"]


# ---------------- 快照载荷 ----------------


@needs_mf
def test_shape_item_statuses():
    ok = mf.shape_item({"bvid": "BV1", "d1": {"status": "ok", "forecast_6h_30h": 12.3, "actual_6h_30h": 15},
                        "d2": {"current": {"t": 12, "k": 40, "p_stable": 0.8, "stable_measured": True}}})
    assert ok["d1"]["status"] == "ok" and ok["d2"]["status"] == "ok"
    assert ok["d2"]["current"]["p_stable"] == 0.8
    empty = mf.shape_item({"bvid": "BV2"})
    assert empty["d1"]["status"] == "unavailable" and empty["d2"]["status"] == "unavailable"


def test_snapshot_file_schema():
    """真实快照（若已生成）必须满足 schema 与状态枚举。"""
    paths = [os.path.join(REPO, "data-out/ml_forecast/ml_snapshot.json")]
    p = next((x for x in paths if os.path.exists(x)), None)
    if p is None:
        pytest.skip("快照未生成（先跑 jobs/ml_forecast_build.py）")
    s = json.load(open(p, encoding="utf-8"))
    assert s["data_window"] == "20261012-20261026" and s["n_items"] == len(s["items"])
    for it in s["items"][:20]:
        assert it["d1"]["status"] in ("ok", "pending", "insufficient")
        for pt in it["d2"]["points"]:
            assert 0.0 <= pt["p_stable"] <= 1.0 and pt["t"] in (1, 3, 6, 12, 24)
        # 节奏卡必备字段：弹幕规模 / 分钟桶 / 零桶数（页面「观测」区与覆盖度告警依赖）
        rh = it["rhythm"]
        assert isinstance(rh["n_danmaku"], int) and rh["n_danmaku"] > 0
        assert isinstance(rh["n_buckets"], int) and rh["n_buckets"] >= 1
        assert 0 <= rh["n_zero_buckets"] <= rh["n_buckets"]


def test_real_model_artifact():
    p = os.path.join(REPO, "experiments/exp38_integrate/models/d1_ols.json")
    if not os.path.exists(p):
        pytest.skip("模型工件未生成（先跑 experiments/exp38_integrate/audit_scan.py）")
    art = json.load(open(p, encoding="utf-8"))
    assert art["features"] == jb.D1_FEATS and len(art["coef"]) == 6
    assert "time-split" in art["eval_ref"]


def test_state_model_persisted():
    """exp47-v1 状态模型持久化：阈值一致性 + 概率归一 + 编码一致 + 短序列/缺失降级"""
    p = os.path.join(REPO, "experiments/exp47_state_structure/model.json")
    if not os.path.exists(p):
        pytest.skip("状态模型未导出（先跑 experiments/exp47_state_structure/export_model.py）")
    m = json.load(open(p, encoding="utf-8"))
    assert m["edges"] == [20.0, 45.0, 92.0]          # 与 exp47 results.json 一致
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in m["M1"])  # 每行概率和为 1
    assert m["K"] == 4 and len(m["state_names"]) == 4
    r = jb.state_fields(np.array([0, 19, 20, 45, 92, 93], float), m)
    assert r["seq"] == [0, 0, 1, 2, 3, 3]             # 状态编码与 searchsorted(right) 一致：c=45 归态 2
    assert r["current_idx"] == 3 and r["current_name"] == "S3 爆"
    assert abs(sum(r["next_prob"]) - 1.0) < 1e-4
    assert jb.state_fields(np.array([7.0]), m) is None        # 单桶 → 降级 None
    assert jb.state_fields(np.array([1.0, 2.0]), None) is None  # 无模型 → 降级 None


def test_state_fields_trans_counts_and_coverage():
    """状态字段扩展：转移计数矩阵、覆盖度、样本量提示所需字段（纯函数，可测）"""
    p = os.path.join(REPO, "experiments/exp47_state_structure/model.json")
    if not os.path.exists(p):
        pytest.skip("状态模型未导出")
    m = json.load(open(p, encoding="utf-8"))
    c60 = np.array([0, 19, 20, 45, 92, 93], float)
    r = jb.state_fields(c60, m)
    tc = np.array(r["trans_counts"])
    assert tc.shape == (4, 4)
    assert int(tc.sum()) == len(c60) - 1            # 每次转移只计一次
    assert tc[0, 0] == 1 and tc[0, 1] == 1 and tc[3, 3] == 1
    cov = r["coverage"]
    assert cov["n_buckets"] == 6 and cov["n_zero"] == 1
    assert cov["zero_share"] == pytest.approx(1 / 6, abs=1e-3)
    assert cov["observed_span"] == 6 and cov["tail_blank"] == 0
    # 尾部空白：末两桶无弹幕
    r2 = jb.state_fields(np.array([5, 5, 0, 0], float), m)
    assert r2["coverage"]["tail_blank"] == 2 and r2["coverage"]["observed_span"] == 2
    assert r2["current_idx"] == 0                   # 当前状态取自最后一桶（含零弹幕桶）


def test_state_fields_next_prob_from_training_matrix():
    """下一状态概率必须直接取自持久化 M1 行，不在本视频上重拟合"""
    p = os.path.join(REPO, "experiments/exp47_state_structure/model.json")
    if not os.path.exists(p):
        pytest.skip("状态模型未导出")
    m = json.load(open(p, encoding="utf-8"))
    r = jb.state_fields(np.array([1, 2, 3, 4, 5], float), m)
    cur = r["current_idx"]
    assert r["next_prob"] == [round(float(x), 4) for x in m["M1"][cur]]
    assert abs(sum(r["next_prob"]) - 1.0) < 1e-3
    assert r["K"] == 4 and r["trans"] == m["M1"]


def test_state_note_declares_exp48_1_boundary():
    """口径守卫：状态说明必须写明 exp48.1 的负结果，防止页面宣称『转移优于活跃度基线』"""
    note = jb.STATE_NOTE
    assert "exp48.1" in note and "无稳健增量" in note
    assert "不等同于人工语义阶段" in note


@needs_mf
def test_shape_item_rhythm_backward_compat():
    """旧快照（rhythm 无 state 字段）与完整快照均不崩"""
    old = mf.shape_item({"bvid": "BVold", "rhythm": {"seg3_dpm": [1, 2, 3], "S1": 0.5}})
    assert old["rhythm"]["seg3_dpm"] == [1, 2, 3] and "state" not in old["rhythm"]
    new = mf.shape_item({"bvid": "BVnew", "rhythm": {"state": {"current_idx": 1, "next_prob": [0.1, 0.2, 0.3, 0.4]}}})
    assert new["rhythm"]["state"]["current_idx"] == 1


# ---------------- 契约（服务在线时） ----------------

BASE = os.environ.get("API_BASE", "http://localhost:8000")
COOKIE = None


def _login():
    global COOKIE
    if COOKIE:
        return COOKIE
    body = urllib.parse.urlencode({"username": os.environ.get("API_USER", "admin"),
                                   "password": os.environ.get("API_PASS", "danmaku@2026")}).encode()
    req = urllib.request.Request(BASE + "/login", data=body, method="POST")

    class NR(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    op = urllib.request.build_opener(NR)
    try:
        r = op.open(req, timeout=10)
        raw = r.headers.get("Set-Cookie", "")
    except urllib.error.HTTPError as e:
        raw = e.headers.get("Set-Cookie", "")
    except Exception:
        COOKIE = ""
        return COOKIE
    COOKIE = raw.split(";")[0] if raw.startswith("ads_session=") else ""
    return COOKIE


def test_contract_forecast_list():
    ck = _login()
    if not ck:
        pytest.skip("服务未在线或登录失败")
    req = urllib.request.Request(BASE + "/api/ml/forecast", headers={"Cookie": ck})
    r = json.load(urllib.request.urlopen(req, timeout=10))
    assert r["code"] == 0 and r["data"]["n_items"] > 0
    it = r["data"]["items"][0]
    assert {"bvid", "d1", "d2"} <= set(it)


def test_contract_404():
    ck = _login()
    if not ck:
        pytest.skip("服务未在线或登录失败")
    req = urllib.request.Request(BASE + "/api/ml/forecast/BVXXXnotexist", headers={"Cookie": ck})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=10)
    assert ei.value.code == 404
