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
    """生产特征列序必须与 exp38 训练口径（OLS-5f）一致。"""
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


def test_real_model_artifact():
    p = os.path.join(REPO, "experiments/exp38_integrate/models/d1_ols.json")
    if not os.path.exists(p):
        pytest.skip("模型工件未生成（先跑 experiments/exp38_integrate/audit_scan.py）")
    art = json.load(open(p, encoding="utf-8"))
    assert art["features"] == jb.D1_FEATS and len(art["coef"]) == 6
    assert "time-split" in art["eval_ref"]


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
