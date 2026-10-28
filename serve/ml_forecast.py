# -*- coding: utf-8 -*-
"""serve/ml_forecast.py · ML 预测卡只读端点（任务38）

数据流：jobs/ml_forecast_build.py 预计算快照 JSON → 本模块只读分发（容器无 sklearn，不做任何计算）。
快照候选路径：/data-out/ml_forecast_snapshot.json（fastapi 容器挂载）→ <repo>/data-out/ml_forecast/ml_snapshot.json（host/pytest）。
端点：GET /api/ml/forecast[?bvid=] ；GET /api/ml/forecast/{bvid}。均被 app.py 全局登录中间件保护。
语义边界（写死，防包装）：
  d1 = 「发布后 6h 时点、对 (6h,30h] 弹幕量的预测」，不是播放量/热度/质量预测；
  d2.p_stable = 「当前节奏结论到采集结束不漂移(≤0.25 IQR)的模型概率」，不是高光/热度预测；
  d2.stable_measured = 全量对照的实测稳定性，与模型概率并列展示、严格区分。
"""
import json
import os

from fastapi import APIRouter, HTTPException, Response

router = APIRouter()

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [
    "/data-out/ml_forecast_snapshot.json",                      # fastapi 容器（/data-out 挂载）
    os.path.join(_REPO, "data-out", "ml_forecast", "ml_snapshot.json"),  # host / pytest
]
_SNAP = {"path": None, "mtime": None, "data": None}


def _load(refresh=False):
    """读快照（带 mtime 缓存；文件更新自动重载）。返回 dict 或 raise FileNotFoundError。"""
    for p in _CANDIDATES:
        if os.path.exists(p):
            mt = os.path.getmtime(p)
            if not refresh and _SNAP["data"] is not None and _SNAP["path"] == p and _SNAP["mtime"] == mt:
                return _SNAP["data"]
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            _SNAP.update(path=p, mtime=mt, data=data)
            return data
    raise FileNotFoundError("ml_snapshot.json 不存在于任何候选路径（先跑 jobs/ml_forecast_build.py）")


def shape_item(it):
    """单视频卡片载荷（纯函数，可测）：缺字段降级为 status=unavailable，不抛错。"""
    out = {"bvid": it.get("bvid"), "title": it.get("title"), "tname": it.get("tname"),
           "batch": it.get("batch"), "age_h": it.get("age_h"), "duration_min": it.get("duration_min"),
           "d1": {"status": "unavailable", "forecast_6h_30h": None, "actual_6h_30h": None},
           "d2": {"status": "unavailable", "current": None},
           "rhythm": it.get("rhythm")}
    d1 = it.get("d1") or {}
    out["d1"].update({k: d1.get(k) for k in ("status", "forecast_6h_30h", "actual_6h_30h")})
    d2 = it.get("d2") or {}
    cur = d2.get("current")
    if cur:
        out["d2"] = {"status": "ok", "current": {"t": cur.get("t"), "k": cur.get("k"),
                                                 "p_stable": cur.get("p_stable"),
                                                 "stable_measured": cur.get("stable_measured")}}
    return out


def register(app):
    app.include_router(router)


@router.get("/api/ml/forecast")
def ml_forecast(bvid: str = "", resp: Response = None):
    try:
        snap = _load()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    items = snap["items"]
    if bvid:
        items = [x for x in items if x["bvid"] == bvid]
        if not items:
            raise HTTPException(404, f"{bvid} 不在快照内")
    shaped = [shape_item(x) for x in items]
    return {"code": 0,
            "data": {"generated_at": snap["generated_at"], "data_window": snap["data_window"],
                     "n_items": len(shaped), "scope_note": snap.get("scope_note"), "items": shaped},
            "meta": {"n": len(shaped), "engine_version": "ml_forecast@exp38"}}


@router.get("/api/ml/forecast/{bvid}")
def ml_forecast_detail(bvid: str, resp: Response = None):
    try:
        snap = _load()
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    for x in snap["items"]:
        if x["bvid"] == bvid:
            out = shape_item(x)
            out["d2"]["points"] = (x.get("d2") or {}).get("points", [])
            out["model_versions"] = x.get("model_versions", {})
            return {"code": 0, "data": out, "meta": {"n": 1, "engine_version": "ml_forecast@exp38"}}
    raise HTTPException(404, f"{bvid} 不在快照内")
