# -*- coding: utf-8 -*-
"""jobs/ml_forecast_build.py · ML 预测卡预计算任务（任务38 集成，宿主运行）

用法：~/danmaku-venv/bin/python jobs/ml_forecast_build.py
输入：data/cli_collected/hotlist_top200_202610*.csv（仅 1012–1026 批，与 exp38 审计口径一致；
      不读 1027+（exp35 D-conf 边界外保守处理））
模型：experiments/exp38_integrate/models/{d1_ols.json, d2_hgb.joblib}（exp38 生产工件，版本记录在内）
产出：
  1) data-out/ml_forecast/ml_snapshot.json          # 规范工件（host/pytest 读）
  2) ~/projects/danmaku/data/cli_collected/ml_forecast_snapshot.json  # fastapi 容器 /data-out 挂载副本
说明：容器无 sklearn ⇒ serve 只读快照，不做任何模型计算；快照内 p_stable/forecast 均为预计算值。
"""
import glob
import json
import os
import re
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
OLD_ANALYSIS = "/home/molly/projects/danmaku/analysis"
MODELS = f"{V2}/experiments/exp38_integrate/models"
OUT_CANON = f"{V2}/data-out/ml_forecast"
OUT_SERVE = "/home/molly/projects/danmaku/data/cli_collected/ml_forecast_snapshot.json"
DATA_WINDOW = (20261012, 20261026)
DIMS = ["cold_pct", "cv", "main_peak_pct"]
D1_FEATS = ["c0", "n36", "n13", "n01", "slope", "dur"]
COLS = ["bvid", "title", "tname", "duration", "pubdate", "time_offset", "send_time", "part_count", "status"]


# ---------- 纯函数（tests 直接导入） ----------
def hour_bins(rel_hours):
    """前 6h 的三个窗口计数（纯函数，可测）。rel: 相对 pubdate 的小时数组。"""
    rel = np.asarray(rel_hours, float)
    n01 = float(((rel >= 0) & (rel < 1)).sum())
    n13 = float(((rel >= 1) & (rel < 3)).sum())
    n36 = float(((rel >= 3) & (rel < 6)).sum())
    return n01, n13, n36


def d1_predict(art, n01, n13, n36, duration):
    c0 = n01 + n13 + n36
    x = np.array([np.log1p(c0), np.log1p(n36), np.log1p(n13), np.log1p(n01),
                  np.log1p(n36) - np.log1p(n13), np.log1p(duration)], float)
    return float(np.exp(np.dot(x, np.array(art["coef"])) + art["intercept"]) - 1)


# ---------- 主流程 ----------
def main():
    sys.path.insert(0, OLD_ANALYSIS)
    from danmaku_metrics import density_curve, extract_8_indicators

    d1_art = json.load(open(f"{MODELS}/d1_ols.json", encoding="utf-8"))
    d2_art = joblib.load(f"{MODELS}/d2_hgb.joblib")
    assert d1_art["features"] == D1_FEATS, "D1 特征口径与训练不一致"

    parts = []
    for f in sorted(glob.glob(f"{V2}/data/cli_collected/hotlist_top200_202610*.csv")):
        day = int(re.search(r"(2026\d{4})", os.path.basename(f)).group(1))
        if not (DATA_WINDOW[0] <= day <= DATA_WINDOW[1]):
            continue
        df = pd.read_csv(f, usecols=COLS, encoding="utf-8-sig", low_memory=False)
        df["batch"] = day
        parts.append(df)
    d = pd.concat(parts, ignore_index=True)
    for c in ["duration", "pubdate", "time_offset", "send_time", "part_count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["status"] == "ok") & d[["pubdate", "duration", "send_time"]].notna().all(axis=1) & (d["duration"] > 0)]
    d = d[~(d["part_count"].fillna(0) >= 2)]

    items, n_d1 = [], {"ok": 0, "pending": 0, "insufficient": 0}
    for (bv, bt), g in d.groupby(["bvid", "batch"]):
        pub = float(g["pubdate"].iloc[0]); dur = float(g["duration"].iloc[0])
        rel = (g["send_time"].values - pub) / 3600.0
        age = float(rel.max())
        off = g["time_offset"].values
        n01, n13, n36 = hour_bins(rel)
        c0 = n01 + n13 + n36
        # ---- D1 ----
        if age < 6.0 or c0 < 1:
            d1 = dict(status="insufficient", forecast_6h_30h=None, actual_6h_30h=None)
            n_d1["insufficient"] += 1
        else:
            fc = d1_predict(d1_art, n01, n13, n36, dur)
            actual = float(((rel > 6) & (rel <= 30)).sum()) if age >= 31.0 else None
            d1 = dict(status="ok" if actual is not None else "pending", forecast_6h_30h=round(fc, 1),
                      actual_6h_30h=actual, inputs=dict(n01=n01, n13=n13, n36=n36))
            n_d1[d1["status"]] += 1
        # ---- D2（取可用的最大 t 作为当前口径） ----
        pts = []
        if age >= 1.5:
            full = extract_8_indicators(density_curve(off, dur), dur)
            for t in [1, 3, 6, 12, 24]:
                if age < t + 0.5:
                    continue
                m = rel <= t; k = int(m.sum())
                if k < 10:
                    continue
                ind = extract_8_indicators(density_curve(off[m], dur), dur)
                x = np.array([[np.log1p(k), t,
                               len(set((off[m] / max(dur, 1) * 20).astype(int))) / 20,
                               float((rel >= t - 1).sum()) / k, np.log1p(dur)]])
                p = float(d2_art["model"].predict_proba(x)[0, 1])
                err = max(abs(ind[x2] - full[x2]) / d2_art["s_d_frozen"][x2] for x2 in DIMS)
                pts.append(dict(t=t, k=k, p_stable=round(p, 4), stable_measured=bool(err <= 0.25)))
        cur = pts[-1] if pts else None
        items.append(dict(bvid=bv, batch=int(bt), title=str(g["title"].iloc[0])[:60],
                          tname=str(g["tname"].iloc[0]), duration_min=round(dur / 60, 1),
                          age_h=round(age, 1), d1=d1, d2=dict(points=pts, current=cur),
                          model_versions=dict(d1=d1_art["trained_on"], d2=d2_art["trained_on"])))

    snap = dict(generated_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                data_window=f"{DATA_WINDOW[0]}-{DATA_WINDOW[1]}",
                scope_note="只含 ≤20261026 批次（exp35 D-conf 边界外）；多P已知视频剔除",
                n_items=len(items), d1_status_counts=n_d1,
                endpoint_hint="GET /api/ml/forecast?bvid=", items=items)
    os.makedirs(OUT_CANON, exist_ok=True)
    for p in (f"{OUT_CANON}/ml_snapshot.json", OUT_SERVE):
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f, ensure_ascii=False)
        os.replace(tmp, p)  # 原子写，避免截断
    print(f"SNAPSHOT-DONE items={len(items)} d1={n_d1} → {OUT_CANON}/ml_snapshot.json + {OUT_SERVE}")


if __name__ == "__main__":
    main()
