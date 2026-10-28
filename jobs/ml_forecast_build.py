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
MODEL_PATH = "/home/molly/projects/danmaku-v2/experiments/exp47_state_structure/model.json"


def load_state_model():
    """exp47-v1 持久化参数（阈值/转移矩阵）；缺失返回 None（快照降级为无状态字段）"""
    try:
        return json.load(open(MODEL_PATH, encoding="utf-8"))
    except Exception:
        return None


STATE_NOTE = ("状态=弹幕强度等级（exp47-v1 分位离散化），不等同于人工语义阶段。"
              "转移概率取自训练 276 支视频的转移矩阵（未用本视频重拟合）。"
              "exp48.1 已验证：该转移结构相对「本视频自身状态构成」基线无稳健增量"
              "（K=4 时 M1−B2b=+0.010 nats/桶，95%CI 含 0），"
              "故此处仅作描述性展示，不得作为『转移模型优于活跃度基线』的证据。")


def state_fields(c60, model):
    """状态序列/当前状态/下一状态概率/转移计数/覆盖度（exp47-v1 参数；纯函数，可测）。

    桶数<2 或无模型 → None（快照降级为无状态字段，前端显示"无状态数据"）。
    trans        = 训练集 M1（全局、本视频不参与拟合）
    trans_counts = 本视频实际发生的转移计数（描述性，样本量小时不可当规律）
    coverage     = 分钟桶覆盖：零弹幕桶占比、尾部空白桶数（采集窗可能未覆盖到片尾）
    """
    if not model:
        return None
    edges = np.array(model["edges"], float)
    c = np.asarray(c60, float)
    if len(c) < 2:
        return None
    K = int(model.get("K") or len(model["M1"]))
    st = np.searchsorted(edges, c, side="right").astype(int)
    cur = int(st[-1])
    tr = np.zeros((K, K), dtype=int)
    for t in range(1, len(st)):
        tr[st[t - 1], st[t]] += 1
    nz = np.flatnonzero(c > 0)
    last = int(nz[-1]) + 1 if nz.size else 0
    return dict(version=model["version"], K=K, current_idx=cur,
                current_name=model["state_names"][cur],
                next_prob=[round(float(x), 4) for x in model["M1"][cur]],
                next_names=list(model["state_names"]),
                seq=[int(v) for v in st], trans=model["M1"],
                trans_counts=tr.tolist(),
                edges=[float(e) for e in edges],
                coverage=dict(n_buckets=int(len(c)), n_zero=int((c == 0).sum()),
                              zero_share=round(float((c == 0).mean()), 4),
                              observed_span=int(last), tail_blank=int(len(c) - last)),
                note=STATE_NOTE)


def main():
    sys.path.insert(0, OLD_ANALYSIS)
    from danmaku_metrics import density_curve, extract_8_indicators

    d1_art = json.load(open(f"{MODELS}/d1_ols.json", encoding="utf-8"))
    d2_art = joblib.load(f"{MODELS}/d2_hgb.joblib")
    state_model = load_state_model()
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
        # ---- 节奏画像（exp42/45 口径：前中后密度 + S1 + 事件候选 + 曲线） ----
        nb60 = max(int(dur // 60), 1)
        c60 = np.bincount(np.minimum((np.clip(g["time_offset"].values.astype(float), 0, None) // 60).astype(int),
                                     nb60 - 1), minlength=nb60).astype(float)
        seg3 = [round(float(((g["time_offset"] >= dur * i / 3) & (g["time_offset"] < dur * (i + 1) / 3)).sum())
                      / max((dur / 3) / 60, 1e-9), 1) for i in range(3)]
        s1v = None
        if nb60 >= 5:
            sg = np.array_split(c60, 5); ms = np.array([x.mean() for x in sg])
            s1v = round(float(ms.std() / ms.mean()), 3) if ms.mean() > 0 else None
        base = np.array([np.median(c60[max(0, i - 6):i + 7]) for i in range(nb60)])
        thr = base + 2 * np.sqrt(base + 1.0)
        over = c60 >= thr
        events, i2 = [], 0
        while i2 < nb60:
            if over[i2]:
                j2 = i2
                while j2 + 1 < nb60 and over[j2 + 1]:
                    j2 += 1
                total = float(c60[i2:j2 + 1].sum())
                excess = float((c60[i2:j2 + 1] - thr[i2:j2 + 1]).clip(0).sum())
                if total >= 5 and excess >= 3:
                    events.append(dict(start_s=i2 * 60, end_s=(j2 + 1) * 60,
                                       peak_s=(i2 + int(np.argmax(c60[i2:j2 + 1]))) * 60, total=int(total)))
                i2 = j2 + 1
            else:
                i2 += 1
        step = max(1, nb60 // 120)
        curve = [dict(t=int(i * 60), c=int(c60[i])) for i in range(0, nb60, step)]
        rhythm = dict(seg3_dpm=seg3, S1=s1v, n_events=len(events), events=events, curve=curve,
                      n_danmaku=int(len(g)), n_buckets=int(nb60),
                      n_zero_buckets=int((c60 == 0).sum()),
                      note="事件=局部基线+2σ 的算法候选，未经人工确认",
                      state=state_fields(c60, state_model))
        items.append(dict(bvid=bv, batch=int(bt), title=str(g["title"].iloc[0])[:60],
                          tname=str(g["tname"].iloc[0]), duration_min=round(dur / 60, 1),
                          age_h=round(age, 1), d1=d1, d2=dict(points=pts, current=cur),
                          rhythm=rhythm,
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
