#!/usr/bin/env python3
"""
m2_l6_burst_risk.py — exp35 正式实现：播放位置短期爆发风险（H = 6 × 5 s = 30 s）

定位（写死，不因结果修改）
------------------------
给定**单个观测快照**，只用 `time_offset < t` 的弹幕，预测 `[t, t+30s)` 内是否出现
"相对决策点基线的统计爆发"。是**离线、快照条件**的播放位置预测，不是实时预警
（快照在采集时刻已含 t 之后的全部弹幕，实时性不可检验 —— 见协议 §1.1 / §12）。

协议常量（改任一值 = 新协议版本，见 docs/exp35_508/35_formal_spec_rc4_candidate.md）
--------------------------------------------------------------------------
B=5 s 桶宽　PRE=12 桶（60 s 历史）　H=6 桶（30 s 预测）　步长 = H
thr_t = max(3.0, λ_t + 2·√λ_t)　　y_t = 1[ max(c[t:t+H]) >= thr_t ]
λ_t = mean(c[t-PRE:t])　　（阈值在 t 处只算一次，不在未来窗口内滚动）

运行
----
  PY=~/danmaku-venv/bin/python
  $PY jobs/m2_l6_burst_risk.py --verify-manifest
  $PY jobs/m2_l6_burst_risk.py --stage power --data-root /home/molly/projects/danmaku
  $PY jobs/m2_l6_burst_risk.py --stage gate-sim --data-root /home/molly/projects/danmaku
  $PY jobs/m2_l6_burst_risk.py --stage build --data-root /home/molly/projects/danmaku

边界（硬约束，违反即 raise，不静默）
----------------------------------
* 只允许 ODS 层；DWD 只作独立敏感性（不同输出目录，禁止合并入模）
* 只允许批次 {20260902,20260907}(train) / {20260914,20260917}(dev)；
  其余批次（含任何 >= 20261027 的 D-conf 候选）一律 DataIsolationError
* 特征白名单 + 显式黑名单断言；非法列立即失败
* 输入文件来自 manifest 白名单，不 glob 目录
* 多 P（A3，2026-10-27 批准）：主分析 `keep`（保留全部，空值不当单 P）；
  敏感性 `exclude_known`（只排除 `part_count > 1` 的已知多 P）。两者不得事后互换。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 0. 协议常量（写死）
# ══════════════════════════════════════════════════════════════════════
PROTOCOL_VERSION = "exp35-formal-rc5-candidate"

B = 5                 # 桶宽（秒）
PRE = 12              # 历史窗口（桶）
H = 6                 # 预测窗口（桶）= 30 s
DECISION_STEP = H     # 决策点步长 = H（窗口不重叠）
MIN_COUNT = 3.0       # 阈值下限
SIGMA = 2.0           # 阈值系数
MIN_DURATION = 120    # 正式可用视频最低时长（秒）

MIN_POINTS = 5        # 每视频有效决策点下限
MIN_POS = 2           # 每视频正例下限
MIN_NEG = 2           # 每视频负例下限
MIN_ELIGIBLE_VIDEOS = 30   # 报告下限（非功效保证）

BOOT_B = 2000
BOOT_SEED = 20261026
BOOT_MIN_ELIGIBLE = 50
CALIB_FOLDS = 5
CALIB_SEED = 20261026
CV_FOLDS = 5
CV_SEED = 20261026

D_TRAIN_BATCHES = ("20260902", "20260907")
D_DEV_BATCHES = ("20260914", "20260917")
KNOWN_BATCHES = D_TRAIN_BATCHES + D_DEV_BATCHES
D_CONF_FIRST_BATCH = "20261027"   # 冻结 commit 之后才开始采集的批次

# 特征名映射（任务书用 lambda / duration，协议 §5 批准的是 log1p 变换）
TASK_FEATURE_ALIAS = {"lambda": "log1p_lam", "duration": "log1p_duration"}

FEATURES_M1 = ("log1p_lam",)
FEATURES_M2 = ("log1p_lam", "slope", "since", "censored", "prog", "log1p_duration")
FEATURES_M3 = FEATURES_M2 + ("ent",)
FEATURES_DURATION_ONLY = ("log1p_duration",)

BANNED = frozenset({
    "n_dm", "danmaku_got", "danmaku_reported", "play", "like", "coin", "favorite",
    "share", "reply", "tname", "up_mid", "up_name", "bvid", "cid", "pubdate",
    "y", "label", "thr", "thr_t", "lam_t", "future_max", "win_max",
    "main_peak_pct", "cold_pct", "cv", "global_ent", "is_spam",
})

# A3（2026-10-27 批准）：主分析保留全部；敏感性排除"可可靠识别的已知多 P"
MULTIP_POLICIES = ("keep", "exclude_known")
MULTIP_POLICY_PRIMARY = "keep"

MODEL_SPECS = {
    "M0": None,   # 未获批准定义 —— 保留接口，不实现
    "M1": FEATURES_M1,
    "M2": FEATURES_M2,
    "M3": FEATURES_M3,
    "M4": None,   # 未获批准定义 —— 保留接口，不实现
}
PRIMARY_MODEL = "M2"
SECONDARY_MODEL = "M3"
COMPARISON_MODEL = "M1"


class ProtocolError(RuntimeError):
    """协议/实现契约被违反。"""


class DataIsolationError(RuntimeError):
    """试图触碰 D-conf / 未登记批次 / 跨层混用。"""


# ══════════════════════════════════════════════════════════════════════
# 1. 数据隔离
# ══════════════════════════════════════════════════════════════════════
_DCONF_PATTERNS = (
    re.compile(r"d[_\-\s]?conf", re.I),
    re.compile(r"dt\s*=\s*2026(102[7-9]|1[12]\d\d)"),
    re.compile(r"(?<!\d)2026(102[7-9]|1[12]\d\d)(?!\d)"),
    re.compile(r"hotlist_(top\d+|videos)_.*2026(102[7-9]|1[12]\d\d)", re.I),
)


_V2_DATA = re.compile(r"danmaku-v2[/\\]+data", re.I)


def guard_path(path) -> str:
    """拒绝任何疑似 D-conf 的路径，以及 v2 探索数据（不进正式路径）。

    白名单之外的一切数据入口都必须先过这里。
    """
    s = str(path)
    for pat in _DCONF_PATTERNS:
        if pat.search(s):
            raise DataIsolationError(
                f"拒绝访问疑似 D-conf 路径: {s}（匹配 {pat.pattern}）")
    if _V2_DATA.search(s):
        raise DataIsolationError(f"v2 探索数据不得进入正式路径: {s}")
    return s


def split_of(batch) -> str:
    """批次 → 集合名。未登记批次一律拒绝，绝不"默认归入 train"。"""
    b = str(batch).strip()
    if b in D_TRAIN_BATCHES:
        return "train"
    if b in D_DEV_BATCHES:
        return "dev"
    if b.isdigit() and b >= D_CONF_FIRST_BATCH:
        raise DataIsolationError(f"批次 {b} 属 D-conf 区间（>= {D_CONF_FIRST_BATCH}），正式路径拒绝读取")
    raise DataIsolationError(f"未登记批次 {b}；允许集合 = {KNOWN_BATCHES}")


def assert_single_layer(layers) -> str:
    """ODS 与 DWD 不得混入同一训练/校准/评估集合。"""
    s = {str(x).lower() for x in layers}
    if len(s) != 1:
        raise ProtocolError(f"禁止混用数据层: {sorted(s)}")
    layer = s.pop()
    if layer not in ("ods", "dwd"):
        raise ProtocolError(f"未知数据层: {layer}")
    return layer


def output_dir_for(layer: str) -> str:
    """ODS 与 DWD 输出目录必须独立。"""
    layer = assert_single_layer([layer])
    return os.path.join("data-out", "m2_l6", "ods" if layer == "ods" else "dwd_sensitivity")


# ══════════════════════════════════════════════════════════════════════
# 2. 序列构建（去重 → 分桶）
# ══════════════════════════════════════════════════════════════════════
@dataclass
class Sequence:
    """单视频在单快照内的 5 s 桶计数序列。"""
    bvid: str
    batch: str
    duration: float
    c: np.ndarray
    audit: dict = field(default_factory=dict)
    part_count: object = None      # ODS 的 part_count（>1 = 已知多 P；None = 未知）

    @property
    def K(self) -> int:
        return int(self.c.size)

    @property
    def usable(self) -> bool:
        return self.duration >= MIN_DURATION and self.K >= PRE + H


def build_sequence(bvid, batch, time_offsets, dmids, duration, *,
                   B=B, min_duration=MIN_DURATION, part_count=None) -> Sequence:
    """正式规则：**先按 dmid 去重，再按 floor(time_offset/B) 分桶**。

    - dmid = ODS 的 `row_id`（19 位；schema 无 `dmid` 字段，映射见协议 §5.1）
    - 时间缺失/非有限、负值、> duration：丢弃并计数（不静默）
    - offset == duration：若 duration 为 B 的整数倍，floor 会得到 K（越界一个桶），
      按"归入最后一桶 K-1"处理并计入 `clipped_to_last_bucket`；否则自然落在 K-1。
      本轮 4 批共 36 行 `offset == duration`，其中 4 行需要夹取（见审计报告 §3）。
    """
    split_of(batch)  # 未登记批次在此 raise
    off = np.asarray(time_offsets, dtype=float).ravel()
    ids = np.asarray(dmids, dtype=str).ravel()
    if off.size != ids.size:
        raise ProtocolError(f"time_offset({off.size}) 与 dmid({ids.size}) 长度不一致")
    dur = float(duration)
    if not math.isfinite(dur) or dur <= 0:
        raise ProtocolError(f"{bvid}/{batch}: 非法 duration={duration!r}")

    audit = dict(rows_in=int(off.size), dropped_nan=0, dropped_negative=0,
                 dropped_beyond_duration=0, dropped_duplicate_dmid=0, clipped_to_last_bucket=0)
    ok = np.isfinite(off)
    audit["dropped_nan"] = int((~ok).sum())
    neg = ok & (off < 0)
    audit["dropped_negative"] = int(neg.sum())
    ok &= ~neg
    oob = ok & (off > dur)
    audit["dropped_beyond_duration"] = int(oob.sum())
    ok &= ~oob
    off, ids = off[ok], ids[ok]

    # 去重：按 (offset, dmid) 排序后保留每个 dmid 的首条 → 确定性、可复现
    order = np.lexsort((ids, off))
    off, ids = off[order], ids[order]
    _, first = np.unique(ids, return_index=True)
    keep = np.zeros(ids.size, dtype=bool)
    keep[first] = True
    audit["dropped_duplicate_dmid"] = int(ids.size - keep.sum())
    off = off[keep]
    audit["rows_kept"] = int(off.size)

    K = int(math.ceil(dur / B))
    idx = np.floor(off / B).astype(np.int64)
    hi = int((idx >= K).sum())
    if hi:
        audit["clipped_to_last_bucket"] = hi
        idx = np.clip(idx, 0, K - 1)
    c = np.bincount(idx, minlength=K).astype(np.int64)
    return Sequence(bvid=str(bvid), batch=str(batch), duration=dur, c=c, audit=audit,
                    part_count=part_count)


def decision_times(K, *, pre=PRE, h=H, step=None) -> np.ndarray:
    """合法决策点：t ∈ {PRE, PRE+step, ...} 且 **t + H <= K**（闭区间上界，最后一点必须保留）。"""
    step = h if step is None else step
    if K < pre + h:
        return np.zeros(0, dtype=np.int64)
    ts = list(range(pre, K - h + 1, step))
    return np.asarray(ts, dtype=np.int64)


def threshold(lam, *, min_count=MIN_COUNT, sigma=SIGMA) -> float:
    """thr_t = max(3.0, λ + 2√λ)；λ=0 时退化为 3.0（float64，不引入 epsilon）。"""
    lam = float(lam)
    return float(max(min_count, lam + sigma * math.sqrt(lam)))


@dataclass(frozen=True)
class Point:
    bvid: str
    batch: str
    t: int
    lam: float
    thr: float
    y: int
    win_start: int
    win_end: int      # 不含
    K: int

    def as_row(self):
        return dict(bvid=self.bvid, batch=self.batch, t=self.t, lam=self.lam,
                    thr=self.thr, y=self.y, win_start=self.win_start,
                    win_end=self.win_end, K=self.K)


def label_points(seq: Sequence, *, pre=PRE, h=H, step=None,
                 min_count=MIN_COUNT, sigma=SIGMA):
    """在单条序列上生成全部合法决策点标签。

    阈值在 t 处**只算一次**，未来窗口只用于比较（不在窗口内滚动重算）。
    """
    ts = decision_times(seq.K, pre=pre, h=h, step=step)
    out = []
    for t in ts:
        t = int(t)
        lam = float(seq.c[t - pre:t].mean())
        thr = threshold(lam, min_count=min_count, sigma=sigma)
        fut = seq.c[t:t + h]
        y = int(fut.size == h and float(fut.max()) >= thr)
        out.append(Point(seq.bvid, seq.batch, t, lam, thr, y, t, t + h, seq.K))
    return out


# ══════════════════════════════════════════════════════════════════════
# 3. 特征（白名单）
# ══════════════════════════════════════════════════════════════════════
def entropy12(win) -> float:
    """ent = -Σ p_k log p_k / log(12)，p_k 在**全部 12 桶**上归一。

    全零窗口 → 0.0；单桶集中 → 0.0；均匀 → 1.0。
    """
    w = np.asarray(win, dtype=float)
    s = float(w.sum())
    if s <= 0:
        return 0.0
    p = w / s
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / math.log(len(w)))


def since_censored(seq: Sequence, t, *, pre=PRE, h=H) -> tuple:
    """D2 滚动参照：距上次"同类爆发"（e_k 与标签同函数形式）的时间。

    e_k = 1[c[k] >= max(3, λ_k + 2√λ_k)]，λ_k = mean(c[k-PRE:k])，只用 k <= t-1。
    只在**可观察的 12 桶**内查找：k ∈ [max(PRE, t-PRE), t)。
    无事件 → since = PRE, censored = 1（首个决策点 t=PRE 必然如此）。
    """
    t = int(t)
    lo = max(pre, t - pre)
    for k in range(t - 1, lo - 1, -1):
        lam_k = float(seq.c[k - pre:k].mean())
        if float(seq.c[k]) >= threshold(lam_k):
            return float(t - k), 0.0
    return float(pre), 1.0


def feature_row(seq: Sequence, t, *, pre=PRE, feature_shift=0) -> dict:
    """单点特征。全部输入只用 c[0:t]（`feature_shift>0` 仅用于负对照，把窗口整体前移）。"""
    t = int(t)
    te = t - int(feature_shift)          # 负对照：窗口整体前移
    if te - pre < 0:
        raise ProtocolError(f"feature_shift={feature_shift} 超出可观察范围（t={t}, PRE={pre}）")
    win = seq.c[te - pre:te].astype(float)
    lam = float(win.mean())
    recent = float(seq.c[te - 6:te].mean())
    older = float(seq.c[te - pre:te - 6].mean())
    since, cens = since_censored(seq, te, pre=pre)
    return dict(
        log1p_lam=math.log1p(lam),
        slope=math.log1p(recent) - math.log1p(older),   # 对数差分，无 epsilon
        since=since,
        censored=cens,
        prog=te / float(seq.K),
        ent=entropy12(win),
        log1p_duration=math.log1p(seq.duration),
        lam=lam,
    )


def select_features(rows, columns) -> np.ndarray:
    """白名单取列 + 黑名单断言。非法列**立即失败**，不静默丢弃。"""
    cols = tuple(columns)
    assert_whitelist(cols)
    try:
        X = np.asarray([[r[c] for c in cols] for r in rows], dtype=float)
    except KeyError as exc:
        raise ProtocolError(f"特征列缺失: {exc}") from exc
    if not np.isfinite(X).all():
        bad = [c for i, c in enumerate(cols) if not np.isfinite(X[:, i]).all()]
        raise ProtocolError(f"特征含非有限值: {bad}")
    return X


def assert_whitelist(columns) -> None:
    cols = tuple(columns)
    if not cols:
        raise ProtocolError("特征白名单为空")
    bad = [c for c in cols if c in BANNED]
    if bad:
        raise ProtocolError(f"禁用特征进入模型: {bad}")
    allowed = set(FEATURES_M2) | {"ent"}
    unknown = [c for c in cols if c not in allowed]
    if unknown:
        raise ProtocolError(f"未获批准的特征: {unknown}（允许: {sorted(allowed)}）")


def model_features(name: str) -> tuple:
    if name not in MODEL_SPECS:
        raise ProtocolError(f"未知模型 {name}；允许 {sorted(MODEL_SPECS)}")
    feats = MODEL_SPECS[name]
    if feats is None:
        raise ProtocolError(f"模型 {name} 未获批准定义（协议无正文），只保留接口，不得训练")
    return tuple(feats)


# ══════════════════════════════════════════════════════════════════════
# 4. 模型（Logistic + L2 + 标准化，C=1.0；不搜索超参）
# ══════════════════════════════════════════════════════════════════════
@dataclass
class FittedModel:
    scaler: object
    clf: object
    features: tuple
    fit_videos: frozenset = frozenset()
    n_fit_points: int = 0

    def predict_proba(self, X):
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        return self.clf.predict_proba(Z)[:, 1]


def fit_model(X, y, features, *, C=1.0, fit_videos=None, n_fit_points=None) -> FittedModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    if X.ndim != 2 or X.shape[1] != len(features):
        raise ProtocolError(f"X 形状 {X.shape} 与特征数 {len(features)} 不符")
    if np.unique(y).size < 2:
        raise ProtocolError("训练集只有单一类别，拒绝拟合")
    assert_whitelist(features)
    scaler = StandardScaler().fit(X)            # 只在传入的（训练）X 上 fit
    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")   # 默认 L2
    clf.fit(scaler.transform(X), y)
    return FittedModel(scaler, clf, tuple(features),
                       frozenset(fit_videos or ()), n_fit_points)


# ══════════════════════════════════════════════════════════════════════
# 5. 评估：视频宏平均视频内 AUC + 配对视频级 Bootstrap
# ══════════════════════════════════════════════════════════════════════
def auc(y, p) -> float:
    """单视频内 AUC（Mann–Whitney，平均秩处理并列）。要求两类齐全。"""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if np.unique(y).size < 2:
        raise ProtocolError("单类视频不得计算/填补 AUC")
    return float(roc_auc_score(y, p))


def macro_within_video_auc(y, p, vid) -> dict:
    """主指标：每视频一票的视频内 AUC 宏平均。单类视频**排除且不填补 0.5**。"""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    vid = np.asarray(vid, dtype=str)
    videos = np.unique(vid)
    per, excluded = {}, []
    for v in videos:
        m = vid == v
        if np.unique(y[m]).size < 2:
            excluded.append(v)
            continue
        per[v] = auc(y[m], p[m])
    vals = np.array(list(per.values()), dtype=float)
    n_el = int(vals.size)
    return dict(
        metric="macro_within_video_auc",
        macro=float(vals.mean()) if n_el else float("nan"),
        per_video=per,
        n_videos=int(videos.size),
        n_eligible=n_el,
        n_excluded_single_class=len(excluded),
        eligibility_share=(n_el / videos.size) if videos.size else 0.0,
        quantiles=dict(
            p10=float(np.percentile(vals, 10)) if n_el else float("nan"),
            p50=float(np.percentile(vals, 50)) if n_el else float("nan"),
            p90=float(np.percentile(vals, 90)) if n_el else float("nan"),
        ) if n_el else {},
    )


def pooled_auc(y, p) -> float:
    return auc(y, p)


def paired_bootstrap_delta(y, p_a, p_b, vid, *, B=BOOT_B, seed=BOOT_SEED,
                           min_eligible=None) -> dict:
    """ΔAUC = macro(a) − macro(b) 的配对视频级 Bootstrap 95% CI。

    重采样单位 = bvid（整簇）；两个模型用**同一批**被抽中的视频；
    单次重采样若可用视频 < min_eligible 则丢弃并计数。
    """
    y = np.asarray(y, dtype=int)
    vid = np.asarray(vid, dtype=str)
    videos = np.unique(vid)
    d = {}
    for v in videos:
        m = vid == v
        if np.unique(y[m]).size < 2:
            continue
        d[v] = auc(y[m], np.asarray(p_a)[m]) - auc(y[m], np.asarray(p_b)[m])
    if not d:
        raise ProtocolError("没有可评估视频，无法做配对 Bootstrap")
    keys = np.array(sorted(d))
    deltas = np.array([d[k] for k in keys], dtype=float)
    if min_eligible is None:      # 一次重采样若丢掉一半以上的视频，视为退化样本
        min_eligible = max(2, int(round(0.5 * keys.size)))
    rng = np.random.default_rng(seed)
    reps, discarded = [], 0
    for _ in range(int(B)):
        samp = rng.choice(keys, size=keys.size, replace=True)
        if np.unique(samp).size < min_eligible:
            discarded += 1
            continue
        reps.append(float(np.mean([d[s] for s in samp])))
    reps = np.asarray(reps, dtype=float)
    if reps.size < 100:
        raise ProtocolError(f"有效重采样仅 {reps.size} 次（<100），拒绝报告 CI")
    return dict(delta=float(deltas.mean()), ci_lo=float(np.percentile(reps, 2.5)),
                ci_hi=float(np.percentile(reps, 97.5)), B=int(B), seed=int(seed),
                unit="bvid", n_eligible=int(keys.size), n_valid_reps=int(reps.size),
                n_discarded=int(discarded))


# ══════════════════════════════════════════════════════════════════════
# 6. 交叉拟合 + Platt 校准（只用 D-train）
# ══════════════════════════════════════════════════════════════════════
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def group_folds(vid, *, folds=CV_FOLDS, seed=CV_SEED) -> dict:
    """按 bvid 分组的分折（同一视频的全部决策点进同一折）。"""
    videos = np.unique(np.asarray(vid, dtype=str))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(videos)
    return {v: int(i % folds) for i, v in enumerate(perm)}


def cross_fit_scores(X, y, vid, features, *, folds=CV_FOLDS, seed=CV_SEED) -> dict:
    """折外（OOF）分数：每个点的分数都来自**没见过该视频**的基础模型。"""
    vid = np.asarray(vid, dtype=str)
    y = np.asarray(y, dtype=int)
    fold_of = group_folds(vid, folds=folds, seed=seed)
    fold_id = np.array([fold_of[v] for v in vid], dtype=int)
    oof = np.full(y.size, np.nan)
    assignments = []
    for f in range(folds):
        te = fold_id == f
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            raise ProtocolError(f"折 {f} 为空，无法交叉拟合")
        model = fit_model(X[tr], y[tr], features,
                          fit_videos=set(vid[tr]), n_fit_points=int(tr.sum()))
        oof[te] = model.predict_proba(X[te])
        assignments.append(dict(fold=f, n_train_points=int(tr.sum()), n_test_points=int(te.sum()),
                                n_train_videos=int(np.unique(vid[tr]).size),
                                n_test_videos=int(np.unique(vid[te]).size),
                                video_overlap=int(len(set(vid[tr]) & set(vid[te])))))
    if not np.isfinite(oof).all():
        raise ProtocolError("OOF 分数不完整")
    return dict(oof=oof, fold_id=fold_id, assignments=assignments,
                fit_split="train_inner", folds=folds, seed=seed)


def fit_platt(oof_scores, y, *, C=1e6) -> dict:
    """在 OOF 分数上拟合 Platt（1 维 logistic，不惩罚）。返回 a,b 与指标。"""
    from sklearn.linear_model import LogisticRegression
    z = _logit(oof_scores).reshape(-1, 1)
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        raise ProtocolError("校准集单一类别")
    lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    lr.fit(z, y)
    a = float(lr.coef_[0, 0])
    b = float(lr.intercept_[0])
    return dict(a=a, b=b, platt=lr, n=int(y.size), fit_split="train_inner_oof")


def apply_platt(calib, scores):
    z = _logit(scores)
    return 1.0 / (1.0 + np.exp(-(calib["a"] * z + calib["b"])))


def calibration_report(p, y, *, bins=10) -> dict:
    """校准斜率/截距 + 可靠性曲线 + Brier。斜率 = 对 logit(p) 做 1 维 logistic。"""
    from sklearn.linear_model import LogisticRegression
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=int)
    z = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=np.inf, max_iter=1000, solver="lbfgs").fit(z, y)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    base = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    brier0 = float(np.mean((base - y) ** 2))
    order = np.argsort(p, kind="mergesort")
    curve = []
    for chunk in np.array_split(order, bins):
        if chunk.size == 0:
            continue
        curve.append(dict(n=int(chunk.size), mean_p=float(p[chunk].mean()),
                          obs_rate=float(y[chunk].mean()),
                          lo=float(p[chunk].min()), hi=float(p[chunk].max())))
    return dict(slope=slope, intercept=intercept, brier=brier, brier_base=brier0,
                brier_skill=float(1 - brier / brier0) if brier0 > 0 else float("nan"),
                reliability=curve)


# ══════════════════════════════════════════════════════════════════════
# 7. 负对照（诊断，不是通过手段）
# ══════════════════════════════════════════════════════════════════════
def shuffle_labels_within_video(y, vid, *, seed=7) -> np.ndarray:
    """负对照①：**视频内**置换标签（保持每视频正负数与总体基率，破坏"点↔特征"对应）。"""
    y = np.asarray(y, dtype=int)
    vid = np.asarray(vid, dtype=str)
    out = y.copy()
    rng = np.random.default_rng(seed)
    for v in np.unique(vid):
        m = np.flatnonzero(vid == v)
        out[m] = rng.permutation(y[m])
    return out


def shift_feature_rows(rows, shift) -> list:
    """负对照②：窗口整体前移 `shift` 桶（标签不变）。"""
    out = []
    for r in rows:
        g = dict(r)
        g["_feature_shift"] = int(shift)
        out.append(g)
    return out


def build_rows(seqs, *, feature_shift=0, models=("M2",)) -> list:
    """把序列展开成决策点行（含标签与特征）。

    `feature_shift>0` 仅用于负对照（时间窗整体前移）：标签不变，窗口前移；
    前移后历史窗不足的点被**丢弃**（负对照的样本是原样本的子集，报告须写明）。
    """
    rows = []
    for seq in seqs:
        for pt in label_points(seq):
            if feature_shift and pt.t - int(feature_shift) < PRE:
                continue
            feats = feature_row(seq, pt.t, feature_shift=feature_shift)
            row = pt.as_row()
            row.update(feats)
            row["part_count"] = seq.part_count
            row["split"] = split_of(pt.batch)
            row["feature_shift"] = int(feature_shift)
            rows.append(row)
    return rows


def _pc_int(v):
    try:
        x = int(float(v))
    except (TypeError, ValueError):
        return None
    return x if x > 0 else None


def known_multip_bvids(rows) -> set:
    """`part_count > 1` 的视频集合（阳性证据；空值不计入，也不当作单 P）。"""
    return {r["bvid"] for r in rows if (_pc_int(r.get("part_count")) or 0) > 1}


def filter_multip(rows, policy=MULTIP_POLICY_PRIMARY) -> list:
    """A3：`keep` = 主分析（全部保留）；`exclude_known` = 敏感性（只排除已知多 P）。"""
    if policy not in MULTIP_POLICIES:
        raise ProtocolError(f"未知多 P 口径 {policy}；允许 {MULTIP_POLICIES}")
    if policy == "keep":
        return list(rows)
    bad = known_multip_bvids(rows)
    return [r for r in rows if r["bvid"] not in bad]


def eligible_video_ids(rows, *, min_points=MIN_POINTS, min_pos=MIN_POS,
                       min_neg=MIN_NEG) -> list:
    """评估资格：≥5 有效点、≥2 正、≥2 负（视频级）。"""
    agg = {}
    for r in rows:
        a = agg.setdefault(r["bvid"], [0, 0])
        a[0] += 1
        a[1] += int(r["y"])
    return sorted(v for v, (n, pos) in agg.items()
                  if n >= min_points and pos >= min_pos and (n - pos) >= min_neg)


# ══════════════════════════════════════════════════════════════════════
# 8. 数据入口（manifest 白名单，不 glob）
# ══════════════════════════════════════════════════════════════════════
def read_file_manifest(path) -> list:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def _sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def verify_manifest(manifest_csv, data_root, *, check_bytes=True) -> dict:
    """逐文件核验 sha256 / 字节数；差异**如实记录**，不重写清单。"""
    rows, ok = [], 0
    for r in read_file_manifest(manifest_csv):
        path = os.path.join(data_root, r["file"])
        guard_path(path)
        actual_bytes = os.path.getsize(path) if os.path.exists(path) else -1
        actual_sha = _sha256(path) if actual_bytes >= 0 else ""
        exp_bytes = int(float(r["bytes"])) if r.get("bytes") else -1
        status = "ok"
        if actual_bytes < 0:
            status = "missing"
        elif check_bytes and exp_bytes >= 0 and actual_bytes != exp_bytes:
            status = "size_mismatch"
        elif actual_sha.lower() != (r.get("sha256") or "").lower():
            status = "sha_mismatch"
        if status == "ok":
            ok += 1
        rows.append(dict(layer=r["layer"], batch=r["batch"], file=r["file"],
                         expected_bytes=exp_bytes, actual_bytes=actual_bytes,
                         expected_sha256=r.get("sha256", ""), actual_sha256=actual_sha,
                         status=status))
    return dict(n_files=len(rows), n_ok=ok, all_ok=(ok == len(rows)), rows=rows)


_ODS_COLS = ["bvid", "duration", "time_offset", "row_id", "part_count"]


def load_sequences(manifest_csv, data_root, *, batches=None, layer="ods") -> tuple:
    """按 manifest 白名单读取指定批次的 ODS（或 DWD），逐 (bvid,batch) 构序列。"""
    import pandas as pd

    layer = assert_single_layer([layer])
    want = tuple(batches or KNOWN_BATCHES)
    for b in want:
        split_of(b)  # 未登记/ D-conf 批次在此 raise
    rows = [r for r in read_file_manifest(manifest_csv)
            if r["layer"] == ("ods_parquet" if layer == "ods" else "dwd_parquet")
            and str(r["batch"]) in want]
    if not rows:
        raise ProtocolError(f"manifest 中没有 {layer} 层 / 批次 {want} 的文件")
    provenance = []
    seqs = []
    for batch in want:
        files = sorted(os.path.join(data_root, r["file"]) for r in rows if str(r["batch"]) == batch)
        frames = []
        for f in files:
            guard_path(f)
            frames.append(pd.read_parquet(f, columns=_ODS_COLS))
            provenance.append(dict(batch=batch, file=os.path.relpath(f, data_root),
                                   layer=layer, sha256=next(r["sha256"] for r in rows
                                                            if os.path.join(data_root, r["file"]) == f)))
        df = pd.concat(frames, ignore_index=True)
        for bvid, g in df.groupby("bvid", sort=True):
            seq = build_sequence(bvid, batch, g["time_offset"].to_numpy(),
                                 g["row_id"].to_numpy(), float(g["duration"].iloc[0]),
                                 part_count=_pc_int(g["part_count"].iloc[0]))
            seqs.append(seq)
    return seqs, provenance


def first_seen(records) -> dict:
    """bvid → 最早出现批次。**必须在未过滤的原始表上计算**（协议 §6.2-1）。"""
    out = {}
    for bvid, batch in records:
        b = str(batch)
        out[str(bvid)] = b if str(bvid) not in out else min(out[str(bvid)], b)
    return out


def split_leak_check(records, *, assigned_split=None) -> dict:
    """用原始表 first_seen 复核"训练视频不得在 dev 批次首次出现"。"""
    fs = first_seen(records)
    assigned = assigned_split or {}
    bad = []
    for bvid, batch in fs.items():
        if bvid in assigned and split_of(batch) == "dev" and assigned[bvid] == "train":
            bad.append(dict(bvid=bvid, first_seen=batch, assigned="train"))
    return dict(n_videos=len(fs), violations=bad, n_violations=len(bad))


def assert_train_only(rows) -> None:
    """训练/校准入口守卫：混入 dev 行立即失败。"""
    bad = sorted({r.get("split") for r in rows if r.get("split") != "train"})
    if bad:
        raise ProtocolError(f"训练入口只接受 D-train，收到 split={bad}")


# ══════════════════════════════════════════════════════════════════════
# 9. D-train-only 功效分析（最小可检测增量）
# ══════════════════════════════════════════════════════════════════════
POWER_GRID = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05)

# 双侧 95% + 80% 功效的正态分位（常量，不引入 scipy 依赖）
Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143


def mde_80(sd, n) -> float:
    """80% 功效下的最小可检测效应（正态近似，SD 已知）。"""
    return (Z_975 + Z_80) * float(sd) / math.sqrt(int(n))


def n_for_effect(sd, delta) -> int:
    """要在 80% 功效下检出 `delta`，所需视频数（向上取整）。"""
    if delta <= 0:
        raise ProtocolError("delta 必须为正")
    return int(math.ceil(((Z_975 + Z_80) * float(sd) / float(delta)) ** 2))


def power_analysis(train_rows, *, grid=POWER_GRID, reps=2000, seed=BOOT_SEED,
                   target_n=(134, 150, 200)) -> dict:
    """用 D-train 的 OOF 配对视频级 AUC 差估计 SD，再模拟不同真实增量的检出功效。

    统计单位 = 视频；效应量网格预先声明；**不按结果删格**。
    功效定义：双侧 95% CI 下界 > 0（等价于 |Δ̂| / SE > 1.96 且方向为正）。
    """
    # 评估资格（≥5 点、≥2 正、≥2 负）先过滤，再算主指标
    elig = set(eligible_video_ids(train_rows))
    n_total = len({r["bvid"] for r in train_rows})
    rows = [r for r in train_rows if r["bvid"] in elig]
    y = np.array([r["y"] for r in rows], dtype=int)
    vid = np.array([r["bvid"] for r in rows], dtype=str)

    out = dict(protocol_version=PROTOCOL_VERSION, split="train", n_points=int(y.size),
               n_videos=int(np.unique(vid).size), n_videos_total=int(n_total),
               eligibility_rule=">=5 points, >=2 pos, >=2 neg",
               grid=list(grid), reps=int(reps), seed=int(seed))
    scores = {}
    for name in (COMPARISON_MODEL, PRIMARY_MODEL):
        feats = model_features(name)
        X = select_features(rows, feats)
        cf = cross_fit_scores(X, y, vid, feats, folds=CV_FOLDS, seed=CV_SEED)
        m = macro_within_video_auc(y, cf["oof"], vid)
        scores[name] = cf["oof"]
        out[name] = dict(features=list(feats), macro_auc=m["macro"],
                         n_eligible=m["n_eligible"], n_videos=m["n_videos"],
                         eligibility_share=m["eligibility_share"],
                         per_video_quantiles=m["quantiles"])
    boot = paired_bootstrap_delta(y, scores[PRIMARY_MODEL], scores[COMPARISON_MODEL], vid)
    out["delta"] = dict(model_a=PRIMARY_MODEL, model_b=COMPARISON_MODEL, **boot)

    # 配对差的经验 SD（只用 D-train，只看离散度，不看方向）
    d = []
    for v in np.unique(vid):
        m = vid == v
        if np.unique(y[m]).size < 2:
            continue
        d.append(auc(y[m], scores[PRIMARY_MODEL][m]) - auc(y[m], scores[COMPARISON_MODEL][m]))
    d = np.asarray(d, dtype=float)
    out["paired_sd"] = float(d.std(ddof=1))
    out["paired_sd_n"] = int(d.size)

    sims = {}
    for n in target_n:
        se = out["paired_sd"] / math.sqrt(n)
        row = {}
        for delta in grid:
            rng = np.random.default_rng(seed + int(delta * 1e6) + n)
            draws = rng.normal(delta, out["paired_sd"], size=(reps, n))
            mean = draws.mean(axis=1)
            lo = mean - 1.96 * se
            row[f"{delta:.3f}"] = dict(power=float((lo > 0).mean()),
                                       mde_note=f"n={n}, SE={se:.4f}")
        sims[str(n)] = row
    out["power"] = sims
    out["mde_80pct"] = {str(n): mde_80(out["paired_sd"], n) for n in target_n}
    out["n_for_effect_0.02"] = n_for_effect(out["paired_sd"], 0.02)
    out["caveat"] = ("模拟功效基于 D-train 内部 OOF 的配对差离散度，"
                     "不等于 D-conf 的真实功效保证；D-conf 批次与训练批次不同源时 SD 只会更大")
    return out


def negative_controls(train_rows, train_seqs, *, seed=7, R=200, shift=6) -> dict:
    """负对照（D-train 内部，B1 方案）：N1 评估侧标签置换 + N5 仅时长。

    N1（保留）：真标签拟合得固定分数 `p`，在**每支视频内**均匀置换标签 R 次，
    报宏平均 AUC 的分布（均值 / 经验 SD / 分位）与并列率检查。
    N5（保留）：仅时长模型，视频内恒定 ⇒ 视频内 AUC 必须恰为 0.5。
    N6（探索性附注，非验收项）：特征窗整体前移 6 桶，只记录、不判定。
    N2（拟合侧标签置换）已按 2026-10-27 决策 **删除**，不再计算或输出。
    """
    elig = set(eligible_video_ids(train_rows))
    rows = [r for r in train_rows if r["bvid"] in elig]
    y = np.array([r["y"] for r in rows], dtype=int)
    vid = np.array([r["bvid"] for r in rows], dtype=str)

    X = select_features(rows, FEATURES_M2)
    base = cross_fit_scores(X, y, vid, FEATURES_M2)
    base_auc = macro_within_video_auc(y, base["oof"], vid)["macro"]

    # ── N1：评估侧置换（分数固定，只打乱标签）──────────────────────
    true_model = fit_model(X, y, FEATURES_M2, fit_videos=set(vid))
    p_true = true_model.predict_proba(X)

    # 并列率：每视频 p 的唯一值数 / 点数（协议要求 ≥0.9）
    tie = {}
    for v in np.unique(vid):
        m = vid == v
        tie[v] = float(np.unique(p_true[m]).size / int(m.sum()))
    tie_min = float(min(tie.values())) if tie else float("nan")
    tie_share_ok = float(np.mean([t >= 0.9 for t in tie.values()])) if tie else float("nan")

    rng = np.random.default_rng(seed)          # 单一预注册种子，R 次共用一条序列
    idx_by_v = {v: np.flatnonzero(vid == v) for v in np.unique(vid)}
    perm_aucs = []
    for _ in range(int(R)):
        yp = y.copy()
        for v, m in idx_by_v.items():
            yp[m] = rng.permutation(y[m])
        perm_aucs.append(macro_within_video_auc(yp, p_true, vid)["macro"])
    perm_aucs = np.asarray(perm_aucs, dtype=float)
    n1_mean = float(perm_aucs.mean())
    n1_sd = float(perm_aucs.std(ddof=1))
    n1 = dict(R=int(R), seed=int(seed), mean=n1_mean, sd=n1_sd,
              p05=float(np.percentile(perm_aucs, 5)),
              p50=float(np.percentile(perm_aucs, 50)),
              p95=float(np.percentile(perm_aucs, 95)),
              tie_rate_min=tie_min, tie_share_ge_0_9=tie_share_ok,
              mean_ok=bool(0.49 <= n1_mean <= 0.51),
              sd_ok=bool(0.008 <= n1_sd <= 0.020),
              tie_ok=bool(tie_min >= 0.9),
              passed=bool(0.49 <= n1_mean <= 0.51 and 0.008 <= n1_sd <= 0.020
                          and tie_min >= 0.9))

    # ── N5：仅时长（视频内恒定）────────────────────────────────────
    Xd = select_features(rows, FEATURES_DURATION_ONLY)
    dur_model = fit_model(Xd, y, FEATURES_DURATION_ONLY, fit_videos=set(vid))
    dur_auc = macro_within_video_auc(y, dur_model.predict_proba(Xd), vid)["macro"]
    n5 = dict(macro_auc=dur_auc, exact_ok=bool(abs(dur_auc - 0.5) < 1e-9))

    # ── N6：探索性（不判定）────────────────────────────────────────
    shifted_rows = [r for r in build_rows(train_seqs, feature_shift=shift) if r["bvid"] in elig]
    ys = np.array([r["y"] for r in shifted_rows], dtype=int)
    vs = np.array([r["bvid"] for r in shifted_rows], dtype=str)
    Xs = select_features(shifted_rows, FEATURES_M2)
    shift_model = fit_model(Xs, ys, FEATURES_M2, fit_videos=set(vs))
    shift_auc = macro_within_video_auc(ys, shift_model.predict_proba(Xs), vs)["macro"]

    return dict(protocol_version=PROTOCOL_VERSION,
                n_videos=int(np.unique(vid).size), n_points=int(y.size),
                base_macro_auc=base_auc, n1=n1, n5=n5,
                n6_exploratory=dict(shift_buckets=int(shift), macro_auc=shift_auc,
                                    n_points=int(ys.size), n_videos=int(np.unique(vs).size),
                                    note="探索性附注，不参与验收，不得更换平移距离寻找下降"),
                verdict=dict(n1_passed=n1["passed"], n5_passed=n5["exact_ok"]),
                note="负对照是管道自检；不达标说明管道有问题，不得据此调整主模型")


# ══════════════════════════════════════════════════════════════════════
# 10. D-conf 事件门槛 count-only 模拟
# ══════════════════════════════════════════════════════════════════════
GATE_N_GRID = (150, 200, 250, 300, 400, 500)


def gate_simulation(train_rows, *, n_grid=GATE_N_GRID, reps=2000, seed=BOOT_SEED) -> dict:
    """用 D-train 的视频级(点数, 正例数)经验分布，模拟不同 D-conf 视频数下的门槛表现。

    只做计数，不训练、不评估；不把相关窗口当独立样本。
    """
    per = {}
    for r in train_rows:
        a = per.setdefault(r["bvid"], [0, 0])
        a[0] += 1
        a[1] += int(r["y"])
    keys = np.array(sorted(per))
    npts = np.array([per[k][0] for k in keys], dtype=float)
    pos = np.array([per[k][1] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    out = dict(protocol_version=PROTOCOL_VERSION, split="train",
               n_train_videos=int(keys.size), reps=int(reps), seed=int(seed),
               grid=list(n_grid), per_video=dict(
                   points_p50=float(np.percentile(npts, 50)), points_p90=float(np.percentile(npts, 90)),
                   pos_rate=float(pos.sum() / npts.sum())))
    rows = []
    for n in n_grid:
        elig, tot_pos, tot_neg, single = [], [], [], []
        for _ in range(reps):
            idx = rng.integers(0, keys.size, size=n)
            p, q = npts[idx], pos[idx]
            neg = p - q
            ok = (p >= MIN_POINTS) & (q >= MIN_POS) & (neg >= MIN_NEG)
            elig.append(int(ok.sum()))
            tot_pos.append(int(q.sum()))
            tot_neg.append(int(neg.sum()))
            single.append(float((((q == 0) | (neg == 0))).mean()))
        elig = np.asarray(elig); tp = np.asarray(tot_pos); tn = np.asarray(tot_neg)
        single = np.asarray(single)
        rows.append(dict(
            n_conf_videos=int(n),
            eligible_p05=float(np.percentile(elig, 5)), eligible_p50=float(np.percentile(elig, 50)),
            eligible_p95=float(np.percentile(elig, 95)),
            p_eligible_ge30=float((elig >= MIN_ELIGIBLE_VIDEOS).mean()),
            pos_p05=float(np.percentile(tp, 5)), pos_p50=float(np.percentile(tp, 50)),
            neg_p05=float(np.percentile(tn, 5)), neg_p50=float(np.percentile(tn, 50)),
            single_class_share_p50=float(np.percentile(single, 50)),
            p_single_class_share_gt_20pct=float((single > 0.2).mean()),
        ))
    out["rows"] = rows
    rec = [r for r in rows if r["p_eligible_ge30"] >= 0.95]
    if rec:
        best = rec[0]
        out["recommendation"] = dict(
            n_conf_videos=best["n_conf_videos"],
            min_pos_windows=int(math.floor(best["pos_p05"])),
            min_neg_windows=int(math.floor(best["neg_p05"])),
            basis=f"n={best['n_conf_videos']} 时 P(可评估>=30)={best['p_eligible_ge30']:.2f}，"
                  f"取正/负窗口总数的 5% 分位向下取整",
            caveat="计数门槛只保证'AUC 可算'，不保证 CI 半宽 <=0.02；后者由视频数与 SD 决定")
    else:
        out["recommendation"] = None
    return out


# ══════════════════════════════════════════════════════════════════════
# 10.5 正式确认运行入口（--stage confirm）
# ══════════════════════════════════════════════════════════════════════
ALLOWED_EVAL_SPLITS = ("dev",)


def run_confirm(train_seqs, eval_seqs, *, eval_split="dev",
                multip_policy=MULTIP_POLICY_PRIMARY, reps=BOOT_B,
                seed=BOOT_SEED, cv_folds=CV_FOLDS) -> dict:
    """在指定独立评估集上执行主分析链：拟合 M1/M2 → 视频内 macro AUC → 配对 bootstrap → Platt 校准。

    边界（硬）：
      * `eval_split` 只接受 "dev"；"dconf" 一律 raise —— D-conf 路径需冻结后单独授权并改码解锁。
      * 训练集只接受 D-train 批次；评估集只接受与 `eval_split` 一致的批次。
      * 训练与评估的视频集合必须不相交（独立构造后求交）。
      * 本函数不读文件、不写文件、不判断协议是否冻结；这些由调用方负责。
    """
    if eval_split == "dconf":
        raise DataIsolationError(
            "D-conf 评估路径未解锁：需协议冻结后单独授权，并显式修改本入口")
    if eval_split not in ALLOWED_EVAL_SPLITS:
        raise ProtocolError(f"未知 eval_split={eval_split}；允许 {ALLOWED_EVAL_SPLITS}")

    for s in train_seqs:
        if split_of(s.batch) != "train":
            raise ProtocolError(f"训练集混入非 D-train 批次: {s.bvid}/{s.batch}")
    for s in eval_seqs:
        if split_of(s.batch) != eval_split:
            raise ProtocolError(f"评估集混入非 {eval_split} 批次: {s.bvid}/{s.batch}")
    train_v = {s.bvid for s in train_seqs}
    eval_v = {s.bvid for s in eval_seqs}
    if train_v & eval_v:
        raise ProtocolError(f"训练/评估视频重叠: {sorted(train_v & eval_v)[:5]}")

    all_eval = {r["bvid"] for r in build_rows(eval_seqs)}
    train_rows = filter_multip(build_rows(train_seqs), multip_policy)
    eval_rows_all = filter_multip(build_rows(eval_seqs), multip_policy)
    tr_elig = set(eligible_video_ids(train_rows))
    ev_elig = set(eligible_video_ids(eval_rows_all))
    train_rows = [r for r in train_rows if r["bvid"] in tr_elig]
    eval_rows = [r for r in eval_rows_all if r["bvid"] in ev_elig]
    if not train_rows or not eval_rows:
        raise ProtocolError("训练或评估侧没有合格决策点")

    y_tr = np.array([r["y"] for r in train_rows], dtype=int)
    v_tr = np.array([r["bvid"] for r in train_rows], dtype=str)
    y_ev = np.array([r["y"] for r in eval_rows], dtype=int)
    v_ev = np.array([r["bvid"] for r in eval_rows], dtype=str)

    results, calib, scores = {}, {}, {}
    for name in (COMPARISON_MODEL, PRIMARY_MODEL):
        feats = model_features(name)
        X_tr = select_features(train_rows, feats)
        X_ev = select_features(eval_rows, feats)
        model = fit_model(X_tr, y_tr, feats, fit_videos=set(v_tr),
                          n_fit_points=int(y_tr.size))
        p_ev = model.predict_proba(X_ev)
        scores[name] = p_ev
        m = macro_within_video_auc(y_ev, p_ev, v_ev)
        results[name] = dict(features=list(feats), macro_auc=m["macro"],
                             n_videos=m["n_videos"], n_eligible=m["n_eligible"],
                             n_excluded_single_class=m["n_excluded_single_class"],
                             eligibility_share=m["eligibility_share"],
                             per_video_quantiles=m["quantiles"],
                             coef=np.asarray(model.clf.coef_, dtype=float).ravel().tolist(),
                             n_iter=int(model.clf.n_iter_[0]))

    delta = paired_bootstrap_delta(y_ev, scores[PRIMARY_MODEL],
                                   scores[COMPARISON_MODEL], v_ev, B=reps, seed=seed)

    # 校准：D-train 内部 5 折 OOF → Platt；只在评估集上评价校准表现
    X_m2 = select_features(train_rows, model_features(PRIMARY_MODEL))
    cf = cross_fit_scores(X_m2, y_tr, v_tr, model_features(PRIMARY_MODEL),
                          folds=cv_folds, seed=CV_SEED)
    platt = fit_platt(cf["oof"], y_tr)
    p_cal = apply_platt(platt, scores[PRIMARY_MODEL])
    calib = dict(fit_split=platt["fit_split"], platt_a=platt["a"], platt_b=platt["b"],
                 n_oof=int(platt["n"]),
                 eval_on=eval_split,
                 report=calibration_report(p_cal, y_ev))

    per_video = {}
    for v in np.unique(v_ev):
        m = v_ev == v
        per_video[v] = dict(points=int(m.sum()), pos=int(y_ev[m].sum()),
                            neg=int(m.sum() - y_ev[m].sum()))

    return dict(
        protocol_version=PROTOCOL_VERSION, stage="confirm", eval_split=eval_split,
        multip_policy=multipolicy_note(multip_policy),
        params=dict(boot_B=int(reps), seed=int(seed), cv_folds=int(cv_folds),
                    calib_seed=int(CALIB_SEED)),
        counts=dict(train_videos=int(np.unique(v_tr).size), train_points=int(y_tr.size),
                    train_videos_with_points=int(len(train_v)),
                    eval_videos_with_points=int(len(all_eval)),
                    eval_videos_eligible=int(np.unique(v_ev).size),
                    eval_points=int(y_ev.size),
                    eval_videos_excluded_by_gate=int(len({r["bvid"] for r in eval_rows_all})
                                                     - len(ev_elig)),
                    eval_videos_excluded_by_multip=int(len(all_eval)
                                                       - len({r["bvid"] for r in eval_rows_all}))),
        models=results, delta=delta, calibration=calib,
        per_video=per_video,
        warnings=["eval_split=dev：这是开发集评估，不构成正式结果",
                  "校准器只在 D-train 的 OOF 分数上拟合；评估集只用于评价",
                  "D-conf 路径未解锁（eval_split=dconf 会直接 raise）"],
    )


def multipolicy_note(policy) -> str:
    if policy not in MULTIP_POLICIES:
        raise ProtocolError(f"未知多 P 口径 {policy}")
    return policy


# ══════════════════════════════════════════════════════════════════════
# 11. 路径级冻结凭据（工作区不洁净时的替代方案）
# ══════════════════════════════════════════════════════════════════════
EXP35_FREEZE_PATHS = (
    "jobs/m2_l6_burst_risk.py",
    "tests/test_burst_label.py",
    "tests/test_burst_features.py",
    "tests/test_burst_split.py",
    "tests/test_burst_calibration.py",
    "tests/test_burst_eval.py",
    "tests/test_burst_isolation.py",
    "docs/exp35_508/file_manifest.csv",
    "docs/exp35_508/manifest.sha256",
    "docs/exp35_508/D-train-508_bvids.txt",
    "docs/exp35_508/D-dev-508_bvids.txt",
    "docs/exp35_508/35_formal_spec_rc5_candidate.md",
    "docs/exp35_508/exp35_freeze_and_isolation_plan.md",
)


def freeze_digest(paths=EXP35_FREEZE_PATHS, root=".") -> dict:
    """对冻结路径集合逐个算 sha256，再对 (path, sha256) 排序后求总摘要。

    用途：工作区存在与本实验无关的修改时，**用路径级摘要**代替 `git_status_clean`。
    任一文件缺失即 raise（不允许"冻结时少一两个文件"）。
    """
    entries = []
    for rel in paths:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            raise ProtocolError(f"冻结路径缺失: {rel}")
        entries.append(dict(path=rel, bytes=os.path.getsize(p), sha256=_sha256(p)))
    entries.sort(key=lambda e: e["path"])
    h = hashlib.sha256()
    for e in entries:
        h.update(f"{e['path']}:{e['sha256']}\n".encode("utf-8"))
    return dict(n_files=len(entries), digest=h.hexdigest(), entries=entries,
                scheme="sha256 over sorted 'path:sha256' lines")


# ══════════════════════════════════════════════════════════════════════
# 12. CLI
# ══════════════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "docs", "exp35_508", "file_manifest.csv")
DEFAULT_DATA_ROOT = "/home/molly/projects/danmaku"


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=float)
    return path


def _split_audit(seqs_by_batch) -> dict:
    """两侧**独立构造**视频集合后再求交（不得用 tr = ~te 的恒等式）。"""
    train_v = {s.bvid for s in seqs_by_batch if split_of(s.batch) == "train"}
    dev_v = {s.bvid for s in seqs_by_batch if split_of(s.batch) == "dev"}
    return dict(train_videos=len(train_v), dev_videos=len(dev_v),
                intersection=sorted(train_v & dev_v),
                n_intersection=len(train_v & dev_v),
                train_batches=list(D_TRAIN_BATCHES), dev_batches=list(D_DEV_BATCHES),
                d_train_end="2026-09-07",
                constructed_independently=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exp35 正式实现（H=6）")
    ap.add_argument("--stage", default="audit",
                    choices=["audit", "build", "power", "gate-sim", "negctl",
                             "confirm", "verify-manifest", "freeze-digest"])
    ap.add_argument("--data-root", default=os.environ.get("EXP35_DATA_ROOT", DEFAULT_DATA_ROOT))
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--layer", default="ods", choices=["ods", "dwd"])
    ap.add_argument("--multip-policy", default=MULTIP_POLICY_PRIMARY,
                    choices=list(MULTIP_POLICIES),
                    help="A3：keep=主分析（默认）；exclude_known=敏感性")
    ap.add_argument("--eval-split", default="dev", choices=["dev", "dconf"],
                    help="confirm 的评估集；dconf 未解锁，会直接 raise")
    ap.add_argument("--reps", type=int, default=2000)
    args = ap.parse_args(argv)

    layer = assert_single_layer([args.layer])
    out_dir = args.out_dir or os.path.join(ROOT, output_dir_for(layer))
    os.makedirs(out_dir, exist_ok=True)
    guard_path(args.manifest)

    if args.stage == "freeze-digest":
        res = freeze_digest(root=ROOT)
        _dump(res, os.path.join(out_dir, "freeze_digest.json"))
        print(json.dumps({k: res[k] for k in ("n_files", "digest", "scheme")},
                         ensure_ascii=False, indent=2))
        return 0

    if args.stage == "verify-manifest":
        res = verify_manifest(args.manifest, args.data_root)
        _dump(res, os.path.join(out_dir, "manifest_verify.json"))
        print(f"manifest: {res['n_ok']}/{res['n_files']} ok, all_ok={res['all_ok']}")
        return 0 if res["all_ok"] else 9

    if args.stage == "audit":
        ver = verify_manifest(args.manifest, args.data_root)
        seqs, prov = load_sequences(args.manifest, args.data_root, layer=layer)
        rows = build_rows(seqs)
        by_split = {}
        for sp in ("train", "dev"):
            sub = [r for r in rows if r["split"] == sp]
            el = eligible_video_ids(sub)
            el_sens = eligible_video_ids(filter_multip(sub, "exclude_known"))
            by_split[sp] = dict(
                videos=len({r["bvid"] for r in sub}), points=len(sub),
                pos=int(sum(r["y"] for r in sub)),
                rate=(sum(r["y"] for r in sub) / len(sub)) if sub else None,
                eligible_videos=len(el), eligible_ids=el,
                eligible_videos_excl_known_multip=len(el_sens))
        res = dict(protocol_version=PROTOCOL_VERSION, layer=layer, data_root=args.data_root,
                   manifest=args.manifest, manifest_verify={k: ver[k] for k in ("n_files", "n_ok", "all_ok")},
                   n_sequences=len(seqs), n_videos=len({s.bvid for s in seqs}),
                   split=_split_audit(seqs), by_split=by_split,
                   dropped={k: int(sum(s.audit.get(k, 0) for s in seqs))
                            for k in ("rows_in", "dropped_nan", "dropped_negative",
                                      "dropped_beyond_duration", "dropped_duplicate_dmid",
                                      "clipped_to_last_bucket", "rows_kept")},
                   provenance=prov)
        res["manifest_rows"] = ver["rows"]
        _dump(res, os.path.join(out_dir, "audit.json"))
        with open(os.path.join(out_dir, "audit_manifest_files.csv"), "w",
                  encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ver["rows"][0]))
            w.writeheader()
            w.writerows(ver["rows"])
        print(json.dumps({k: res[k] for k in ("n_sequences", "n_videos", "split", "dropped")},
                         ensure_ascii=False, indent=2))
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "eligible_ids"}
                          for k, v in by_split.items()}, ensure_ascii=False, indent=2))
        return 0

    # 以下阶段只用 D-train
    seqs, prov = load_sequences(args.manifest, args.data_root,
                               batches=D_TRAIN_BATCHES, layer=layer)
    train_rows = filter_multip(build_rows(seqs), args.multip_policy)

    if args.stage == "build":
        import pandas as pd
        df = pd.DataFrame(train_rows)
        path = os.path.join(out_dir, "points_train.parquet")
        df.to_parquet(path, index=False)
        summary = dict(protocol_version=PROTOCOL_VERSION, layer=layer, n_rows=len(df),
                       n_videos=df.bvid.nunique(), pos=int(df.y.sum()),
                       rate=float(df.y.mean()), split="train",
                       eligible_videos=len(eligible_video_ids(train_rows)),
                       output=path, provenance=prov)
        _dump(summary, os.path.join(out_dir, "build.json"))
        print(json.dumps({k: summary[k] for k in ("n_rows", "n_videos", "pos", "rate",
                                                  "eligible_videos")}, ensure_ascii=False))
        return 0

    if args.stage == "power":
        res = power_analysis(train_rows, reps=args.reps)
        res["multip_policy"] = args.multip_policy
        _dump(res, os.path.join(out_dir, "power_dtrain.json"))
        print(json.dumps({k: res[k] for k in ("M1", "M2", "delta", "paired_sd",
                                              "mde_80pct")}, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "negctl":
        res = negative_controls(train_rows, seqs)
        _dump(res, os.path.join(out_dir, "negative_controls_dtrain.json"))
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    if args.stage == "confirm":
        eval_batches = {"dev": D_DEV_BATCHES}.get(args.eval_split)
        if args.eval_split == "dconf":
            raise DataIsolationError("D-conf 评估路径未解锁：需协议冻结后单独授权")
        eval_seqs, eval_prov = load_sequences(args.manifest, args.data_root,
                                              batches=eval_batches, layer=layer)
        res = run_confirm(seqs, eval_seqs, eval_split=args.eval_split,
                          multip_policy=args.multip_policy, reps=args.reps)
        res["data"] = dict(manifest=os.path.relpath(args.manifest, ROOT),
                           data_root=args.data_root, layer=layer,
                           train_batches=list(D_TRAIN_BATCHES),
                           eval_batches=list(eval_batches),
                           eval_files=[p["file"] for p in eval_prov])
        _dump(res, os.path.join(out_dir, f"confirm_{args.eval_split}.json"))
        print(json.dumps({k: res[k] for k in ("eval_split", "counts", "delta")},
                         ensure_ascii=False, indent=2))
        print(json.dumps({k: {"macro_auc": v["macro_auc"], "n_eligible": v["n_eligible"]}
                          for k, v in res["models"].items()}, ensure_ascii=False))
        print("calibration:", json.dumps(
            {k: res["calibration"]["report"][k] for k in ("slope", "intercept", "brier_skill")},
            ensure_ascii=False))
        return 0

    if args.stage == "gate-sim":
        res = gate_simulation(train_rows, reps=args.reps)
        res["multip_policy"] = args.multip_policy
        _dump(res, os.path.join(out_dir, "gate_sim_dtrain.json"))
        print(json.dumps(res["rows"], ensure_ascii=False, indent=2))
        print(json.dumps(res["recommendation"], ensure_ascii=False, indent=2))
        return 0

    raise ProtocolError(f"未知 stage {args.stage}")


if __name__ == "__main__":
    sys.exit(main())
