#!/usr/bin/env python3
"""
m4_track_common.py — M4 跨批轨迹公共纯函数（2026-10-14；m4_track / m4_evolution / 单测共用）

背景：B 站弹幕接口对饱和视频返回固定容量「最近 N 条」滑动窗（danmaku_got 量子化 600 倍数），
跨批水平/形状/半衰期一律改用可观测量（play/like/coin/favorite/share/reply/rank_proxy/
danmaku_reported）；danmaku_got 只作 coverage / pool_saturation 的分子。

纯函数、无 pyspark 依赖（集群经 --py-files 分发；宿主可单测）。
"""
import math


def geom_growth(first, last, span_days):
    """几何日均增长率：(last/first)^(1/span) − 1。
    first≤0 / span≤0 / 任一 None → None（不可定义，不猜）。"""
    if first is None or last is None or span_days is None:
        return None
    if first <= 0 or span_days <= 0:
        return None
    return float((last / first) ** (1.0 / span_days) - 1.0)


def build_trajectory(obs):
    """按采集时刻排序并剔除「批次倒挂」观测。

    obs: list[dict]，每项须含 'batch'(int) 与 't'(epoch 秒 float|None)。
    规则（2026-10-14 收口批次 ⑤）：
      - 全部 obs 都带 t：按 t 升序扫描，要求 batch 严格递增；违反者剔除（保留先到者）。
        —— 1012 第三窗口（10-13 17:43–18:45 续采）的 45 个视频，其「1012 观测」晚于
        「1013 观测」，跨批差分方向为负 ⇒ 由此规则自动剔除出轨迹。
      - 存在 t 缺失：退化按 batch 升序、不剔除（返回 assumed=True 供报告标注）。
    返回 (kept_sorted_list, n_dropped, assumed_order_bool)。
    """
    if not obs:
        return [], 0, False
    if any(o.get('t') is None for o in obs):
        kept = sorted(obs, key=lambda o: o['batch'])
        return kept, 0, True
    kept, dropped = [], 0
    for o in sorted(obs, key=lambda o: o['t']):
        if kept and o['batch'] <= kept[-1]['batch']:
            dropped += 1
            continue
        kept.append(o)
    return kept, dropped, False


def trajectory_features(kept):
    """轨迹特征（需 ≥2 个观测；len<2 → None）。全部来自指标，不用 danmaku_got。"""
    if len(kept) < 2:
        return None
    a, b = kept[0], kept[-1]
    span = ((b['t'] - a['t']) / 86400.0
            if a.get('t') is not None and b.get('t') is not None else None)

    def ratio(m):
        if b.get(m) is None or a.get(m) is None or a.get(m) <= 0:
            return None
        return b[m] / float(a[m])

    rp, rl = ratio('play'), ratio('like')
    g_play = geom_growth(a.get('play'), b.get('play'), span)
    g_like = geom_growth(a.get('like'), b.get('like'), span)
    g_reported = geom_growth(a.get('reported'), b.get('reported'), span)

    eng_shift = None
    if (a.get('play') and b.get('play') and a.get('like') is not None
            and b.get('like') is not None and a['play'] > 0 and b['play'] > 0):
        eng_shift = float(b['like'] / b['play'] - a['like'] / a['play'])

    rank_delta = None
    if a.get('rank') is not None and b.get('rank') is not None:
        rank_delta = int(a['rank'] - b['rank'])   # >0 = 名次上升（rank_proxy 越小越靠前）

    return {
        'n_obs': len(kept),
        'span_days': span,
        'g_play': g_play, 'g_like': g_like, 'g_reported': g_reported,
        'rank_first': a.get('rank'), 'rank_last': b.get('rank'),
        'rank_delta': rank_delta,
        'eng_shift': eng_shift,
        'sat_last': b.get('sat'),
        '_rp': rp, '_rl': rl,
    }


FEATS_V2 = ['g_play', 'g_like', 'g_reported', 'rank_delta', 'rank_last',
            'eng_shift', 'sat_last']
TYPES_V2 = ['快增型', '稳增型', '缓增型']


def name_types_v2(centers):
    """三型命名（规则固定、可复现）：按 g_play 质心降序 → [快增型, 稳增型, 缓增型]。"""
    gi = FEATS_V2.index('g_play')
    order = sorted(range(len(centers)), key=lambda i: -centers[i][gi])
    assert len(order) == len(TYPES_V2), 'k 与命名表不一致'
    return {int(order[i]): TYPES_V2[i] for i in range(len(order))}


def metric_halflife(obs, metric='play'):
    """度量增长半衰期（v0·实验性，2026-10-14 ④「半衰期改用指标」的最小实现）：

    把相邻观测的日增率 rate_i 视作随时间衰减（rate ∝ 2^(−t/τ)），对 log(rate)~t 做线性回归，
    τ = ln2 / (−slope)。要求 ≥3 个 rate（≥4 观测）且 slope<0；否则返回 (None, n)。
    返回 (tau_days|None, n_rates)。
    """
    ts, rates = [], []
    for a, b in zip(obs[:-1], obs[1:]):
        if a.get('t') is None or b.get('t') is None or a.get(metric) in (None, 0):
            continue
        dt = (b['t'] - a['t']) / 86400.0
        if dt <= 0 or b.get(metric) is None or b[metric] <= 0:
            continue
        v = (b[metric] / float(a[metric])) ** (1.0 / dt) - 1.0
        ts.append((a['t'] + b['t']) / 2.0 / 86400.0)
        rates.append(max(v, 1e-9))
    n = len(rates)
    if n < 3:
        return None, n
    mt, mlog = sum(ts) / n, sum(math.log(r) for r in rates) / n
    num = sum((t - mt) * (math.log(r) - mlog) for t, r in zip(ts, rates))
    den = sum((t - mt) ** 2 for t in ts)
    if den <= 0:
        return None, n
    slope = num / den
    if slope >= 0:
        return None, n
    return float(math.log(2) / (-slope)), n
