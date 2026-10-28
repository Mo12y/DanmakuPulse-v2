"""精彩时刻聚合载荷（PLAN_HIGHLIGHTS §2.6）。

把 4 张新 ads 表组装成一份前端载荷：
  ads_resonance_peaks        共振峰（秒级定位 + 对标）
  ads_critique_segments      质疑段（宁缺毋滥，未检出就是未检出）
  ads_video_feedback_profile 从众 / 中性 / 批判占比 + 分区对标
  ads_dict_critique          质疑词表版本
外加切片建议（长视频）/ 高光定位（短视频）与诚实边界。

约定：本模块只读 PG，不改既有端点；数据缺失返回 None，由 app.py 转 404，不静默换视频。
"""
import datetime
import json
import os

import psycopg

import diagnosis

DSN = os.environ.get('ADS_PG_DSN', '')

LONG_SEC = 1800            # >1800s 出切片建议，否则出高光定位（前端 is_long 同口径）
BENCH_MIN = 20             # 分区基准最小样本：不足退回全库基准，partition_ok=false

# 诚实边界（主视图常驻小字，不折叠）
CAVEATS = [
    '只测情绪共振与质疑，不测内容质量 —— 量大不等于内容好。',
    '不判定情绪：弹幕情绪词语义随内容语境漂移，系统只标「哪里起了反应 / 观众在刷什么」，不猜「这是什么情绪」。',
    '批判层可能很薄：未检出就是未检出，不放宽阈值凑数。',
    '「该切 / 该看」指情绪共振最强处，不等于内容最好处。',
    '复读近似 = 文本完全相等（变体可能漏检）；弹幕只反映观众反应，不解释内容含义。',
]

_FB_COLS = ('bvid, batch_date, n_danmaku, n_herd, n_neutral, n_critique, '
            'ratio_herd, ratio_neutral, ratio_critique, n_peaks, resonance_total, '
            'max_peak_score, duration, tname, partition_n, resonance_pctl, '
            'resonance_z, engine_version')
_PEAK_COLS = ('peak_id, start_sec, end_sec, peak_sec, peak_score, n_repeat, '
              'n_rep_texts, top_danmaku, top_counts, '
              'pos_ratio, neg_ratio, neu_ratio, in_opening, pctl, partition_n')
_CRIT_COLS = ('seg_id, start_sec, end_sec, critique_score, n_critique, '
              'top_danmaku, top_counts, neg_ratio, method, ml_score')


def _q(sql, params=()):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _q1(sql, params=()):
    rows = _q(sql, params)
    return rows[0] if rows else None


def _q_safe(sql, params=()):
    """导出作业还没跑过（表不存在）时按「没有数据」处理，不把 500 甩给前端。"""
    try:
        return _q(sql, params)
    except psycopg.errors.UndefinedTable:
        return []


def _i(v):
    return None if v is None else int(v)


def _f(v, nd=6):
    if v is None:
        return None
    f = float(v)
    if f != f:                     # NaN（基准缺失）→ null，避免 JSON 里出现 NaN
        return None
    return round(f, nd)


def _pairs(texts, counts):
    texts = list(texts or [])
    counts = list(counts or [])
    out = []
    for i, t in enumerate(texts):
        out.append({'text': t, 'n': _i(counts[i]) if i < len(counts) else None})
    return out


def _peak(row):
    return {
        'peak_id': _i(row.get('peak_id')),
        'start_sec': _i(row.get('start_sec')),
        'end_sec': _i(row.get('end_sec')),
        'peak_sec': _f(row.get('peak_sec'), 1),
        'score': _i(row.get('peak_score')),
        'n_repeat': _i(row.get('n_repeat')),
        'n_rep_texts': _i(row.get('n_rep_texts')),
        'top_danmaku': _pairs(row.get('top_danmaku'), row.get('top_counts')),
        'sentiment': {'pos': _f(row.get('pos_ratio')),
                      'neg': _f(row.get('neg_ratio')),
                      'neu': _f(row.get('neu_ratio'))},
        'in_opening': bool(row.get('in_opening')),
        'pctl': _f(row.get('pctl')),
        'partition_n': _i(row.get('partition_n')),
    }


def _crit_segment(row):
    return {
        'seg_id': _i(row.get('seg_id')),
        'start_sec': _i(row.get('start_sec')),
        'end_sec': _i(row.get('end_sec')),
        'score': _f(row.get('critique_score')),
        'n': _i(row.get('n_critique')),
        'top_danmaku': _pairs(row.get('top_danmaku'), row.get('top_counts')),
        'neg_ratio': _f(row.get('neg_ratio')),
        # 2026-10-20（G2）：来源与弱监督均分。'dict' = 词典命中，'ml' = 弱监督泛化（召回更高、可能误报）
        'method': row.get('method') or 'dict',
        'ml_score': _f(row.get('ml_score')),
    }


def _title(bvid, batch_date):
    """标题取榜单快照（新 4 表不带标题），再退回 v3 诊断快照；都缺失给空串，不编。

    榜单只覆盖 4 个批次（20260914 起），20260902/07 这类旧批次的标题在 v3 快照里。
    """
    row = _q1('SELECT title FROM ads_hotlist_history WHERE bvid = %s AND batch_date = %s',
              (bvid, batch_date))
    if row and row.get('title'):
        return row['title']
    row = _q1('SELECT title FROM ads_hotlist_history WHERE bvid = %s '
              'ORDER BY batch_date DESC LIMIT 1', (bvid,))
    if row and row.get('title'):
        return row['title']
    try:
        for r in diagnosis.index():
            if r.get('bvid') == bvid:
                return r.get('title') or ''
    except Exception:
        pass
    return ''


# dict_version 不在 PG 表里：detector 的作业把它写在 data-out/m2_l4_highlights_report.json，
# 这里优先读报告（fastapi 容器需挂 data-out），读不到再退回词典表 engine_version。
_REPORT_PATHS = (
    os.environ.get('HL_HIGHLIGHTS_REPORT') or '',
    '/app/data-out/m2_l4_highlights_report.json',
    '/opt/work/data-out/m2_l4_highlights_report.json',
)


def _dict_version():
    """优先读词典表（fastapi 容器没挂 data-out）；实在没有再看 detector 的报告 JSON。"""
    rows = _q_safe('SELECT engine_version, COUNT(*) AS n FROM ads_dict_critique '
                   'GROUP BY engine_version ORDER BY n DESC LIMIT 1')
    v = (rows[0] if rows else {}).get('engine_version')
    if v:
        return v
    for p in _REPORT_PATHS:
        if not p:
            continue
        try:
            with open(p, encoding='utf-8') as f:
                r = json.load(f)
            v = r.get('dict_version') or (r.get('dict') or {}).get('version')
            if v:
                return v
        except (OSError, ValueError):
            continue
    return None


def _benchmark(fb):
    """视频级对标：基准 = **每个 bvid 只取最新 batch** 的去重集合（Lead 2026-10-19 裁定）。

    跨批有 85 支视频出现两批，直接对 (bvid,batch) 全表算中位会给它们重复计权，
    所以先 DISTINCT ON (bvid) 取最新批次，再按 tname 取中位/样本数。
    partition_ok / partition_n 用 DWS 给的数（与 detector 口径一致）；fallback 用全库 latest 集，
    n_peers 在 partition_ok 时即 partition_n。
    """
    tname = fb.get('tname')
    pn = _i(fb.get('partition_n')) or 0
    ok = bool(tname) and pn >= BENCH_MIN
    base = ('SELECT COUNT(*) AS n, '
            'percentile_cont(0.5) WITHIN GROUP (ORDER BY max_peak_score) AS med '
            'FROM (SELECT DISTINCT ON (bvid) bvid, tname, max_peak_score '
            '      FROM ads_video_feedback_profile '
            '      ORDER BY bvid, batch_date DESC) t')
    if ok:
        ref = _q1(base + ' WHERE tname = %s', (tname,))
    else:
        ref = _q1(base)
    med = (ref or {}).get('med')
    return {
        'tname': tname,
        'partition_n': pn,
        'partition_ok': ok,
        'resonance_pctl': _f(fb.get('resonance_pctl')),
        'resonance_z': _f(fb.get('resonance_z')),
        'max_peak_score': _f(fb.get('max_peak_score')),
        'partition_med': _f(med),
        'n_peers': pn if ok else (_i((ref or {}).get('n')) or 0),
    }


def _pick_peaks(peaks, top=3):
    """切片/高光取峰：先只用非片头的峰（片头打卡噪声），不足 top 个才回落片头峰。

    评分口径：peak_score 降序、再按时间升序（可复现）。opening_only 标记该条是否来自片头回落。
    """
    ranked = sorted(peaks, key=lambda p: (-(p['score'] or 0), p['start_sec'] or 0))
    normal = [p for p in ranked if not p['in_opening']]
    opening = [p for p in ranked if p['in_opening']]
    picked = normal[:top]
    if len(picked) < top:
        picked = picked + opening[:top - len(picked)]
    return picked


def not_found_msg():
    row = _q1('SELECT COUNT(DISTINCT bvid) AS n FROM ads_video_feedback_profile')
    n = _i((row or {}).get('n')) or 0
    return '这支视频不在精彩时刻检测快照里（覆盖 %d 支）' % n


def card(bvid, batch_date=None):
    """组装一支视频的精彩时刻载荷；没有快照数据返回 None。"""
    sql = 'SELECT ' + _FB_COLS + ' FROM ads_video_feedback_profile WHERE bvid = %s'
    params = [bvid]
    if batch_date:
        try:
            params.append(int(str(batch_date).strip()))
        except (TypeError, ValueError):
            return None
        sql += ' AND batch_date = %s'
    else:
        sql += ' ORDER BY batch_date DESC LIMIT 1'
    fb = _q1(sql, tuple(params))
    if not fb:
        return None

    bd = fb['batch_date']
    peaks = [_peak(r) for r in _q_safe(
        'SELECT ' + _PEAK_COLS + ' FROM ads_resonance_peaks '
        'WHERE bvid = %s AND batch_date = %s ORDER BY start_sec, peak_id', (bvid, bd))]
    segments = [_crit_segment(r) for r in _q_safe(
        'SELECT ' + _CRIT_COLS + ' FROM ads_critique_segments '
        'WHERE bvid = %s AND batch_date = %s '
        'ORDER BY critique_score DESC, start_sec', (bvid, bd))]

    duration = _f(fb.get('duration'), 2)
    is_long = bool(duration and duration > LONG_SEC)
    picked = _pick_peaks(peaks)
    slices = [{'rank': i,
               'start_sec': p['start_sec'], 'end_sec': p['end_sec'],
               'peak_sec': p['peak_sec'],
               'score': p['score'], 'pctl': p['pctl'],
               'opening_only': p['in_opening'],
               'top_danmaku': p['top_danmaku']} for i, p in enumerate(picked, 1)]
    positions = [{'rank': i, 'sec': p['peak_sec'],
                  'start_sec': p['start_sec'], 'end_sec': p['end_sec'],
                  'score': p['score'], 'pctl': p['pctl'],
                  'opening_only': p['in_opening'],
                  # 卡片上直接给原始弹幕，让弹幕自己说话（不判定情绪，2026-10-19 用户拍板）
                  'top_danmaku': p['top_danmaku']}
                 for i, p in enumerate(picked, 1)]

    data = {
        'bvid': bvid,
        'batch_date': _i(bd),
        'title': _title(bvid, bd),
        'tname': fb.get('tname'),
        'duration': duration,
        'n_danmaku': _i(fb.get('n_danmaku')),
        'is_long': is_long,
        'peaks': peaks,
        'critique': {
            'detected': bool(segments),
            'n_segments': len(segments),
            'n_dict_segments': sum(1 for x in segments if x.get('method') == 'dict'),
            'n_ml_segments': sum(1 for x in segments if x.get('method') == 'ml'),
            'n_critique_danmaku': _i(fb.get('n_critique')),
            'segments': segments,
        },
        'feedback': {
            'herd': _f(fb.get('ratio_herd')),
            'neutral': _f(fb.get('ratio_neutral')),
            'critique': _f(fb.get('ratio_critique')),
            'n_danmaku': _i(fb.get('n_danmaku')),
            'n_herd': _i(fb.get('n_herd')),
            'n_neutral': _i(fb.get('n_neutral')),
            'n_critique': _i(fb.get('n_critique')),
        },
        'benchmark': _benchmark(fb),
        'slice_suggestions': slices if is_long else [],
        'highlight_positions': [] if is_long else positions,
        'caveats': list(CAVEATS),
    }
    meta = {
        'n_peaks': len(peaks),
        'n_critique_segments': len(segments),
        'engine_version': fb.get('engine_version'),
        'dict_version': _dict_version(),
        'generated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    return {'data': data, 'meta': meta}
