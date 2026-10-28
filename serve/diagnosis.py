# -*- coding: utf-8 -*-
'''v3 对标诊断供数逻辑（33 号 §3 四视图）

输入：serve/v3data/ 快照（由 jobs/export_v3_snapshot.py 只读导出，见该文件头注）
输出：一块 JSON，直接对应 33 号四视图：
  verdict    视图 4 一句话诊断 + 行动
  position   定位（百分位，不是 z）
  deviations 视图 2 偏离诊断（3–4 维，带方向 + 人话 + 建议）
  benchmarks 视图 3 相似对标（3 支 + 后来怎样 + 可学什么）
  curve      视图 1 对标对比图（你 / 同类中位 / p25-p75 包络 + 冲出掉出区间 + 爆点）

口径：
  - 偏离 z = (v − 分区中位) / (1.4826·MAD)   ← 与 32 号 P4 同口径
  - 相似度 = 形状 6 维在分区基准上 z 化后的余弦 ← 与 32 号 P1 同口径（within 0.992）
  - 曲线 = 每 1% 进度的弹幕占比（形状），展示前做 5 点滑动平均（33 号：不要 5s 毛刺）
  - 方向只来自 32 号 P3 控龄偏相关：量维有方向，形状维标"中性差异"
'''
import datetime
import json
import math
import os

from dict_zh import (ACTION_CAVEAT, CONCERN, CURVE_CAVEAT, DICT, DIR, DICT_VERSION,
                     human_pos, label_of, level_of, pole_word, pos_of, sim_label, trait_of)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3data')
SMOOTH_WIN = 5
BAND_MIN = 3
NEIGHBOR_SCAN = 12
NEIGHBOR_SHOW = 3
MIN_PARTITION = 20

_SNAP = None


def _smooth(s, w=SMOOTH_WIN):
    if not s:
        return []
    n = len(s)
    half = w // 2
    return [sum(s[max(0, i - half):min(n, i + half + 1)]) / (min(n, i + half + 1) - max(0, i - half))
            for i in range(n)]


def _cos(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _pctl(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    f = pos - lo
    return sorted_vals[lo] * (1 - f) + sorted_vals[hi] * f


def _load():
    global _SNAP
    if _SNAP is not None:
        return _SNAP
    d = {}
    for name in ('videos', 'curves', 'baselines', 'events', 'meta'):
        with open(os.path.join(DATA_DIR, name + '.json'), encoding='utf-8') as f:
            d[name] = json.load(f)
    by_bvid = {}
    for i, v in enumerate(d['videos']):
        b = v['bvid']
        if b not in by_bvid or v['batch_date'] > d['videos'][by_bvid[b]]['batch_date']:
            by_bvid[b] = i
    d['by_bvid'] = by_bvid
    d['by_key'] = {'%s|%s' % (v['bvid'], v['batch_date']): i for i, v in enumerate(d['videos'])}
    glob = {}
    for c in d['meta']['shape_dims']:
        vals = sorted(v['shape'][c] for v in d['videos'] if v['shape'].get(c) is not None)
        if not vals:
            continue
        med = _pctl(vals, 0.5)
        mad = _pctl(sorted(abs(v - med) for v in vals), 0.5) * 1.4826
        glob[c] = (med, mad if mad > 1e-9 else 1.0)
    d['global_shape'] = glob
    d['global_base'] = _global_baseline(d)
    d['outcome_medians'] = _outcome_medians(d)
    d['outlier_rank'] = _outlier_rank(d)
    _SNAP = d
    return d


def _outlier_rank(d):
    '''全市场"最不典型"排序：每支视频在自身分区内的最大 |z|（一次算好，别每次请求重算）'''
    rows = []
    for i, v in enumerate(d['videos']):
        base = d['baselines'].get(v.get('tname')) or d['global_base']
        best = 0.0
        for dim in DICT:
            if not DICT[dim].get('display'):
                continue
            b = (base.get('feats') or {}).get(dim)
            val = (v.get('f') or {}).get(dim)
            if not b or val is None or not b.get('mad'):
                continue
            best = max(best, abs((val - b['med']) / b['mad']))
        rows.append((best, i))
    rows.sort(reverse=True)
    return rows


def _outcome_medians(d):
    '''"后来怎样"的对照尺：该结果变量在全体有数据视频里的中位'''
    out = {}
    for k in ('span_days', 'half_decay_days', 'hk_halflife', 'halflife_d'):
        vals = sorted(v['outcome'][k] for v in d['videos'] if (v.get('outcome') or {}).get(k))
        if vals:
            out[k] = _pctl(vals, 0.5)
    return out


def _global_baseline(d):
    '''全样本基准（分区样本 <20 时的兜底参照系）——不能用别的分区冒充同类'''
    feats = {}
    for c in d['meta']['dims']:
        vals = sorted(v['f'][c] for v in d['videos'] if v['f'].get(c) is not None)
        if not vals:
            continue
        med = _pctl(vals, 0.5)
        mad = _pctl(sorted(abs(v - med) for v in vals), 0.5) * 1.4826
        feats[c] = {'med': med, 'mad': (mad if mad > 1e-9 else 1.0), 'n': len(vals)}
    per = [[] for _ in range(101)]
    for cur in d['curves'].values():
        for b in range(101):
            per[b].append(cur[b])
    env = {}
    for name, q in (('p25', 0.25), ('med', 0.5), ('p75', 0.75)):
        env[name] = [(_pctl(sorted(per[b]), q) or 0.0) for b in range(101)]
    return {'n_videos': len(d['videos']), 'low_support': False, 'is_global': True,
            'feats': feats, 'curve_p25': env['p25'], 'curve_med': env['med'],
            'curve_p75': env['p75'], 'n_curve': len(d['curves'])}


# ── 视图 1：对标对比图 ───────────────────────────────────────

def _band_dim(a, b):
    if b <= 8:
        return 'opening_5pct'
    if a >= 92:
        return 'ending_5pct'
    if a >= 5 and b <= 95:
        return 'mid_ratio'
    return None


def _bands(you, p25, p75, med):
    """冲出/掉出包络的连续区间（在展示用的平滑曲线上算，与用户所见完全一致）"""
    n = len(you)
    tag = []
    for i in range(n):
        if you[i] > p75[i] * 1.06:
            tag.append('above')
        elif you[i] < p25[i] * 0.94:
            tag.append('below')
        else:
            tag.append('')
    out = []
    i = 0
    while i < n:
        if not tag[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (tag[j + 1] == tag[i] or
                             (not tag[j + 1] and j + 2 < n and tag[j + 2] == tag[i])):
            j += 1
        if j - i + 1 >= BAND_MIN:
            sy = you[i:j + 1]
            sm = med[i:j + 1]
            m = sum(sm) / len(sm)
            r = (sum(sy) / len(sy) / m) if m else 1.0
            kind = tag[i]
            out.append({
                'from': i, 'to': j, 'kind': kind, 'ratio': round(r, 3),
                '_dim': _band_dim(i, j),
                'text': ('第 %d–%d%% 你冲出同类区间（是同类中位的 %d%%）' % (i, j, round(r * 100))
                         if kind == 'above' else
                         '第 %d–%d%% 你掉出同类区间（只有同类中位的 %d%%）' % (i, j, round(r * 100))),
            })
        i = j + 1
    out.sort(key=lambda b: (b['kind'] != 'below', b['from']))
    return out


def _mmss(sec):
    """时长格式化：<1 小时 mm:ss；≥1 小时 h:mm:ss。

    2026-10-22：长视频（如 25205 秒）原来会显示 420:05，用户读不出这是第几分钟；
    统一成 7:00:05。前端 diagnosis.html 的 mmss 同口径。
    """
    if sec is None:
        return None
    s = int(round(sec))
    h, m, ss = s // 3600, (s % 3600) // 60, s % 60
    return ('%d:%02d:%02d' % (h, m, ss)) if h else ('%02d:%02d' % (m, ss))


def _curve_block(rec, base, key, snap=None):
    snap = snap if snap is not None else _load()
    raw = snap['curves'].get(key)
    if not raw or not base:
        return None
    you, med = _smooth(raw), _smooth(base['curve_med'])
    p25, p75 = _smooth(base['curve_p25']), _smooth(base['curve_p75'])
    bands = _bands(you, p25, p75, med)
    peaks = []
    for e in snap['events'].get(key, []):
        c = None
        if e['b0'] is not None and e['b1'] is not None:
            c = int(round((e['b0'] + e['b1']) / 2.0))
        elif e['t0'] is not None and rec.get('duration'):
            c = int(min(99, round(e['t0'] / rec['duration'] * 100)))
        if c is None:
            continue
        peaks.append({'pct': c, 't0': e['t0'], 't1': e['t1'], 'event_id': e['id'],
                      'score': e['score'], 'n_methods': e['n_methods'],
                      'label': _mmss(e['t0']), 'danmaku': e['tops']})
    peaks.sort(key=lambda p: -(p['score'] or 0))
    peaks = peaks[:5]
    return {
        'you': [round(x, 6) for x in you],
        'median': [round(x, 6) for x in med],
        'p25': [round(x, 6) for x in p25],
        'p75': [round(x, 6) for x in p75],
        'you_raw': [round(x, 6) for x in raw],
        'duration': rec.get('duration'),
        'n_peers': base.get('n_curve'),
        'bands': bands,
        'peaks': peaks,
        'band_text': bands[0]['text'] if bands else None,
        'summary': _curve_summary(bands, peaks),
    }


def _curve_summary(bands, peaks):
    if not bands:
        return '你的弹幕曲线基本落在同类区间内——节奏形状和同类没有明显差别。'
    s = '；'.join(b['text'] for b in bands[:2])
    if peaks:
        s += '。最集中的反应出现在第 %d%% 附近' % peaks[0]['pct']
    return s + '。'


# ── 视图 2：偏离诊断 ────────────────────────────────────────

def _z(rec, base, dim):
    b = (base.get('feats') or {}).get(dim) if base else None
    v = (rec.get('f') or {}).get(dim)
    if not b or v is None or not b.get('mad'):
        return None
    return (v - b['med']) / b['mad']


def _compare_text(dim, v, med, pct, z):
    d = DICT[dim]
    if dim == 'main_peak_pct' and med is not None and med <= 10:
        # 同类的尖峰几乎都在最开头 → 真正的事实是"你没有开场尖峰"，不是"铺垫太长"
        if z > 0:
            return '同类的弹幕尖峰几乎都在最开头（中位第 %d%%），你的到第 %d%% 才出现' % (round(med), round(v))
        return '你的弹幕尖峰在第 %d%%（同类中位第 %d%%）' % (round(v), round(med))
    tpl = d['compare_high'] if z > 0 else d['compare_low']
    tail = max(1, round((100 - pct) if z > 0 else pct)) if pct is not None else None
    if '{v}' in tpl or '{m}' in tpl:
        return tpl.format(v=(round(v) if d['metric_kind'] == 'pct' else int(round(v))),
                          m=(round(med) if med is not None else '—'))
    if '{tail}' in tpl:
        return tpl.format(tail=tail if tail is not None else 50)
    r = (v / med * 100.0) if med else None
    if r is not None and '{r}' in tpl:
        return tpl.format(r=round(r))
    return '%s（%s）' % (d['high_text'] if z > 0 else d['low_text'], human_pos(pct))


def _direction(dim, z):
    d = DICT[dim]
    if d['direction'] != 'evidence' or d['metric_kind'] == 'index':
        return 'neutral', d['high_text'] if z > 0 else d['low_text']
    return ('good', d['high_text']) if z > 0 else ('bad', d['low_text'])


def _advice(dim, z, bench):
    '''33 号军规 3：偏离必须带方向 + 建议。形状维无因果证据 → 提醒这是参照不是承诺'''
    d = DICT[dim]
    a = d['advice_high'] if z > 0 else d['advice_low']
    if d['direction'] == 'evidence':
        return a
    if bench:
        return '%s（同类里《%s》就是这么做这一维的）' % (a, bench['title'])
    return a


def _pick_deviations(rec, base, pct_map):
    '''稳健闸门：进"偏离"候选必须 (在分区前/后 10%) 或 (|z|>=2)。
    否则小 MAD 维度（如爆点位置）会把平凡视频吹成 z=11——那是分布退化，不是偏离。'''
    allc, strong = [], []
    for dim, d in DICT.items():
        if not d.get('display'):
            continue
        z = _z(rec, base, dim)
        if z is None:
            continue
        t = (abs(z), z, dim)
        allc.append(t)
        p = pct_map.get(dim)
        if (p is not None and (p <= 10 or p >= 90)) or abs(z) >= 2.0:
            strong.append(t)
    allc.sort(reverse=True)
    strong.sort(reverse=True)
    pool = strong if len(strong) >= 3 else allc
    for th in (1.0, 0.75, 0.5):
        picked = [c for c in pool if c[0] >= th][:4]
        if len(picked) >= 3:
            return picked
    return pool[:4]


def _deviations(rec, base, pct_map, benchmarks, snap=None):
    snap = snap if snap is not None else _load()
    n_peers = int((base or {}).get('n') or 0) or 25
    out = []
    for a, z, dim in _pick_deviations(rec, base, pct_map):
        d = DICT[dim]
        v = rec['f'].get(dim)
        med = base['feats'][dim]['med']
        pct = pct_map.get(dim)
        direction, human = _direction(dim, z)
        drill = _drill_seg(snap, rec, base, dim, z)
        bench = None
        for b in benchmarks:
            if dim in (b.get('_learn_dim'), b.get('_diff_dim')):
                bench = b
                break
        out.append({
            '_dim': dim, 'label': d['label'], '_tech': d['tech'],
            'z': round(z, 2), 'abs_z': round(a, 2), 'level': level_of(a),
            'direction': direction, 'direction_label': DIR[direction]['label'],
            'direction_mark': DIR[direction]['mark'], 'color': DIR[direction]['color'],
            'pct': pct, 'position_text': human_pos(pct), 'pole': pole_word(dim, z),
            'human': human,
            'compare_text': _compare_text(dim, v, med, pct, z),
            'advice': _advice(dim, z, bench),
            'ref': ({'bvid': bench['bvid'], 'title': bench['title']} if bench else None),
            'seg': list(d['seg']), 'seg_name': d['seg_name'],
            'drill_seg': drill[0],
            'drill_seg_name': ('第 %d–%d%%' % (drill[0][0], drill[0][1])),
            'drill_why': drill[1] or ('词典给这一维定的观察窗口是「%s」' % d['seg_name']),
            'peer_count': (round(segment_peer_count(snap, rec.get('tname'),
                                                    drill[0][0], drill[0][1],
                                                    exclude=rec['bvid']) or 0) or None),
            'value': round(v, 6) if v is not None else None,
            'median': round(med, 6) if med is not None else None,
        })
    # 精确值 + 出处徽章：ML 可信度靠「精确 / 出处 / 诚实」三签名，
    # 不靠「展示 ML 动作」的器件（PCA 地图 / 人工 vs 算法 / 参数滑块上主视图 —— 均已否决）。
    for x in out:
        x['you_text'] = _precise(x['_dim'], x.get('value'))
        x['median_text'] = _precise(x['_dim'], x.get('median'))
        x['unit_label'] = UNIT_LABEL.get(x['_dim'], '')
        x['badges'] = _badges(x, n_peers)
    return out


# 精确到 2 位小数的显示规则：单位由维度定，个数是整数就不硬凑两位小数
# （「1.00 个爆点」是错的写法）
UNIT_LABEL = {
    'dm_per_min': '条/分钟', 'wdm_per_min': '条/分钟（加权）', 'avg_text_len': '字',
    'opening_5pct': '前 5% 弹幕占比', 'ending_5pct': '后 5% 弹幕占比',
    'main_peak_pct': '爆点出现在整片第几 %', 'mid_ratio': '中段弹幕占比',
    'rhythm_entropy': '起伏熵（0–1）', 'burst_cv': '突变变异系数',
    'early_lag_ratio': '1 小时内到达的弹幕占比', 'peak_cnt': '爆点个数',
}
_PCT_DIMS = ('opening_5pct', 'ending_5pct', 'early_lag_ratio')
_RATIO_DIMS = ('mid_ratio', 'rhythm_entropy', 'burst_cv')
_RHO = {'dm_per_min': '+0.398', 'wdm_per_min': '+0.402', 'avg_text_len': '+0.303'}


def _precise(dim, v):
    # 按维度给 2 位小数；个数是整数就不硬凑小数
    if v is None:
        return '—'
    if dim == 'peak_cnt':
        return '%d 个' % int(round(v))
    if dim == 'main_peak_pct':
        return '%.2f%%' % v
    if dim in _PCT_DIMS:
        return '%.2f%%' % (v * 100)
    if dim in _RATIO_DIMS:
        return '%.2f' % v
    if dim in ('dm_per_min', 'wdm_per_min'):
        return '%.2f 条/分' % v
    if dim == 'avg_text_len':
        return '%.2f 字' % v
    return '%.2f' % v


def _badges(x, n_peers):
    # 关键数字挂小徽章：样本量 / 跨批一致性 / 方向出处。
    # 每个徽章都要能回答「这个数凭什么可信」，不写空话。
    out = [{'text': '同类 %d 支' % n_peers, 'time': '样本量：同分区可比视频数'}]
    if x['_dim'] in _RATIO_DIMS + _PCT_DIMS + ('main_peak_pct',):
        out.append({'text': '跨批一致 99.2%',
                    'time': '32 号 P1：形状签名跨批 within 中位 0.992 vs between 中位 0.069'})
    if x['_dim'] in _RHO:
        out.append({'text': '控龄 ρ %s' % _RHO[x['_dim']],
                    'time': '32 号 P3 控龄偏相关（BH-FDR 0.05，n=679）'})
    return out


# ── 视图 3：相似对标 ────────────────────────────────────────

OUT_LABEL = {'span_days': '上榜跨度', 'half_decay_days': '热度半衰期',
             'hk_halflife': '热度半衰期', 'halflife_d': '热度半衰期'}
OUT_ORDER = ('span_days', 'half_decay_days', 'hk_halflife', 'halflife_d')


def _outcome_text(o):
    '''返回 (人话读数, 数值, 变量名)——变量名用于和全体中位比"走得好不好"'''
    if not o:
        return None, None, None
    for k in OUT_ORDER:
        if o.get(k):
            return _days_text(OUT_LABEL[k], o[k]), round(o[k], 2), k
    return None, None, None


def _outcome_strength(d, o):
    '''2 = 比一半同类走得远（可以拿来当"该学谁"的依据）；1 = 有数据但不算突出；0 = 没数据'''
    text, val, key = _outcome_text(o or {})
    if key is None:
        return 0, text, val, key
    med = (d.get('outcome_medians') or {}).get(key)
    return (2 if (med is not None and val > med) else 1), text, val, key


def _days_text(prefix, days):
    d = float(days)
    if d < 1:
        return '%s %d 小时' % (prefix, max(1, round(d * 24)))
    return '%s %d 天' % (prefix, round(d))


def _outcome_rank(o):
    if not o:
        return None
    # 注：n_days_obs 不计入——30 号 N2 已证其是采集属性（标签污染），不是视频的结果
    for k in ('span_days', 'half_decay_days', 'hk_halflife', 'halflife_d'):
        if o.get(k):
            return float(o[k])
    return None


def _shape_z(rec, base, global_shape):
    out = []
    for c in _load()['meta']['shape_dims']:
        v = rec['shape'].get(c)
        if v is None:
            out.append(0.0)
            continue
        b = (base or {}).get('feats', {}).get(c) if base else None
        med, mad = (b['med'], b['mad']) if b and b.get('mad') else global_shape.get(c, (v, 1.0))
        out.append((v - med) / (mad if mad else 1.0))
    return out


def _gap_text(dim, v, med):
    '''相对分区中位的方位描述（33 号军规 2：不说小数说方位）'''
    if med in (None, 0) or v is None:
        return '接近同类中位'
    r = v / med
    if abs(r - 1.0) <= 0.12:
        return '和同类中位差不多'
    return '比同类中位%s约 %d%%' % ('高' if r > 1 else '低', round(abs(r - 1) * 100))


def _benchmarks(rec, base, pool_idx, snap=None):
    snap = snap if snap is not None else _load()
    sd = snap['meta']['shape_dims']
    you = _shape_z(rec, base, snap['global_shape'])
    my_rank = _outcome_rank(rec.get('outcome'))
    scored = []
    for i in pool_idx:
        o = snap['videos'][i]
        if o['bvid'] == rec['bvid']:
            continue
        scored.append((_cos(you, _shape_z(o, base, snap['global_shape'])), i))
    scored.sort(reverse=True)
    top = scored[:NEIGHBOR_SCAN]
    strong = [t for t in top if _outcome_strength(snap, snap['videos'][t[1]].get('outcome'))[0] == 2]
    have = [t for t in top if _outcome_strength(snap, snap['videos'][t[1]].get('outcome'))[0] == 1]
    picks, seen = [], {rec['bvid']}
    for t in strong + have + top:
        bv = snap['videos'][t[1]]['bvid']
        if bv in seen:
            continue
        seen.add(bv)
        picks.append(t)
        if len(picks) >= NEIGHBOR_SHOW:
            break
    picks.sort(key=lambda t: -t[0])

    out = []
    for sim, i in picks:
        o = snap['videos'][i]
        ov = _shape_z(o, base, snap['global_shape'])
        o_strength, otext, oval, o_key = _outcome_strength(snap, o.get('outcome') or {})
        o_rank = _outcome_rank(o.get('outcome') or {})
        deltas = sorted((abs(you[k] - ov[k]), sd[k]) for k in range(len(sd)))
        same_dims = [d for _, d in deltas[:2]]
        ranked = sorted(((abs(you[k]) - abs(ov[k]), c) for k, c in enumerate(sd)), reverse=True)
        learn = None
        evidence = o_strength == 2
        for gap, c in (ranked if evidence else []):
            k = sd.index(c)
            if abs(you[k]) < 0.8 or gap <= 0.3:
                break
            mv = (rec['f'] or {}).get(c)
            med = ((base or {}).get('feats', {}).get(c) or {}).get('med')
            if mv is not None and med and abs(mv / med - 1.0) > 0.15:
                learn = c
                break
        diff_dim = sd[max(range(len(sd)), key=lambda k: abs(you[k] - ov[k]))]
        key = '%s|%s' % (o['bvid'], o['batch_date'])
        cur = snap['curves'].get(key)
        sim_pct = max(0, round(sim * 100))

        like = '它和你在【%s】上最像' % '、'.join(label_of(d) for d in same_dims)
        if learn:
            like += '；不同的是它在【%s】上比你更靠近同类中位' % label_of(learn)
        else:
            like += '；差别最大的是【%s】' % label_of(diff_dim)
        like += ('。它后来%s' % otext) if otext else '。它暂时没有后续追踪数据'

        learn_txt = None
        if learn:
            lv, mv = (o['f'] or {}).get(learn), (rec['f'] or {}).get(learn)
            med = ((base or {}).get('feats', {}).get(learn) or {}).get('med')
            if lv is not None and mv is not None:
                lead = ''
                if o_rank is not None and my_rank is not None and o_rank > my_rank:
                    lead = '它后来%s，比你走得远；' % (otext or '')
                elif otext:
                    lead = '它后来%s，在同类里算走得远的；' % otext
                learn_txt = '%s参考它在【%s】上的做法——它%s，你%s' % (
                    lead, label_of(learn), _gap_text(learn, lv, med), _gap_text(learn, mv, med))
        if not learn_txt:
            learn_txt = ('它和你的节奏最接近，可作为同类里的对照样本；'
                         '但它没有"比一半同类走得远"的数据，不能当作"该学谁"的依据')
        out.append({
            'bvid': o['bvid'], 'title': o.get('title') or o['bvid'],
            'tname': o.get('tname'), 'batch_date': o['batch_date'],
            'similarity_pct': sim_pct, 'similarity_label': sim_label(sim_pct),
            '_same_dims': same_dims, '_diff_dim': diff_dim, 'diff_label': label_of(diff_dim),
            'same_labels': [label_of(x) for x in same_dims],
            '_learn_dim': learn, 'learn_label': label_of(learn) if learn else None,
            'learn_seg': _seg_of(learn) if learn else None,
            'learn_seg_name': (DICT.get(learn) or {}).get('seg_name') if learn else None,
            'like_text': like, 'learn_text': learn_txt,
            'outcome_text': otext or '暂无后续追踪数据',
            'outcome_value': oval, '_outcome_key': o_key, 'outcome_rank': o_rank,
            'did_better': evidence,
            'curve': [round(x, 6) for x in _smooth(cur)] if cur else None,
            'has_outcome': otext is not None,
        })
    return out


def segment_peer_count(snap, tname, b0, b1, exclude=None, want='median'):
    """同类在这一段里大概有多少条弹幕（供"证据钻取"和真实条数对照）。

    口径：视频自己的曲线是"每 1% 桶占全片比例"，所以 sum(该段各桶) × 该视频总条数
    = 该视频在段内的条数估计。对同分区每支视频都这么算，再取中位。
    这是估计值（曲线口径），页面上以"同类中位 ≈ N 条"呈现，不当作精确计数。
    """
    tidx = [i for i, v in enumerate(snap['videos']) if v.get('tname') == tname]
    vals = []
    for i in tidx:
        v = snap['videos'][i]
        if exclude and v['bvid'] == exclude:
            continue
        cur = snap['curves'].get('%s|%s' % (v['bvid'], v['batch_date']))
        n = v.get('n_danmaku')
        if not cur or not n:
            continue
        share = sum(cur[max(0, b0):min(101, b1 + 1)])
        vals.append(share * n)
    if not vals:
        return None
    vals.sort()
    return _pctl(vals, 0.5 if want == 'median' else 0.75)


def _drill_seg(snap, rec, base, dim, z, width=10):
    """钻取用哪一段：词典给了窄分段（开场/收尾…）就用它；
    否则（整片类维度）**按数据找**——取"你与同类中位差得最多"的一段，
    而不是把整片丢给用户看（整片钻取等于没钻）。"""
    d = DICT.get(dim) or {}
    seg = list(d.get('seg') or [0, 100])
    if seg[1] - seg[0] < 60:
        return seg, None
    # 位置型维度（爆点位置）：看"同类尖峰在哪一段"才有对比意义
    # （说"你的尖峰在第 23%"时，证据是"同类在第 5% 就爆了而你这里没有"）
    if dim == 'main_peak_pct' and base and base.get('curve_med'):
        med = _smooth(base['curve_med'])
        pk = max(range(len(med)), key=lambda i: med[i])
        return [max(0, pk - 4), min(100, pk + 5)], '同类的弹幕尖峰就在这一段，你这里没有'
    cur = snap['curves'].get('%s|%s' % (rec['bvid'], rec['batch_date']))
    if not cur or not base or not base.get('curve_med'):
        return seg, None
    you, med = _smooth(cur), _smooth(base['curve_med'])
    best, bi = -1e9, 0
    for i in range(0, max(1, 101 - width)):
        ya = sum(you[i:i + width]) / width
        ma = sum(med[i:i + width]) / width
        score = (ma - ya) if z < 0 else (ya - ma)
        if score > best:
            best, bi = score, i
    if best <= 0:
        return seg, None
    return [bi, bi + width - 1], ('这一段你和同类差得最多'
                                  if z < 0 else '这一段你比同类高得最多')


def _seg_of(dim):
    d = DICT.get(dim) or {}
    seg = d.get('seg')
    return list(seg) if seg else None


def _wan(v):
    if v is None:
        return None
    v = float(v)
    return ('%.1f 万' % (v / 10000.0)) if v >= 10000 else ('%d' % round(v))


def _position(rec, base, pct_map, pool_idx, partition_ok, snap=None):
    snap = snap if snap is not None else _load()
    main = 'dm_per_min'
    vals = sorted((snap['videos'][i]['f'].get(main) for i in pool_idx
                   if snap['videos'][i]['f'].get(main) is not None), reverse=True)
    mine = rec['f'].get(main)
    rank = (1 + sum(1 for x in vals if x > mine)) if (mine is not None and vals) else None
    pct = pct_map.get(main)
    dims = []
    for dim, d in DICT.items():
        if not d.get('display'):
            continue
        p = pct_map.get(dim)
        dims.append({'_dim': dim, 'label': d['label'], 'pct': p,
                     'position_text': human_pos(p), 'seg': list(d['seg']),
                     'human': (d['high_text'] if (p is not None and p >= 50) else d['low_text']),
                     'ahead': (round(100 - p) if p is not None else None)})
    dims.sort(key=lambda x: (x['pct'] is None, -(x['pct'] or 0)))
    scope = (rec.get('tname') or '全部样本') if partition_ok else '全部样本'
    return {
        'scope': scope, 'n_peers': len(pool_idx), 'rank': rank,
        'percentile': pct, 'ahead_pct': (round(100 - pct) if pct is not None else None),
        '_main_dim': main, 'main_label': DICT[main]['label'],
        'headline': ('你在【%s】的%s%s' % (scope, DICT[main]['label'], pos_of(pct))
                     if pct is not None else '该视频暂无同类定位数据'),
        'detail': ('%d 支同类视频中排第 %s' % (len(pool_idx), rank) if rank else None),
        'dims': dims,
    }


def _problem_short(d):
    dim, low = d['_dim'], d['z'] < 0
    if dim == 'main_peak_pct' and (d.get('median') or 0) <= 10 and not low:
        return '没有开场尖峰'
    table = {
        'opening_5pct': ('开场没抓住人', None),
        'ending_5pct': ('结尾掉得快', None),
        'mid_ratio': ('中段掉链子', None),
        'dm_per_min': ('弹幕比同类冷清', None),
        'wdm_per_min': ('弹幕比同类冷清', None),
        'avg_text_len': ('观众只刷短弹幕', None),
        'early_lag_ratio': ('观众来得比同类慢', None),
        'rhythm_entropy': (None, '全片没有明确爆点'),
        'burst_cv': ('节奏太平，没有起伏', '起伏比同类大得多'),
        'peak_cnt': ('只有一个爆点', '爆点比同类多，分段碎'),
        'main_peak_pct': ('爆点偏前，后面可能乏力', '爆点偏后，前面铺垫太长'),
    }
    a, b = table.get(dim, (None, None))
    if dim == 'main_peak_pct':
        return a if low else b
    return a if low else (b or ('%s%s' % (d['label'], d['pole'])))


def _actions(position, concern, deviations, benchmarks):
    out = []
    if concern:
        # 用 drill_seg 而不是词典里的原始分段：整片类维度若照抄 seg=[0,100]，
        # 前端会把整张图铺成橙色高亮，等于把"该照哪一段"这条信息抹掉
        out.append({'n': 1, 'text': concern['advice'],
                    'basis': '%s%s（%s）' % (concern['label'], concern['pole'],
                                          concern['compare_text']),
                    'ref': concern.get('ref'),
                    'seg': concern.get('drill_seg') or concern.get('seg'),
                    'seg_why': concern.get('drill_why')})
    else:
        out.append({'n': 1,
                    'text': '你的节奏在同类里没有明显短板——维持现在的节奏结构就行。',
                    'basis': '各展示维都没有落在"该往哪调"的方向上',
                    'ref': None, 'seg': None})
    better = next((b for b in benchmarks if b.get('_learn_dim')), None)
    if better:
        out.append({'n': len(out) + 1,
                    'text': '参考《%s》：%s。%s' % (better['title'], better['similarity_label'],
                                                better['learn_text']),
                    'basis': '它后来%s' % better['outcome_text'],
                    'ref': {'bvid': better['bvid'], 'title': better['title']},
                    'seg': better.get('learn_seg')})
    elif benchmarks and benchmarks[0]['has_outcome']:
        # 33 号 §3 视图 4：行动建议一律对标式——没有"可学一维"、但对方有后续数据时给同类参照
        b0 = benchmarks[0]
        out.append({'n': len(out) + 1,
                    'text': '把《%s》当同类参照：%s，%s' % (
                        b0['title'], b0['similarity_label'],
                        ('它后来%s' % b0['outcome_text']) if b0['has_outcome'] else '它暂无后续追踪数据'),
                    'basis': b0['like_text'],
                    'ref': {'bvid': b0['bvid'], 'title': b0['title']}, 'seg': None})
    good = next((d for d in deviations if d['direction'] == 'good'), None)
    if good:
        out.append({'n': len(out) + 1, 'text': '%s是你的优势项，保持。' % good['label'],
                    'basis': good['compare_text'], 'ref': None, 'seg': good.get('seg')})
    elif deviations:
        out.append({'n': len(out) + 1,
                    'text': '其余节奏维都在同类常见范围内，不用动。',
                    'basis': '除上面这条外，其它展示维都没冲出/掉出同类区间',
                    'ref': None, 'seg': None})
    return out[:3]


def _is_complaint(d):
    """这一维是否落在"值得改/值得注意"的那一侧（方向来自 32 号 P3 / 词典 CONCERN）"""
    pole = CONCERN.get(d['_dim'])
    if pole is None:
        return False
    if pole == 'both':
        return d['abs_z'] >= 1.0
    return (pole == 'low' and d['z'] < 0) or (pole == 'high' and d['z'] > 0)


def _verdict(rec, position, deviations, benchmarks, partition_ok):
    complaints = sorted((d for d in deviations if _is_complaint(d)), key=lambda d: -d['abs_z'])
    concern = complaints[0] if (complaints and complaints[0]['abs_z'] >= 1.0) else None
    pos_phrase = ('你在【%s】的%s%s' % (position['scope'], position['main_label'],
                                    pos_of(position['percentile']))
                  if position['percentile'] is not None else '这支视频')
    detail = None
    if concern:
        headline = '%s，但【%s】%s：%s' % (pos_phrase, concern['label'], concern['pole'],
                                       concern['compare_text'])
        problem = _problem_short(concern)
        tone = 'ok' if concern['direction'] == 'good' else 'warn'
    else:
        headline = '%s，节奏整体健康——没有需要调整的偏离项。' % pos_phrase
        if deviations:
            top = max(deviations, key=lambda d: d['abs_z'])
            headline += '你和同类最不一样的地方是【%s】%s，但这一维没有"该往哪调"的证据。' % (
                top['label'], top['compare_text'])
        problem, tone = '节奏健康，没有明显短板', 'ok'
    if benchmarks:
        b0 = benchmarks[0]
        detail = ('同类里《%s》%s（%d%%）%s' % (
            b0['title'], b0['similarity_label'], b0['similarity_pct'],
            ('，它后来%s' % b0['outcome_text']) if b0['has_outcome'] else '，它暂无后续追踪数据'))
    return {
        'headline': headline, 'detail': detail, 'problem_5s': problem, 'tone': tone,
        '_concern_dim': (concern or {}).get('_dim'),
        'concern_label': (concern or {}).get('label'),
        'concern_seg': _seg_of((concern or {}).get('_dim')) if concern else None,
        'actions': _actions(position, concern, deviations, benchmarks),
        'caveat': ACTION_CAVEAT, 'curve_caveat': CURVE_CAVEAT,
        'partition_used_full_sample': not partition_ok,
    }


def diagnose(bvid, batch_date=None, snap=None, include_sensitivity=True):
    # snap 可选：在线诊断（serve/live_diag.py）把算好的库外视频注入一份影子快照后复用本函数，
    # 这样参照系 / 稳健 z / 百分位 / kNN / 结论 / 行动全部走库内同一套口径，不另写一份。
    snap = snap if snap is not None else _load()
    key = '%s|%s' % (bvid, batch_date) if batch_date else None
    if key and key in snap['by_key']:
        idx = snap['by_key'][key]
    elif bvid in snap['by_bvid']:
        idx = snap['by_bvid'][bvid]
    else:
        return None
    rec = snap['videos'][idx]
    tname = rec.get('tname')
    base_p = snap['baselines'].get(tname) if tname else None
    partition_ok = (bool(base_p) and not base_p.get('low_support')
                    and base_p.get('n_videos', 0) >= MIN_PARTITION)
    base = base_p if partition_ok else snap['global_base']
    same_t = [i for i, v in enumerate(snap['videos']) if v.get('tname') == tname] if tname else []
    pool_idx = same_t if len(same_t) >= 8 else list(range(len(snap['videos'])))
    pct_map = {k: v for k, v in (rec.get('pct') or {}).items() if v is not None}
    benchmarks = _benchmarks(rec, base if partition_ok else None, pool_idx, snap=snap)
    knn_scope = (tname if len(same_t) >= 8 else '全体样本')
    deviations = _deviations(rec, base, pct_map, benchmarks, snap=snap)
    curve = _curve_block(rec, base, '%s|%s' % (bvid, rec['batch_date']), snap=snap)
    position = _position(rec, base, pct_map, pool_idx, partition_ok, snap=snap)
    verdict = _verdict(rec, position, deviations, benchmarks, partition_ok)
    return {
        'video': {
            'bvid': bvid, 'title': rec.get('title'), 'up_name': rec.get('up_name'),
            'tname': tname, 'batch_date': rec['batch_date'],
            'duration': rec.get('duration'), 'duration_text': _mmss(rec.get('duration')),
            'n_danmaku': rec.get('n_danmaku'), 'play': rec.get('play'),
            'play_text': _wan(rec.get('play')),
            'url': 'https://www.bilibili.com/video/%s' % bvid,
        },
        'verdict': verdict, 'position': position, 'deviations': deviations,
        'benchmarks': benchmarks, 'curve': curve, 'knn_scope': knn_scope,
        'scope_note': (None if partition_ok else
                       '该分区样本不足 %d 支，本次以全体样本作参照系' % MIN_PARTITION),
        'tech': {
            'dict_version': DICT_VERSION,
            'engine_version': snap['meta'].get('engine_version'),
            'z_formula': snap['meta'].get('z_formula'),
            'curve_norm': snap['meta'].get('curve_norm'),
            'partition': tname, 'partition_n': (base or {}).get('n_videos'),
            'partition_low_support': (base or {}).get('low_support'),
            'partition_used_full_sample': not partition_ok,
            'shape_dims': snap['meta']['shape_dims'],
            'raw_features': {k: rec['f'].get(k) for k in snap['meta']['dims']},
            'raw_shape': rec.get('shape'),
            'percentiles': pct_map,
            'outcome_raw': rec.get('outcome'),
            'similarity_metric': 'cosine on partition-z shape vector (32 号 P1 口径)',
            # include_sensitivity=False is used by batch_check only.
            # Otherwise sensitivity -> batch_check -> diagnose recurses forever (85 multi-batch bvids, 2026-10-19).
            'sensitivity': sensitivity_payload(snap, rec) if include_sensitivity else None,
            'dict_map': [{'tech': v.get('tech'), 'label': v['label'],
                          'direction': v['direction'], 'display': bool(v.get('display'))}
                         for v in DICT.values()],
            'snapshot_meta': snap['meta'],
        },
    }


def snapshot_meta():
    return _load()['meta']


def index():
    '''供前端选片用的轻量索引（803 支，按分区+热度排序）'''
    snap = _load()
    rows = []
    for b, i in snap['by_bvid'].items():
        v = snap['videos'][i]
        rows.append({'bvid': b, 'title': v.get('title') or b, 'tname': v.get('tname'),
                     'dm_per_min': (v.get('f') or {}).get('dm_per_min'),
                     'n_danmaku': v.get('n_danmaku'), 'batch_date': v['batch_date'],
                     'has_curve': v.get('has_curve'),
                     'has_outcome': _outcome_rank(v.get('outcome') or {}) is not None})
    size = {}
    for r in rows:
        size[r['tname']] = size.get(r['tname'], 0) + 1
    rows.sort(key=lambda r: (-size.get(r['tname'], 0), r['tname'] or 'zzz', -(r['dm_per_min'] or 0)))
    return rows


def latest_batch(bvid):
    snap = _load()
    i = snap['by_bvid'].get(bvid)
    return snap['videos'][i]['batch_date'] if i is not None else None


# ── v3 其余视图：市场总览 / 分区画像（同一套快照与词典，不新造口径） ──

def _partition_trait(snap, tname):
    """这个分区的观众习惯（人话）：取分区中位偏离全市场中位最狠的 2 维"""
    b = snap['baselines'].get(tname)
    g = snap['global_base']
    if not b:
        return None, []
    scored = []
    for dim in DICT:
        if not DICT[dim].get('display') or dim == 'main_peak_pct':
            continue
        bm = (b['feats'].get(dim) or {}).get('med')
        gm = (g['feats'].get(dim) or {}).get('med')
        gmad = (g['feats'].get(dim) or {}).get('mad')
        if bm is None or gm is None or not gmad:
            continue
        scored.append(((bm - gm) / gmad, dim))
    scored.sort(key=lambda x: -abs(x[0]))
    top = [s for s in scored[:2] if abs(s[0]) >= 0.35]
    return '、'.join(trait_of(dim, z) for z, dim in top), top


def _part_rank(snap):
    rows = []
    for t, b in snap['baselines'].items():
        if b.get('n_videos', 0) < 8:
            continue
        hm = (b['feats'].get('dm_per_min') or {}).get('med')
        if hm is not None:
            rows.append((hm, t))
    rows.sort(reverse=True)
    return {t: i + 1 for i, (_, t) in enumerate(rows)}


def _max_dim(rec, base):
    best = None
    for dim in DICT:
        if not DICT[dim].get('display'):
            continue
        z = _z(rec, base, dim)
        if z is not None and (best is None or abs(z) > abs(best[1])):
            best = (dim, z)
    return best[0] if best else None


def _pv(v, extra=None):
    return {'bvid': v['bvid'], 'title': v.get('title') or v['bvid'], 'tname': v.get('tname'),
            'dm_per_min_text': ('%.0f 条/分' % v['f']['dm_per_min'])
                               if (v.get('f') or {}).get('dm_per_min') else '—',
            'n_danmaku': v.get('n_danmaku'), 'note': extra}


def overview(snap):
    """市场总览：池子规模 + 各分区性格（人话）+ 全市场最不典型的视频"""
    rank = _part_rank(snap)
    parts = []
    for t, b in snap['baselines'].items():
        if b.get('n_videos', 0) < 8:
            continue
        trait, top = _partition_trait(snap, t)
        hm = (b['feats'].get('dm_per_min') or {}).get('med')
        parts.append({
            'tname': t, 'n_videos': b['n_videos'], 'low_support': bool(b.get('low_support')),
            'hot_med_text': ('%.0f 条/分' % hm) if hm else '—',
            'hot_rank': rank.get(t),
            'trait': trait or '节奏和全市场差不多',
            'trait_dims': [d for _, d in top],
            'n_curve': b.get('n_curve'),
            'curve': [round(x, 6) for x in _smooth(b['curve_med'])],
        })
    parts.sort(key=lambda p: (-p['n_videos'], p['tname']))
    outl, used = [], {}
    for _, i in snap['outlier_rank']:
        if len(outl) >= 12:
            break
        v = snap['videos'][i]
        base = snap['baselines'].get(v.get('tname')) or snap['global_base']
        dim = _max_dim(v, base)
        if not dim or used.get(dim, 0) >= 3:
            continue          # 同一维最多 3 支，否则整张榜被一个重尾维度刷屏
        used[dim] = used.get(dim, 0) + 1
        zz = _z(v, base, dim) or 0.0
        cur = snap['curves'].get('%s|%s' % (v['bvid'], v['batch_date']))
        outl.append({'bvid': v['bvid'], 'title': v.get('title') or v['bvid'], 'tname': v.get('tname'),
                     'label': DICT[dim]['label'], 'pole': pole_word(dim, zz),
                     'level': level_of(abs(zz)), 'pct': (v.get('pct') or {}).get(dim),
                     'curve': [round(x, 6) for x in _smooth(cur)] if cur else None})
    n_out = sum(1 for v in snap['videos'] if _outcome_rank(v.get('outcome') or {}) is not None)
    n_bv = len(snap['by_bvid'])
    n_batch = len({v['batch_date'] for v in snap['videos']})
    big = [p for p in parts if not p['low_support']]
    return {
        'headline': ('收录 %d 支视频、%d 个批次、%d 个分区；样本最足的是【%s】（%d 支）'
                     % (n_bv, n_batch, len(parts), parts[0]['tname'] if parts else '—',
                        parts[0]['n_videos'] if parts else 0)),
        'sub': ('每个视频都只跟"自己分区里的其它视频"比，不跟全站比——'
                '因为不同分区的观众习惯本来就不一样。所以同一个数字，在不同分区含义不同。'),
        'stats': [
            {'label': '收录视频', 'value': n_bv, 'note': '%d 个批次' % n_batch},
            {'label': '可比分区', 'value': len(parts), 'note': '其中 %d 个样本 ≥20 支' % len(big)},
            {'label': '检出高能点', 'value': snap['meta']['n_events'], 'note': '点爆点能看当时弹幕'},
            {'label': '有后续追踪', 'value': n_out,
             'note': '能讲"后来怎样"的占 %.0f%%' % (100.0 * n_out / max(1, n_bv))},
        ],
        'partitions': parts,
        'outliers': outl,
        'notes': ['分区榜按样本量排序；样本 <20 支的分区标为"样本少"，基准仅供参考。',
                  '"最不典型的视频"按分区内偏离幅度排序，点任意一支可进它的诊断卡。'],
    }


def partition_profile(snap, tname):
    """分区画像：这个分区的观众习惯是什么、谁是标杆、谁最特别"""
    b = snap['baselines'].get(tname)
    if not b:
        return None
    g = snap['global_base']
    trait, _ = _partition_trait(snap, tname)
    dims = []
    for dim, d in DICT.items():
        if not d.get('display'):
            continue
        bm = (b['feats'].get(dim) or {}).get('med')
        gm = (g['feats'].get(dim) or {}).get('med')
        if bm is None or gm is None:
            continue
        rel = (bm / gm - 1.0) if gm else 0.0
        # 中位为 0 时"低 100%"是退化读数（分区里多数视频该维就是 0）→ 改用"多数视频…"说
        if bm == 0 and gm:
            vs, rel = '多数视频' + trait_of(dim, -1), -1.0
        elif rel > 0.05:
            vs = '比全市场高 %d%%' % round(rel * 100)
        elif rel < -0.05:
            vs = '比全市场低 %d%%' % round(-rel * 100)
        else:
            vs = '和全市场差不多'
        dims.append({'label': d['label'], '_dim': dim, 'part_med': bm, 'global_med': gm,
                     'rel_pct': round(rel * 100), 'vs_text': vs,
                     'trait': trait_of(dim, 1 if rel >= 0 else -1)})
    dims.sort(key=lambda x: -abs(x['rel_pct']))
    idx = [i for i, v in enumerate(snap['videos']) if v.get('tname') == tname]
    hot = sorted(((_z(snap['videos'][i], b, 'dm_per_min') or 0.0), i) for i in idx)
    mid = len(idx) // 2
    dev = sorted((abs(_z(snap['videos'][i], b, _max_dim(snap['videos'][i], b)) or 0.0), i)
                 for i in idx if _max_dim(snap['videos'][i], b))
    special = []
    for _, i in dev[::-1][:3]:
        v = snap['videos'][i]
        dim = _max_dim(v, b)
        special.append(_pv(v, extra='%s%s' % (DICT[dim]['label'], pole_word(dim, _z(v, b, dim) or 0))))
    return {
        'tname': tname, 'n_videos': b['n_videos'], 'low_support': bool(b.get('low_support')),
        'n_curve': b.get('n_curve'),
        'headline': '【%s】区 %d 支视频的观众习惯：%s' % (tname, b['n_videos'], trait or '节奏和全市场差不多'),
        'sub': ('这个分区的中位曲线就是"同类标准"，该分区任何一支视频都能跟它比。样本 %d 支%s。'
                % (b['n_videos'], '（偏少，基准仅供参考）' if b.get('low_support') else '')),
        'curve': {'median': [round(x, 6) for x in _smooth(b['curve_med'])],
                  'p25': [round(x, 6) for x in _smooth(b['curve_p25'])],
                  'p75': [round(x, 6) for x in _smooth(b['curve_p75'])],
                  'global_median': [round(x, 6) for x in _smooth(g['curve_med'])],
                  'n_peers': b.get('n_curve')},
        'dims': dims,
        'hottest': [_pv(snap['videos'][i]) for _, i in hot[::-1][:3]],
        'typical': [_pv(snap['videos'][i]) for _, i in hot[mid:mid + 3]],
        'special': special,
        'dims_note': '分区中位 vs 全市场中位：差得越多，说明这个分区的观众习惯越有特色。',
    }



# ── 参数 / 尺度敏感性（只进技术层，主卡不出现；给答辩看"结果稳不稳"） ──

# 22 号 §1 的 θ 扫描实测表（原样搬进来，不重算）：
#   θ | 事件数 | 覆盖 video-batch | 覆盖占比 | 每 vb 事件数
THETA_SCAN = [
    (0.0, 13952, 836, 100.0, 16.69), (0.1, 3765, 836, 100.0, 4.50),
    (0.2, 2366, 836, 100.0, 2.83), (0.3, 1341, 784, 93.8, 1.71),
    (0.4, 722, 582, 69.6, 1.24), (0.5, 551, 473, 56.6, 1.16),
    (0.6, 385, 353, 42.2, 1.09), (0.7, 320, 305, 36.5, 1.05),
    (0.8, 198, 193, 23.1, 1.03), (0.9, 147, 146, 17.5, 1.01),
    (1.0, 113, 113, 13.5, 1.00),
]
# 分母是 836 不是 888：22 号 §1 的 θ 扫描跑在 l2_candidates 候选池上（θ=0 时覆盖 836 个
# video-batch）。用 888 会算出零事件 415 个，与文档的 363 个对不上。
THETA_CURRENT = 0.5
TOTAL_VB = 836


def z_sensitivity(snap):
    """偏离阈值 z 的敏感性：改阈值，全市场有多少支会被判"有偏离"。
    本页"偏离卡"进不进候选就用这个闸门，所以这是直接相关的参数。"""
    out = []
    for th in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
        n = sum(1 for z, _ in snap['outlier_rank'] if z >= th)
        out.append({'z': th, 'n': n, 'pct': round(100.0 * n / max(1, len(snap['outlier_rank'])), 1)})
    return out


def knn_sensitivity(snap, rec):
    """K（取几支最像的）敏感性：第 K 近的相似度掉到多少。
    说明"取 3 支"不是拍脑袋——超过某点之后就不像了。"""
    pool = [i for i, v in enumerate(snap['videos']) if v.get('tname') == rec.get('tname')] \
        if rec.get('tname') else []
    if len(pool) < 8:
        pool = list(range(len(snap['videos'])))
    base = snap['baselines'].get(rec.get('tname'))
    if not (base and not base.get('low_support')):
        base = None
    you = _shape_z(rec, base, snap['global_shape'])
    sims = []
    for i in pool:
        o = snap['videos'][i]
        if o['bvid'] == rec['bvid']:
            continue
        sims.append(_cos(you, _shape_z(o, base, snap['global_shape'])))
    sims.sort(reverse=True)
    out = []
    for k in (1, 2, 3, 5, 8, 12, 20):
        if k <= len(sims):
            out.append({'k': k, 'sim_pct': max(0, round(sims[k - 1] * 100))})
    return out


def batch_check(snap, rec):
    """批次敏感性：同一支视频换批次，结论会不会变。
    只对有 ≥2 个批次的视频算（32 号 P1：85 个 bvid 出现在 ≥2 批）。"""
    bv = rec['bvid']
    rows = [v for v in snap['videos'] if v['bvid'] == bv]
    if len(rows) < 2:
        return None
    out = []
    for v in sorted(rows, key=lambda x: x['batch_date']):
        # reuse the same snap and skip sensitivity; otherwise batch_check <-> diagnose recurses
        d = diagnose(bv, v['batch_date'], snap=snap, include_sensitivity=False)
        if not d:
            continue
        top = d['deviations'][0] if d['deviations'] else None
        out.append({
            'batch_date': v['batch_date'],
            'problem': d['verdict']['problem_5s'],
            'tone': d['verdict']['tone'],
            'top_label': top['label'] if top else '—',
            'top_pole': top['pole'] if top else '',
            'pct': d['position']['percentile'],
            'n_danmaku': v.get('n_danmaku'),
        })
    same = len({r['problem'] for r in out}) == 1
    return {'rows': out, 'stable': same, 'n_batches': len(out)}


def sensitivity_payload(snap, rec):
    return {
        'theta': {'current': THETA_CURRENT, 'total_vb': TOTAL_VB,
                  'rows': [{'theta': t, 'events': e, 'vb': v, 'cover': c, 'per_vb': p}
                           for t, e, v, c, p in THETA_SCAN],
                  'note': '出处 docs/05-新系统重构/22_theta敏感性扫描.md §1（原样搬运，未重算）'},
        'z': {'rows': z_sensitivity(snap),
              'note': '偏离候选闸门；偏离卡就是按这个阈值筛的（32 号 P4 惯例 z>3.5）'},
        'knn': {'rows': knn_sensitivity(snap, rec),
                'note': '第 K 近的相似度；相似对标只展示 3 支，K 再大就不像了'},
        'batch': batch_check(snap, rec),
        'batch_stability': {'within_median': 0.992, 'between_median': 0.069, 'between_p90': 0.814,
                            'n_multi': 85,
                            'note': '32 号 P1：形状签名跨批 within 中位 0.992 > between P90 0.814'},
    }


def deviation_view(snap=None):
    """偏离检测：全库「最不典型」排序 + 分区偏离率分布。

    口径 = 32 号 P4 的稳健 z：z = (v − 分区中位) / (1.4826·MAD)，取每支视频在自身分区内
    各展示维上的**最大 |z|** 作为「有多不典型」。阈值沿用 P4 惯例 z >= 3.5。

    ⚠️ 只统计**样本 >= 20 支的分区**（快照 snapshot_meta.min_partition 的同一口径）。
    为什么必须过滤（2026-10-16 实测）：不过滤时榜首出现 z=185，而它来自一个 n=8 的分区
    （该维中位 0.0006、MAD 0.00088，整区几乎全 0，一支 0.1629 就成了 185 倍）；
    另一个 z=153 来自 n=3 的分区 —— 小分区算不出稳定的中位/MAD，z 不可解释。
    过滤后 maxZ 185 -> 43，且判偏离的维度从「观众来得快不快」变成有意义的
    「爆点位置 / 节奏起伏 / 开场吸引力」；全库偏离率 33.1% -> 31.3%（头条数字不变）。

    这页只回答「跟同类不一样」，不回答「好不好」——后者由 32 号 P3 控龄关联决定。
    """
    snap = snap or _load()
    THR = 3.5
    MIN_N = 20

    def ok_part(tname):
        b = snap['baselines'].get(tname) or {}
        return (b.get('n_videos') or 0) >= MIN_N

    rank = [(z, i) for z, i in snap['outlier_rank']
            if ok_part(snap['videos'][i].get('tname'))]

    per = {}
    for z, i in rank:
        v = snap['videos'][i]
        t = v.get('tname') or '未标注'
        a = per.setdefault(t, {'n': 0, 'n_out': 0})
        a['n'] += 1
        if z >= THR:
            a['n_out'] += 1
    parts = []
    for t, a in per.items():
        parts.append({'tname': t, 'n': a['n'], 'n_out': a['n_out'],
                      'rate': round(100.0 * a['n_out'] / max(1, a['n']), 1)})
    parts.sort(key=lambda r: (-r['rate'], -r['n']))

    n_all = max(1, len(rank))
    n_out = sum(1 for z, _ in rank if z >= THR)
    rows = []
    for z, i in rank:
        v = snap['videos'][i]
        base = snap['baselines'].get(v.get('tname')) or snap['global_base']
        dim = _max_dim(v, base)
        if not dim:
            continue
        zz = abs(_z(v, base, dim) or 0.0)
        cur = snap['curves'].get('%s|%s' % (v['bvid'], v['batch_date']))
        rows.append({'bvid': v['bvid'],
                     'title': v.get('title') or v['bvid'],
                     'tname': v.get('tname'),
                     'n_peers': (base.get('n_videos') or 0),
                     'label': DICT[dim]['label'],
                     'pole': pole_word(dim, _z(v, base, dim) or 0.0),
                     'level': level_of(zz),
                     'z': round(zz, 2),
                     'pct': (v.get('pct') or {}).get(dim),
                     'curve': [round(x, 6) for x in _smooth(cur)] if cur else None})

    # 偏离维度的分布：全库被判偏离的视频里，各维度各占多少（哪个维度最常偏离）。
    # 这回答的是「这个市场里，大家最常在**哪一维**上不像同类」，跟 ③ 的分区视角互补。
    dimc = {}
    for z, i in rank:
        if z < THR:
            continue
        v = snap['videos'][i]
        b = snap['baselines'].get(v.get('tname')) or snap['global_base']
        dim = _max_dim(v, b)
        if dim:
            lab = DICT[dim]['label']
            dimc[lab] = dimc.get(lab, 0) + 1
    dim_dist = [{'label': k, 'n': v2} for k, v2 in dimc.items()]
    dim_dist.sort(key=lambda r: -r['n'])
    for r in dim_dist:
        r['rate'] = round(100.0 * r['n'] / max(1, n_out), 1)

    bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 999)]
    dist = [{'lo': lo, 'hi': (hi if hi < 999 else None),
             'n': sum(1 for z, _ in rank if lo <= z < hi)} for lo, hi in bins]

    n_bv = len({snap['videos'][i]['bvid'] for _, i in rank})
    n_batch = len({snap['videos'][i]['batch_date'] for _, i in rank})
    top_part = parts[0] if parts else {'rate': 0, 'tname': '—', 'n': 0}
    n_part_all = len(snap['baselines'])
    return {
        'headline': '样本 >= 20 支的分区里，%d 支视频中有 %d 支（%.1f%%）在某一维上明显偏离同类'
                    % (n_all, n_out, 100.0 * n_out / n_all),
        'scope_note': ('只统计样本 >= %d 支的分区（%d/%d 个）—— 小分区算不出稳定的中位与 MAD，'
                       'z 会被放大到不可解释（实测未过滤时榜首 z=185 来自 n=8 的分区）。'
                       % (MIN_N, len(parts), n_part_all)),
        'z_note': ('z 是 MAD 倍数（稳健 z）。某个维度在同类里几乎没有差异时（MAD 很小），'
                   'z 会被放大 —— 所以榜上同时给出 z 与分区样本量，两个一起看。'),
        'threshold': THR,
        'min_partition': MIN_N,
        'stats': [
            {'label': '纳入视频', 'value': n_all, 'unit': '支', 'note': '%d 个批次' % n_batch},
            {'label': '检出偏离', 'value': n_out, 'unit': '支',
             'note': 'z >= %.1f（32 号 P4 惯例）' % THR},
            {'label': '偏离占比', 'value': round(100.0 * n_out / n_all, 1), 'unit': '%',
             'note': '样本 >= %d 支的分区内' % MIN_N},
            {'label': '分区偏离率最高', 'value': top_part['rate'], 'unit': '%',
             'note': '%s（%d 支）' % (top_part['tname'], top_part.get('n', 0))},
        ],
        'parts': parts,
        'dist': dist,
        'dim_dist': dim_dist,
        'rows': rows[:40],
        'n_rows': len(rows),
        'n_unique_bvid': n_bv,
    }


def overview_view():
    return overview(_load())


def partition_compare(snap=None, top=6):
    """分区间节奏对比：取样本最足的 N 个分区，给它们的中位曲线 + 关键指标 + 偏离率。

    为什么要这一页：分区画像原来只有「本分区 vs 全市场」一条线，
    看不出**分区之间**彼此差在哪。这里把几个大区的中位曲线叠在一张图上，
    形状差在哪一眼能看出来。
    """
    snap = snap or _load()
    dev = deviation_view(snap)
    rate = {p['tname']: p['rate'] for p in dev['parts']}
    parts = sorted([(t, b) for t, b in snap['baselines'].items()
                    if (b.get('n_videos') or 0) >= MIN_PARTITION and not b.get('low_support')],
                   key=lambda kv: -kv[1]['n_videos'])[:top]
    out = []
    for t, b in parts:
        hm = (b['feats'].get('dm_per_min') or {}).get('med')
        op = (b['feats'].get('opening_5pct') or {}).get('med')
        mi = (b['feats'].get('mid_ratio') or {}).get('med')
        en = (b['feats'].get('rhythm_entropy') or {}).get('med')
        out.append({
            'tname': t, 'n_videos': b['n_videos'],
            'curve': [round(x, 6) for x in _smooth(b['curve_med'])],
            'dm_per_min': round(hm, 1) if hm else None,
            'opening_5pct': round(op * 100, 2) if op is not None else None,
            'mid_ratio': round(mi, 3) if mi is not None else None,
            'rhythm_entropy': round(en, 3) if en is not None else None,
            'dev_rate': rate.get(t),
        })
    return {
        'headline': '样本最足的 %d 个分区，中位曲线叠在一张图上看' % len(out),
        'sub': ('每个分区都只跟自己区内的视频比 —— 所以同一个数字在不同分区含义不同。'
                '下面这几条是各分区的中位节奏，形状差在哪就是观众习惯差在哪。'),
        'parts': out,
        'metric_note': '表里是中位数：条/分 = 互动热度；前 5% 占比 = 开场吸引力；'
                       '中段占比 = 中段留存；起伏熵 = 有没有高潮。偏离率见「偏离检测」页。',
    }


def deviation_data():
    return deviation_view(_load())


def partition_compare_data(top=6):
    return partition_compare(_load(), top)


def partition_view(tname):
    return partition_profile(_load(), tname)


def partition_list():
    snap = _load()
    rows = []
    for t, b in snap['baselines'].items():
        if b.get('n_videos', 0) < 8:
            continue
        rows.append({'tname': t, 'n_videos': b['n_videos'], 'low_support': bool(b.get('low_support'))})
    rows.sort(key=lambda r: (-r['n_videos'], r['tname']))
    return rows


# ── 跨批追踪（= 收藏追踪的跨批变化）────────────────────────────
# 受众：UP 主 + MCN/运营。它帮用户做成的**那一件事**：看一支视频"跟上一次比变了没"。
#
# ⚠️ 数据上限（2026-10-16 实测，它直接决定"完整"的定义）：
#   803 支去重视频里 718 支只出现在 1 个批次、85 支出现在 2 个批次、
#   **0 支出现在 >=3 个批次**。⇒ 最多只有两个时间点，**画不出趋势线**。
#   所以这里给的是"两批之间的逐维差（delta）"，**不是**时间序列趋势图。
#   这条必须守住，否则会做出"两点折线图"冒充趋势 —— 那不是数据支持的东西。
#
# 另一个真实含义（页面上要点明）：这 85 支 = **连续两批都在榜**，
# 正好对上 docs/05-新系统重构/07_最终设计方案.md 里 MCN 那条「榜单持续性预测」。
TRACK_THR = 3.5          # 偏离阈值，与 deviation_view 同口径（32 号 P4 惯例）


def _bd_gap_days(bd1, bd2):
    '''两个批次日期（YYYYMMDD 字符串）之间隔了几天'''
    try:
        a = datetime.datetime.strptime(str(bd1), '%Y%m%d')
        b = datetime.datetime.strptime(str(bd2), '%Y%m%d')
        return (b - a).days
    except Exception:
        return None


def _delta_phrase(dim, a, b):
    '''两批之间这一维"动了多少"的人话。

    形状维（direction=neutral）只说动没动，不判好坏 —— 与主视图同纪律：
    形状差异与后续热度没有稳定关联，所以不许把"变了"说成"变好了"。
    '''
    if a is None or b is None:
        return '这一批没测到'
    if dim == 'main_peak_pct':
        d = b - a
        if abs(d) < 2:
            return '位置基本没动'
        return ('往后挪了约 %d 个百分点' % round(abs(d))) if d > 0 \
            else ('往前挪了约 %d 个百分点' % round(abs(d)))
    if dim == 'peak_cnt':
        d = b - a
        if abs(d) < 1:
            return '个数没变'
        return ('多了 %d 个' % round(d)) if d > 0 else ('少了 %d 个' % round(-d))
    if a == 0:
        return '从 0 变成 %s' % _precise(dim, b)
    r = b / a
    if abs(r - 1.0) <= 0.08:
        return '基本没动'
    return ('高了约 %d%%' % round((r - 1) * 100)) if r > 1 \
        else ('低了约 %d%%' % round((1 - r) * 100))


def _real_move(dim, a, b):
    '''数值**本身**有没有实质变化。

    必须与"|z2-z1|"分开：z 是相对**同类基准**的偏离度，而两次采集用的是各自批次的基准；
    所以基准变了，z 会变，即使这支视频的值一点没动。把两者混为一谈会写出
    「【中段留存】基本没动」这种自相矛盾的结论句（实测踩过）。
    '''
    if a is None or b is None:
        return False
    if dim == 'main_peak_pct':
        return abs(b - a) >= 2
    if dim == 'peak_cnt':
        return abs(b - a) >= 1
    if a == 0:
        return b != 0
    return abs(b / a - 1.0) > 0.08


def _track_groups(snap):
    '''按 bvid 归组，只保留出现在 >=2 个批次的（当前全库 85 支）'''
    g = {}
    for i, v in enumerate(snap['videos']):
        g.setdefault(v['bvid'], []).append(i)
    out = []
    for bvid, idxs in g.items():
        if len(idxs) < 2:
            continue
        idxs.sort(key=lambda i: snap['videos'][i]['batch_date'])
        out.append((bvid, idxs))
    return out


def _pair_summary(snap, idxs):
    v1, v2 = snap['videos'][idxs[0]], snap['videos'][idxs[-1]]
    b1 = snap['baselines'].get(v1.get('tname')) or snap['global_base']
    b2 = snap['baselines'].get(v2.get('tname')) or snap['global_base']
    deltas = []
    for dim, d in DICT.items():
        if not d.get('display'):
            continue
        a = (v1.get('f') or {}).get(dim)
        b = (v2.get('f') or {}).get(dim)
        if a is None and b is None:
            continue
        z1, z2 = _z(v1, b1, dim), _z(v2, b2, dim)
        # 变化幅度用"两批偏离度的差"（|z2-z1|）—— 与展示维同口径，
        # 且是"相对同类"的变化，不是被时长/总量混淆的绝对差。
        moved = abs(z2 - z1) if (z1 is not None and z2 is not None) else None
        crossed = None
        if z1 is not None and z2 is not None:
            crossed = (abs(z1) < TRACK_THR) != (abs(z2) < TRACK_THR)
        deltas.append({
            'dim': dim, 'label': d['label'], 'seg_name': d.get('seg_name'),
            'unit': d.get('unit'), 'direction': d.get('direction'),
            'v1_text': _precise(dim, a), 'v2_text': _precise(dim, b),
            'phrase': _delta_phrase(dim, a, b),
            'real_move': _real_move(dim, a, b),
            'z1': round(z1, 2) if z1 is not None else None,
            'z2': round(z2, 2) if z2 is not None else None,
            'crossed': crossed,
            'moved': round(moved, 3) if moved is not None else None,
        })
    deltas.sort(key=lambda r: -(r['moved'] or 0))
    moved_ones = [r for r in deltas if r['real_move']]
    top = moved_ones[0] if moved_ones else None
    n_changed = len(moved_ones)
    n_crossed = sum(1 for r in deltas if r['crossed'])
    if not deltas or all(r['moved'] is None for r in deltas):
        conclusion = '这两批的节奏差没能算出来（缺少可比的分区基准）。'
    elif top is None and n_crossed == 0:
        conclusion = '跟上一批比，节奏没怎么变。'
    elif top is None:
        conclusion = ('跟上一批比，各维的数值几乎没动，但有 %d 维跨过了偏离阈值 —— '
                      '那是同类基准在两次采集之间变了，不是这支视频变了。' % n_crossed)
    else:
        conclusion = '跟上一批比，【%s】%s。' % (top['label'], top['phrase'])
    return {
        'bvid': v1['bvid'], 'title': v1.get('title') or v1['bvid'],
        'tname': v1.get('tname'), 'up_name': v1.get('up_name'),
        'batches': [{'batch_date': v1['batch_date'], 'n_danmaku': v1.get('n_danmaku')},
                    {'batch_date': v2['batch_date'], 'n_danmaku': v2.get('n_danmaku')}],
        'gap_days': _bd_gap_days(v1['batch_date'], v2['batch_date']),
        'n_changed': n_changed, 'n_crossed': n_crossed,
        'score': top['moved'] if (top and top['moved'] is not None) else 0,
        'top': top, 'deltas': deltas,
        'conclusion': conclusion,
    }


def track_list(snap=None):
    '''跨批追踪总表：**直接列出那 85 支**（而不是做成一个 89% 时间是空的页面）。

    为什么直接列：如果只把它挂在"我的收藏"里，约 89% 的收藏会显示"暂无追踪数据"——
    用户看不到这个功能到底能用在哪里。所以另开一个入口，把真正有两次数据的列出来。
    '''
    snap = snap or _load()
    items = [_pair_summary(snap, idxs) for _, idxs in _track_groups(snap)]
    items.sort(key=lambda r: -(r['score'] or 0))
    n_all = len({v['bvid'] for v in snap['videos']})
    n2 = len(items)
    share = round(100.0 * n2 / n_all, 1) if n_all else 0.0
    # 实际批次对（2026-10-16 实测，必须说出来）：84/85 是相邻两天，只有 1 支隔 3 天。
    # 不说出来，"跟上一次比变了没"会被读成"隔了很久的变化"——那是误导。
    gaps = {}
    for r in items:
        k = ' → '.join(b['batch_date'] for b in r['batches'])
        a = gaps.setdefault(k, {'pair': k, 'n': 0, 'gap_days': r['gap_days']})
        a['n'] += 1
    gap_pairs = sorted(gaps.values(), key=lambda x: (-x['n'], x['pair']))
    tp = gap_pairs[0] if gap_pairs else None
    gap_note = ('按实际批次对看：%d/%d 支是**相邻两天**（%s，日频采集）%s —— 所以'
                '"跟上一次比变了没"实际是"**隔 %s 天**变了没"，差值里有一部分可能来自'
                '采集本身（同一支视频在两天各采一次），不能全当作"视频变了"。'
                '要更长的跨度，只能继续按日频积累批次。'
                % (tp['n'], n2, tp['pair'],
                   ('，其余 %d 支隔更久' % (n2 - tp['n'])) if (tp and n2 > tp['n']) else '',
                   tp['gap_days'] if tp else '—')) if tp else ''
    return {
        'headline': '803 支视频里 %d 支（%.1f%%）连续两批都被采到 —— 这是全库能看"跟上一次比"的全部' % (n2, share),
        'sub': '**连续两批都在榜**，正好对上 07 设计里 MCN/运营 那条「榜单持续性预测」：'
               '这些就是"被市场连续选中"的候选集。',
        'stats': [
            {'label': '去重视频', 'value': n_all, 'unit': '支', 'note': '快照覆盖的全部'},
            {'label': '跨批出现', 'value': n2, 'unit': '支', 'note': '占 %.1f%%' % share},
            {'label': '最多批次数', 'value': 2, 'unit': '批', 'note': '0 支出现在 3 批及以上'},
            {'label': '主要间隔', 'value': (tp['gap_days'] if tp else 0), 'unit': '天',
             'note': '%d/%d 支是这一对批次' % (tp['n'], n2) if tp else ''},
        ],
        'day_note': '所有跨批视频都只有**两个时间点**（0 支出现在 3 个及以上批次）—— '
                    '所以给的是**两批之间的逐维差**，不是时间序列趋势线。',
        'gap_pairs': gap_pairs,
        'gap_note': gap_note,
        'items': items,
        'n_videos': n2, 'n_all': n_all, 'share_pct': share, 'max_batches': 2,
        'threshold': TRACK_THR,
    }


def track_detail(bvid, snap=None):
    '''单支视频的跨批详情：两批曲线叠加 + 逐维 delta 表'''
    snap = snap or _load()
    groups = dict(_track_groups(snap))
    idxs = groups.get(bvid)
    if not idxs:
        return None
    s = _pair_summary(snap, idxs)
    curves = []
    for i in idxs:
        v = snap['videos'][i]
        cur = snap['curves'].get('%s|%s' % (v['bvid'], v['batch_date']))
        curves.append({'batch_date': v['batch_date'], 'n_danmaku': v.get('n_danmaku'),
                       'duration': v.get('duration'),
                       'curve': [round(x, 6) for x in _smooth(cur)] if cur else None})
    s['curves'] = curves
    s['note'] = ('两批之间隔了 %s 天。图上 x 轴 = 播放进度（1%% 桶），y 轴 = 该位置的弹幕占比；'
                 '两条线是**同一支视频在两个批次**的形状，不是趋势线。'
                 % (s['gap_days'] if s['gap_days'] is not None else '？'))
    return s


# ── 两视频对比 ──────────────────────────────────────────────
# 受众：UP 主。它帮用户做成的那一件事：**对标竞品，看出差在哪一维**。
#
# ⚠️ 数据前提（2026-10-16 实测，它直接改设计）：
#   结果变量非空率很低（快照 888 条「视频-批次」）：halflife_d 28.2% / hk_halflife 18.7% /
#   span_days 15.2% / half_decay_days 14.1%。而配对要**两支都有同一个变量**，
#   概率约等于两个单支率的乘积（halflife_d 只有 ≈8%）⇒ 绝大多数配对比不出"谁更强"。
#   所以：**两支都有同一个结果变量才显示"谁更强"**，否则如实写"只能比节奏形状"——
#   绝不用别的字段凑一个"强弱"出来（这是本轮判据"不许用别的字段凑"的直接落地）。
CMP_DIMS = tuple(k for k, d in DICT.items() if d.get('display'))


def _cmp_phrase(dim, va, vb):
    '''第二支相对第一支差在哪（页面上列头写明这个方向）'''
    if va is None or vb is None:
        return '有一支没测到'
    if dim == 'main_peak_pct':
        d = vb - va
        if abs(d) < 2:
            return '爆点位置差不多'
        return (('第二支的爆点更靠后约 %d 个百分点' % round(abs(d))) if d > 0
                else ('第二支的爆点更靠前约 %d 个百分点' % round(abs(d))))
    if dim == 'peak_cnt':
        d = vb - va
        if abs(d) < 1:
            return '爆点个数相同'
        return ('第二支多 %d 个' % round(d)) if d > 0 else ('第二支少 %d 个' % round(-d))
    if va == 0:
        return '第一支为 0，第二支 %s' % _precise(dim, vb)
    r = vb / va
    if abs(r - 1.0) <= 0.08:
        return '差不多'
    return ('第二支高约 %d%%' % round((r - 1) * 100)) if r > 1 \
        else ('第二支低约 %d%%' % round((1 - r) * 100))


def _cmp_side(bvid, snap):
    i = snap['by_bvid'].get(bvid)
    if i is None:
        return None
    rec = snap['videos'][i]
    tname = rec.get('tname')
    base_p = snap['baselines'].get(tname) if tname else None
    base = base_p or snap['global_base']
    cur = snap['curves'].get('%s|%s' % (bvid, rec['batch_date']))
    med = (base or {}).get('curve_med')
    return {
        'bvid': bvid, 'title': rec.get('title') or bvid,
        'tname': tname, 'up_name': rec.get('up_name'),
        'batch_date': rec['batch_date'], 'duration': rec.get('duration'),
        'duration_text': _mmss(rec.get('duration')),
        'n_danmaku': rec.get('n_danmaku'), 'play': rec.get('play'),
        'curve': [round(x, 6) for x in _smooth(cur)] if cur else None,
        'median': [round(x, 6) for x in _smooth(med)] if med else None,
        'f': rec.get('f') or {}, 'pct': rec.get('pct') or {},
        'outcome': rec.get('outcome') or {},
        'partition_n': (base_p or {}).get('n_videos'),
        'partition_low': bool((base_p or {}).get('low_support')),
    }


def _outcome_coverage(snap):
    '''结果变量的**实际非空率**（现算，不硬编码）——用来如实交代"谁更强"为什么常常算不出来。'''
    n = len(snap['videos']) or 1
    out = []
    for k in OUT_ORDER:
        c = sum(1 for v in snap['videos'] if (v.get('outcome') or {}).get(k) is not None)
        out.append({'key': k, 'label': OUT_LABEL[k], 'n': c,
                    'pct': round(100.0 * c / n, 1),
                    # 两支同时有该变量的概率 ≈ 单支率²（上界估计，用于对照）
                    'pair_pct': round(100.0 * (c / n) ** 2, 1)})
    return out


_PAIR_STAT = None


def _pair_outcome_rate(snap):
    '''在**全部** C(n,2) 个两两配对里，能比"谁更强"的**精确**占比（不抽样、不用上界估计）。

    做法：把每支视频的结果变量存在情况压成 4 位掩码（16 种），统计每种掩码的支数，
    再在 16×16 上数"两个掩码有交集"的配对 —— O(n) 一次算好并缓存，
    因为 322,003 个配对逐个比会每次请求都白跑一遍。

    为什么值得精确算：页面上要写"多常能比出强弱"。写"约等于单支率平方"是**估计**，
    写"322,003 个配对里只有 X 个能比"是**数出来的**——后者才经得起追问。
    '''
    global _PAIR_STAT
    if _PAIR_STAT is not None:
        return _PAIR_STAT
    masks = {}
    for i in snap['by_bvid'].values():
        o = snap['videos'][i].get('outcome') or {}
        m = 0
        for b, k in enumerate(OUT_ORDER):
            if o.get(k) is not None:
                m |= (1 << b)
        masks[m] = masks.get(m, 0) + 1
    keys = sorted(masks)
    n = sum(masks.values())
    tot = n * (n - 1) // 2
    can = 0
    for a in keys:
        for b in keys:
            if a == b:
                if a:
                    can += masks[a] * (masks[a] - 1) // 2
            elif a < b and (a & b):
                can += masks[a] * masks[b]
    _PAIR_STAT = {'n_videos': n, 'n_pairs': tot, 'n_can': can,
                  'pct': (round(100.0 * can / tot, 1) if tot else 0.0)}
    return _PAIR_STAT


def compare_view(a, b, snap=None):
    snap = snap or _load()
    A, B = _cmp_side(a, snap), _cmp_side(b, snap)
    if not A or not B:
        return None
    rows = []
    for dim in CMP_DIMS:
        va, vb = A['f'].get(dim), B['f'].get(dim)
        pa, pb = A['pct'].get(dim), B['pct'].get(dim)
        rows.append({
            'dim': dim, 'label': DICT[dim]['label'], 'unit': DICT[dim].get('unit'),
            'seg_name': DICT[dim].get('seg_name'),
            'a_text': _precise(dim, va), 'b_text': _precise(dim, vb),
            'a_pct': pa, 'b_pct': pb,
            'pct_gap': (round(pa - pb, 1) if (pa is not None and pb is not None) else None),
            'phrase': _cmp_phrase(dim, va, vb),
        })
    # 表按"百分位差"从大到小 —— 排在上面的是这两支最不一样的地方
    rows.sort(key=lambda r: -(abs(r['pct_gap']) if r['pct_gap'] is not None else -1))

    cov = _outcome_coverage(snap)
    who = {'available': False, 'var': None, 'key': None, 'label': None,
           'a_value': None, 'b_value': None, 'winner': None, 'basis': None}
    # 在"两支都有"的结果变量里挑**覆盖最高**的那个来比：
    # ① 覆盖高 ⇒ 这个结论适用的配对面更广；② 覆盖高的两个都是连续值，
    # 少碰到 half_decay_days 这种"整数天"造成的并列（实测踩过：两支都 1.0 天 → 判不出强弱）。
    both = [k for k in OUT_ORDER if A['outcome'].get(k) is not None and B['outcome'].get(k) is not None]
    if both:
        cvm = {c['key']: c['pct'] for c in cov}
        k = max(both, key=lambda x: cvm.get(x, 0))
        va, vb = A['outcome'][k], B['outcome'][k]
        who = {'available': True, 'var': k, 'key': k, 'label': OUT_LABEL[k],
               'a_value': round(float(va), 2), 'b_value': round(float(vb), 2),
               'winner': ('a' if va > vb else ('b' if vb > va else None)),
               'basis': ('两支都有「%s」（%s，单支覆盖 %s%%）数据，按这一个变量比'
                         % (OUT_LABEL[k], k, cvm.get(k, '—')))}
    same_part = bool(A['tname']) and A['tname'] == B['tname']
    ps = _pair_outcome_rate(snap)
    return {
        'a': A, 'b': B,
        'rows': rows,
        'same_partition': same_part,
        'median_lines': ([A['median']] if same_part else [A['median'], B['median']]),
        'who': who,
        'who_note': (('这两支都有「%s」数据，所以能比强弱。' % who['label']) if who['available'] else
                     '这两支没有共同的后续表现数据，只能比节奏形状 —— '
                     '不拿别的字段凑一个"强弱"出来。'),
        'outcome_coverage': cov,
        'pair_stat': ps,
        'coverage_note': ('结果变量的覆盖很低（快照 %d 条「视频-批次」现算）：%s。'
                          '而对比要**两支都有同一个变量**：在全部 %s 个两两配对里，'
                          '只有 **%s%%**（%s 个）能比出"谁更强"，剩下的只能比节奏形状。'
                          '这是数据上限，不是功能缺失。'
                          % (len(snap['videos']),
                             '；'.join('%s %s%%' % (c['label'], c['pct']) for c in cov),
                             format(ps['n_pairs'], ','), ps['pct'], format(ps['n_can'], ','))),
        'scope_note': (('两支同在【%s】，同类基准一致，百分位可直接比。' % A['tname']) if same_part else
                       ('两支分属【%s】与【%s】—— 同类基准不同，所以逐维比的是各自分区内的**百分位**，'
                        '绝对值只作参考。' % (A['tname'] or '未标注', B['tname'] or '未标注'))),
        'gap_hint': '表按"百分位差"从大到小排 —— 排在最上面的是这两支最不一样的地方。',
    }
