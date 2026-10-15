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
import json
import math
import os

from dict_zh import (ACTION_CAVEAT, CONCERN, CURVE_CAVEAT, DICT, DIR, DICT_VERSION,
                     human_pos, label_of, level_of, pole_word, pos_of, sim_label)

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
    _SNAP = d
    return d


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
    if sec is None:
        return None
    s = int(round(sec))
    return '%02d:%02d' % (s // 60, s % 60)


def _curve_block(rec, base, key):
    snap = _load()
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


def _deviations(rec, base, pct_map, benchmarks):
    out = []
    for a, z, dim in _pick_deviations(rec, base, pct_map):
        d = DICT[dim]
        v = rec['f'].get(dim)
        med = base['feats'][dim]['med']
        pct = pct_map.get(dim)
        direction, human = _direction(dim, z)
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
            'value': round(v, 6) if v is not None else None,
            'median': round(med, 6) if med is not None else None,
        })
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


def _benchmarks(rec, base, pool_idx):
    snap = _load()
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
            'like_text': like, 'learn_text': learn_txt,
            'outcome_text': otext or '暂无后续追踪数据',
            'outcome_value': oval, '_outcome_key': o_key, 'outcome_rank': o_rank,
            'did_better': evidence,
            'curve': [round(x, 6) for x in _smooth(cur)] if cur else None,
            'has_outcome': otext is not None,
        })
    return out


def _wan(v):
    if v is None:
        return None
    v = float(v)
    return ('%.1f 万' % (v / 10000.0)) if v >= 10000 else ('%d' % round(v))


def _position(rec, base, pct_map, pool_idx, partition_ok):
    snap = _load()
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
        out.append({'n': 1, 'text': concern['advice'],
                    'basis': '%s%s（%s）' % (concern['label'], concern['pole'],
                                          concern['compare_text']),
                    'ref': concern.get('ref'), 'seg': concern.get('seg')})
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
                    'ref': {'bvid': better['bvid'], 'title': better['title']}, 'seg': None})
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
        'actions': _actions(position, concern, deviations, benchmarks),
        'caveat': ACTION_CAVEAT, 'curve_caveat': CURVE_CAVEAT,
        'partition_used_full_sample': not partition_ok,
    }


def diagnose(bvid, batch_date=None):
    snap = _load()
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
    benchmarks = _benchmarks(rec, base if partition_ok else None, pool_idx)
    knn_scope = (tname if len(same_t) >= 8 else '全体样本')
    deviations = _deviations(rec, base, pct_map, benchmarks)
    curve = _curve_block(rec, base, '%s|%s' % (bvid, rec['batch_date']))
    position = _position(rec, base, pct_map, pool_idx, partition_ok)
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

