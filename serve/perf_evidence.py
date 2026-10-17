# -*- coding: utf-8 -*-
'''「为什么用 Spark」证据页供数（08 §10.3 / B5.1 性能实验）——**受众是答辩组**。

性质：只读。数据 = serve/v3data/perf.json（由 tools/make_perf_snapshot.py 从
data-out/perf_results.jsonl 导出），**不重跑实验、不新造任何数字**。

为什么单独成页（规则 13 题目锚定）：
  题目第一个词就是「基于 Spark」，而上一版界面里它在**空转** —— 没有任何一处能回答
  「分布式到底证明了什么」。这页把 12 条实测摆出来，且受众明标为答辩组。

口径纪律（页面必须照实说，不许把两件事说成一件）：
  · 线性的是**数据量**（n_out ×4.0003 / ×7.987）；墙钟是**亚线性**（×8 数据 → 单并行只 ×1.77）。
  · 放大是「重放+扰动」合成放大，不是真实流量。
  · VM 单 worker × 2 核 ⇒ par=4 不可得（记录在 Makefile 的 SUBMIT conf）。
'''
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v3data')
_P = None


def _load():
    global _P
    if _P is None:
        with open(os.path.join(DATA_DIR, 'perf.json'), encoding='utf-8') as f:
            _P = json.load(f)
    return _P


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def _pick(recs, mode, amp=None, par=None):
    for r in recs:
        if r.get('mode') != mode:
            continue
        if amp is not None and r.get('amp') != amp:
            continue
        if par is not None and str(r.get('par')) != str(par):
            continue
        return r
    return None


def _ratio(a, b):
    if not a or not b:
        return None
    return round(float(a) / float(b), 2)


def view():
    p = _load()
    recs = p.get('records') or []
    AMP_LABEL = {1: '1×', 4: '4×', 8: '8×'}

    # ① 规模放大：同一份作业，数据量按倍率放大，各跑 1 路 / 2 路并行
    scaling = []
    base_n = None
    for amp in (1, 4, 8):
        r1 = _pick(recs, 'density', amp, 1)
        r2 = _pick(recs, 'density', amp, 2)
        n_out = (r1 or r2 or {}).get('n_out')
        if base_n is None:
            base_n = n_out
        scaling.append({
            'label': AMP_LABEL[amp],
            'n_out': n_out,
            'volume_x': _ratio(n_out, base_n),
            'wall_par1': _r((r1 or {}).get('wall_s')),
            'wall_par2': _r((r2 or {}).get('wall_s')),
            'speedup': _ratio((r1 or {}).get('wall_s'), (r2 or {}).get('wall_s')),
        })
    wall_x_par1 = _ratio(scaling[2]['wall_par1'], scaling[0]['wall_par1'])
    wall_x_par2 = _ratio(scaling[2]['wall_par2'], scaling[0]['wall_par2'])

    # ② 并行加速比：1 路 → 2 路（密度三档 + Hawkes 两次拟合）
    speedups = []
    for s in scaling:
        speedups.append({'name': '密度重算 · ' + s['label'],
                         'one': s['wall_par1'], 'two': s['wall_par2'],
                         'x': s['speedup'], 'kind': 'density'})
    for i, r in enumerate([x for x in recs if x.get('mode') == 'hawkes'], 1):
        speedups.append({'name': 'Hawkes 拟合 · 第 %d 次' % i,
                         'one': _r(r.get('serial_s')), 'two': _r(r.get('parallel_s')),
                         'x': _ratio(r.get('serial_s'), r.get('parallel_s')), 'kind': 'hawkes'})

    # ③ 事件网格：事件表（不是密度表）在不同放大下的墙钟
    events = []
    for amp in (1, 4, 8):
        r = _pick(recs, 'events', amp, 2)
        if not r:
            continue
        events.append({'label': AMP_LABEL[amp],
                       'wall': _r(r.get('wall_s')), 'n_events': r.get('n_events'),
                       'n_series': r.get('n_series')})

    # ④ 数据倾斜：加盐两阶段 vs 直算
    skew = None
    rs = _pick(recs, 'skew')
    if rs:
        skew = {'plain_s': _r(rs.get('plain_s')), 'salted_s': _r(rs.get('salted_s')),
                'x': _ratio(rs.get('plain_s'), rs.get('salted_s')),
                'parts': (rs.get('plain') or {}).get('parts'),
                'note': '同一份数据，加盐打散后走两阶段聚合；本条数据本身 skew 很低'
                        '（plain_skew=%s），所以它证明的是"加盐不亏"，不是"加了盐救了灾难"。'
                        % _r(rs.get('plain_skew'), 3)}

    fast = [s for s in speedups if s['x'] is not None]
    best = max(fast, key=lambda s: s['x']) if fast else None
    return {
        'source': p.get('source'),
        'snapshot_at': p.get('snapshot_at'),
        'n_records': p.get('n_records'),
        'headline': '「基于 Spark」不是挂名：放大 8 倍数据、2 路并行、倾斜对照，共 %d 条墙钟实测' % len(recs),
        'sub': '集群只有 1 台虚拟机上的 1 个 Spark worker、2 个核 —— 答案不是'
               '「集群有多大」，而是「数据变大、并行变多时，时间怎么变」。',
        'scaling': scaling,
        'wall_x_par1': wall_x_par1,
        'wall_x_par2': wall_x_par2,
        'scaling_note': ('线性的是**数据量**（%s×）；墙钟是**亚线性**（单并行 %s×、双并行 %s×）。'
                         '两者不是一回事，所以不写成"时间线性增长"。'
                         % (scaling[2]['volume_x'], wall_x_par1, wall_x_par2)),
        'speedups': speedups,
        'best_speedup': best,
        'events': events,
        'skew': skew,
        'limits': p.get('limits') or [],
        'records': [{'mode': r.get('mode'), 'amp': r.get('amp'), 'par': r.get('par'),
                     'wall_s': r.get('wall_s'), 'ts': (r.get('ts') or '')[:19].replace('T', ' ')}
                    for r in recs],
        'suite': p.get('suite'),
        'records_note': ('每条都可回到 data-out/perf_results.jsonl 逐字核对；'
                         'perf.json 由 tools/make_perf_snapshot.py 从该文件导出，不手抄。'),
    }
