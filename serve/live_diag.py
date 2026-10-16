# -*- coding: utf-8 -*-
"""live_diag.py — 在线单视频诊断（库外 bvid）

为什么要有这条链路：诊断卡的核心价值是「拿**自己的**视频跟库内基线比」。
没有它时，输入一个库外 bvid 只会得到「没找到」；更糟的是旧搜索框拿输入当子串
模糊匹配，会默默加载一支**无关的库内视频**，看起来像"诊断了但诊断的不是它"。

口径与库内严格一致（逐条对照，改这里必须同步改这几处）：
  派生（jobs/m0_full.py）：
    offset_pct  = time_offset / duration
    lag_sec     = send_time - pubdate
    attr_weight = w_mode × w_color × w_font（pool == 1 字幕池 → 0）
  聚合（jobs/m2_l1_density.py + jobs/m2_l1_fingerprint.py）：
    1% 桶       = min(floor(offset_pct * 100), 99)
    main_peak_pct = wcnt 最大的那个 1% 桶
    opening_5pct / ending_5pct / mid_ratio = 对应区间弹幕占比
    early_lag_ratio = lag_sec < 21600 的占比
    rhythm_entropy  = (-Σ c·ln c / tot + ln tot) / ln 100
    burst_cv        = std/mean over **稠密** 5s 桶（0..max，缺行补 0）
    dm_per_min = n / (duration/60)；wdm_per_min = Σw / (duration/60)
  比对：把算好的记录注入快照后**直接调 diagnosis.diagnose()** ——
    参照系、稳健 z、百分位、kNN、结论、行动全部复用库内那套，不另写一份。

⚠️ 与库内的**已知差异**（页面必须如实写出，不许藏）：
  1. 不做刷量排除 —— is_spam 需要批次级 user_hash 统计（同一用户在批次内的
     占比与爆发度），单视频算不出来。所以在线结果可能含刷量弹幕。
  2. 弹幕池是**实时快照**，可能被 B站截断（历史上限约 1800 条）。
  3. peak_cnt 不在线计算 —— 库内该列由 L2 事件回填，此处无对应来源。
"""
import gzip
import json
import math
import re
import time
import urllib.error
import urllib.request

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36')
HDR = {'User-Agent': UA, 'Referer': 'https://www.bilibili.com/'}

# 5s / 1% / 6h 与库内一致
BUCKET_SEC = 5
LAG_EARLY = 21600
SEG_SEC = 360          # PB 每段 6 分钟（B站口径）
WHITE = 16777215       # 默认白色（不算"彩色"）


def _get(url, params='', timeout=20):
    req = urllib.request.Request(url + params, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        enc = (r.headers.get('Content-Encoding') or '').lower()
        return _unzip(raw, enc)


def _unzip(raw, enc):
    '''按 Content-Encoding 解压。

    ⚠️ 2026-10-16 修：原来**只解 gzip**，而 dm/list.so 与 comment.bilibili.com 返回的是
    **deflate** —— 于是 XML 那条路一直把压缩包当文本用（表现为乱码、<d p= 命中 0），
    只能退回 PB 分段；而 PB 分段有额度，采不满。库内采集走 requests（自动解压）所以是满的。
    这条 bug 直接导致在线诊断的样本量只有库内的一半左右，会污染形状指纹。
    '''
    if enc == 'gzip':
        return gzip.decompress(raw)
    if enc == 'deflate':
        import zlib
        try:
            return zlib.decompress(raw)              # zlib 包装
        except zlib.error:
            return zlib.decompress(raw, -zlib.MAX_WBITS)   # 裸 deflate
    if enc == 'br':
        try:
            import brotli
            return brotli.decompress(raw)
        except ImportError:
            return raw
    return raw


# ── 纯 Python protobuf 读（不引 protobuf 依赖，也不在镜像里跑 protoc）──
def _varint(b, i):
    r = s = 0
    while True:
        x = b[i]
        i += 1
        r |= (x & 0x7F) << s
        if not (x & 0x80):
            return r, i
        s += 7


def _iter_fields(b):
    i, n = 0, len(b)
    while i < n:
        try:
            key, i = _varint(b, i)
        except IndexError:
            return
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
            yield fn, v
        elif wt == 2:
            ln, i = _varint(b, i)
            yield fn, b[i:i + ln]
            i += ln
        elif wt == 5:
            yield fn, b[i:i + 4]
            i += 4
        elif wt == 1:
            yield fn, b[i:i + 8]
            i += 8
        else:
            return


def parse_seg(raw):
    """DmSegMobileReply → [{progress_ms, mode, fontsize, color, send_time, pool, text}]"""
    out = []
    for fn, val in _iter_fields(raw):
        if fn != 1 or not isinstance(val, bytes):
            continue                      # field 1 = repeated DanmakuElem
        e = {'progress_ms': 0, 'mode': 1, 'fontsize': 25, 'color': WHITE,
             'send_time': 0, 'pool': 0, 'text': ''}
        for f2, v2 in _iter_fields(val):
            if f2 == 2:
                e['progress_ms'] = v2
            elif f2 == 3:
                e['mode'] = v2
            elif f2 == 4:
                e['fontsize'] = v2
            elif f2 == 5:
                e['color'] = v2
            elif f2 == 7:
                e['text'] = v2.decode('utf-8', 'ignore')
            elif f2 == 8:
                e['send_time'] = v2
            elif f2 == 11:
                e['pool'] = v2
        if e['text']:
            out.append(e)
    return out


def fetch_xml(cid):
    """XML 全量弹幕 —— **与库内采集（collect/bili.py::fetch_danmaku_xml）同一条路径**。

    ⚠️ 为什么必须优先走这条：它一次给**整个保留池**；而 seg.so(PB) 是分段的、有额度，
    实测同一支视频 PB 只拿到 677 条 / XML 拿到 1417 条（与 B站报告数一致）。
    在线诊断要跟库内比形状，样本量差一半会系统性污染 opening_5pct / mid_ratio / ending_5pct。
    """
    raw = _get('https://api.bilibili.com/x/v1/dm/list.so', '?oid=%d' % cid)
    txt = raw.decode('utf-8', 'ignore')
    out = []
    for attr, text in re.findall(r'<d p="([^"]+)"[^>]*>(.*?)</d>', txt):
        p = attr.split(',')
        if len(p) < 8:
            continue
        t = text.strip()
        if not t:
            continue
        try:
            out.append({
                'progress_ms': int(float(p[0]) * 1000),
                'mode': int(p[1]), 'fontsize': int(p[2]), 'color': int(p[3]),
                'send_time': int(p[4]), 'pool': int(p[5]), 'text': t,
            })
        except ValueError:
            continue
    return out


def fetch_danmaku(cid, duration):
    """XML 优先，PB 兜底。返回 (弹幕列表, 来源, 段数)。"""
    dms = []
    try:
        dms = fetch_xml(cid)
    except Exception:
        dms = []
    if dms:
        return dms, 'xml', 0
    pb, segs, blocked = fetch_pb(cid, duration)
    if blocked:
        return [], 'blocked', segs
    return pb, 'pb', segs


def fetch_pb(cid, duration):
    """逐段拉 PB。返回 (弹幕列表, 段数, 是否提前停止)。"""
    est = max(1, int((duration + SEG_SEC - 1) // SEG_SEC))
    dms, segs = [], 0
    for seg in range(1, min(est * 2 + 4, 60)):
        try:
            raw = _get('https://api.bilibili.com/x/v2/dm/web/seg.so',
                       '?type=1&oid=%s&segment_index=%d' % (cid, seg), timeout=20)
        except urllib.error.HTTPError as e:
            if e.code == 412:
                return dms, segs, 'blocked'
            break
        except Exception:
            break
        if len(raw) < 10:
            break
        got = parse_seg(raw)
        segs += 1
        if not got:
            break
        dms.extend(got)
        if len(got) < 50:
            break
        time.sleep(0.25)                   # 单视频最多几十次请求，这个间隔足够克制
    return dms, segs, None


def fetch_view(bvid):
    """视频元信息。cid / duration / pubdate / title / 分区名。"""
    j = json.loads(_get('https://api.bilibili.com/x/web-interface/view',
                        '?bvid=' + bvid, timeout=15).decode('utf-8'))
    if j.get('code') != 0 or not j.get('data'):
        return None
    d = j['data']
    return {
        'bvid': bvid, 'cid': d.get('cid'), 'duration': d.get('duration') or 0,
        'pubdate': d.get('pubdate') or 0, 'title': d.get('title') or bvid,
        'tname': (d.get('tname') or '').strip(),
        'up_name': (d.get('owner') or {}).get('name') or '',
        'play': ((d.get('stat') or {}).get('view') or 0),
        'danmaku_reported': ((d.get('stat') or {}).get('danmaku') or 0),
    }


# ── attr_weight（孔玲玲权重表，m0_full.py 逐条对齐）──
def attr_weight(mode, color, fontsize, pool):
    if pool == 1:                          # 字幕池 → 0
        return 0.0
    w_mode = {1: 1.0, 4: 1.2, 5: 1.3, 6: 1.1, 7: 1.2}.get(mode, 1.0)
    if mode in (8, 9):
        w_mode = 0.0
    w_color = 1.0 if color == WHITE else 1.3
    if fontsize == 25:
        w_font = 1.0
    elif fontsize > 25:
        w_font = 1.2
    elif fontsize < 18:
        w_font = 0.9
    else:
        w_font = 1.0
    return w_mode * w_color * w_font


SHAPE_DIMS = ['opening_5pct', 'ending_5pct', 'mid_ratio',
              'rhythm_entropy', 'burst_cv', 'early_lag_ratio']
DIMS = ['dm_per_min', 'wdm_per_min', 'opening_5pct', 'early_lag_ratio', 'main_peak_pct',
        'rhythm_entropy', 'burst_cv', 'peak_cnt', 'ending_5pct', 'mid_ratio', 'avg_text_len']


def compute(dms, duration, pubdate):
    """由弹幕列表算 11 维 + 101 点曲线。返回 (f, shape, curve)。"""
    n = len(dms)
    wsum = 0.0
    tlen = 0
    b1 = [0] * 100          # 1% 桶：cnt
    w1 = [0.0] * 100        # 1% 桶：wcnt
    b5 = {}                 # 5s 桶：cnt
    o5 = e5 = mid = early = 0
    peak_pct = 0
    for d in dms:
        pct = d['progress_ms'] / 1000.0
        off = (pct / duration) if duration > 0 else None
        w = attr_weight(d['mode'], d['color'], d['fontsize'], d['pool'])
        wsum += w
        txt = d['text']
        tlen += len(txt)
        b5[int(pct // BUCKET_SEC)] = b5.get(int(pct // BUCKET_SEC), 0) + 1
        if off is not None:
            i = min(int(off * 100), 99)
            b1[i] += 1
            w1[i] += w
            if off < 0.05:
                o5 += 1
            elif off > 0.95:
                e5 += 1
            else:
                mid += 1
        if pubdate and d['send_time'] and (d['send_time'] - pubdate) < LAG_EARLY:
            early += 1
    denom = float(n) or 1.0
    tot = sum(b1)
    if tot > 0:
        s = 0.0
        for c in b1:
            if c > 0:
                s += c * math.log(c)
        ent = (-s / tot + math.log(tot)) / math.log(100.0)
    else:
        ent = 0.0
    # 稠密 5s 桶（0..max，缺行补 0）——与库内 m2_l1_fingerprint 的修复版一致
    cv = 0.0
    if b5:
        m = max(b5)
        arr = [b5.get(k, 0) for k in range(m + 1)]
        mean = sum(arr) / float(len(arr))
        if mean > 0:
            var = sum((x - mean) ** 2 for x in arr) / float(len(arr))
            cv = math.sqrt(var) / mean
    if duration > 0:
        peak_pct = max(range(100), key=lambda i: w1[i]) if sum(w1) > 0 else 0
    f = {
        'dm_per_min': (n / (duration / 60.0)) if duration > 0 else 0.0,
        'wdm_per_min': (wsum / (duration / 60.0)) if duration > 0 else 0.0,
        'opening_5pct': o5 / denom,
        'ending_5pct': e5 / denom,
        'mid_ratio': mid / denom,
        'early_lag_ratio': (early / denom) if pubdate else 0.0,
        'rhythm_entropy': ent,
        'burst_cv': cv,
        'main_peak_pct': float(peak_pct),
        'peak_cnt': None,                  # 见模块头注「已知差异 3」
        'avg_text_len': tlen / denom,
    }
    shape = {k: f[k] for k in SHAPE_DIMS}
    return f, shape, b1


# ── 编排：拉取 → 计算 → 注入快照 → 复用库内 diagnose() ──────────
_CACHE = {}
_CACHE_TTL = 900          # 15 分钟：同一支视频反复看不必重复采集
_CACHE_MAX = 24


class LiveError(Exception):
    """带用户可读文案的在线诊断失败（前端按 kind 区分展示）。"""

    def __init__(self, kind, msg):
        super().__init__(msg)
        self.kind = kind
        self.msg = msg


def _today():
    return time.strftime('%Y%m%d')


def _pct_map(snap, tname, f, partition_ok):
    """同分区百分位：pct = 100 × (#同类中 <= 本值) / 同类数。

    与快照里的 pct 同算法（已用 156 个样本逐条验证过：全部相符）。
    分区样本不足时退化为全体样本 —— 与 diagnose() 的参照系选择保持一致。
    """
    peers = [v for v in snap['videos']
             if (v.get('tname') == tname) and v.get('f')] if partition_ok else snap['videos']
    out = {}
    for dim in DIMS:
        vals = [v['f'].get(dim) for v in peers if v.get('f', {}).get(dim) is not None]
        mine = f.get(dim)
        if mine is None or len(vals) < 5:
            out[dim] = None
            continue
        le = sum(1 for x in vals if x <= mine)
        out[dim] = round(100.0 * le / len(vals), 1)
    return out


def diagnose_live(bvid):
    """库外 bvid 的在线诊断。返回 (卡片, 在线元信息)。

    已经在库里的视频**不走在线**：库内那份是完整批次口径（含刷量排除），更准。
    """
    import diagnosis as D

    snap = D._load()
    if bvid in snap['by_bvid']:
        return D.diagnose(bvid), {'live': False, 'reason': 'in_library'}

    hit = _CACHE.get(bvid)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1], hit[2]

    view = fetch_view(bvid)
    if not view:
        raise LiveError('not_found', 'B站查不到这个 BV 号（可能输错、或视频已删除/私密）。')
    if not view['duration']:
        raise LiveError('no_duration', '这支视频没有时长信息，无法把弹幕定位到播放进度上。')

    dms, source, segs = fetch_danmaku(view['cid'], view['duration'])
    if source == 'blocked':
        raise LiveError('blocked', 'B站拒绝了这次采集（HTTP 412，触发反爬限流）。'
                                   '等几分钟再试；这不是这支视频的问题。')
    if not dms:
        raise LiveError('pool_expired',
                        '这支视频的弹幕池不可采或已过期（B站只保留约 2-6 个月，'
                        '且大 UP 主的池子常被截断）。没有弹幕就没法算节奏。')

    f, shape, b1 = compute(dms, view['duration'], view['pubdate'])
    tot = sum(b1) or 1.0
    # 101 点：桶 0-99 + 末点恒 0（与 export_v3_snapshot.py 的 range(101) 对齐；
    # 库内 bucket1pct 被 clamp 到 99，所以第 100 点永远是 0）
    curve = [round((b1[i] if i < len(b1) else 0) / tot, 6) for i in range(101)]

    tname = view['tname']
    base_p = snap['baselines'].get(tname) or {}
    partition_ok = (bool(base_p) and not base_p.get('low_support')
                    and base_p.get('n_videos', 0) >= D.MIN_PARTITION)

    rec = {
        'bvid': bvid, 'batch_date': _today(), 'title': view['title'],
        'tname': tname, 'up_name': view['up_name'], 'duration': view['duration'],
        'pubdate': view['pubdate'], 'n_danmaku': len(dms),
        'play': view['play'], 'danmaku_reported': view['danmaku_reported'],
        'f': f, 'shape': shape,
        'pct': _pct_map(snap, tname, f, partition_ok),
        'outcome': {},                     # 后续表现：新视频还没有追踪数据
    }

    # 注入：只浅拷贝会被写的三个结构，其余（基线/全局基准/事件）按引用共享
    shadow = dict(snap)
    shadow['videos'] = list(snap['videos']) + [rec]
    shadow['by_bvid'] = dict(snap['by_bvid'])
    shadow['by_bvid'][bvid] = len(shadow['videos']) - 1
    shadow['curves'] = dict(snap['curves'])
    shadow['curves']['%s|%s' % (bvid, rec['batch_date'])] = curve

    card = D.diagnose(bvid, snap=shadow)
    if not card:
        raise LiveError('internal', '在线诊断内部错误：注入后仍取不到诊断卡。')

    # scope_note 说人话：本次是**拿不到分区标签**，不是分区样本少。
    # diagnose() 只会写「该分区样本不足 N 支」，对在线视频是误报。
    if not tname:
        card['scope_note'] = ('拿不到这支视频的分区标签（B站接口本次没返回 tname，非本系统问题），'
                               '所以参照系用的是**全体样本**而不是同分区 —— 结论方向仍成立，'
                               '但「同类」的范围更宽。')

    meta = {
        'live': True,
        'collected_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_collected': len(dms),
        'segments': segs,
        'source': source,
        'reported': view['danmaku_reported'],
        'partition_in_library': partition_ok,
        'tname': tname or None,
        'tid': view.get('tid'),
        'notes': [
            ('弹幕是**实时快照**（本次采到 %d 条%s），可能被 B站截断 —— 历史上限约 1800 条。'
             % (len(dms), ('，B站报告 %d 条' % view['danmaku_reported'])
                if view['danmaku_reported'] else '')),
            ('采样路径：%s。' % ('XML 全量池 —— 与库内采集同一条路径'
                              if source == 'xml' else
                              'PB 分段兜底（XML 不可用时的退路，**样本可能不全**）')),
            '**不做刷量排除**：库内那 803 支按批次统计刷量，单视频算不出同一口径，所以在线结果可能含刷量弹幕。',
            '除以上两点，曲线与 11 维指纹的算法与库内**逐条对齐**（见 live_diag.py 头注）。',
        ],
    }

    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(min(_CACHE, key=lambda k: _CACHE[k][0]), None)
    _CACHE[bvid] = (time.time(), card, meta)
    return card, meta
