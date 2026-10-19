#!/usr/bin/env python3
"""test_highlights_api.py — 精彩时刻端点契约测试（PLAN_HIGHLIGHTS §2.6）

运行（服务需在线，fastapi 默认发布在宿主机 8000）：
  ~/danmaku-venv/bin/python -m pytest tests/test_highlights_api.py -q

覆盖：
  1. 长视频（BV1GjYe6KEGt，25205s）→ slice_suggestions 有值、highlight_positions 为空
  2. 短视频（BV1DnYe6aEjn，154s）→ highlight_positions 有值、slice_suggestions 为空
  3. 字段存在性（data / peaks / critique / benchmark / caveats）
  4. 比例和≈1（从众+中性+批判）、时间落在 [0, duration]
  5. 404 负例（不在快照里的 bvid）
  6. 新旧路由不冲突（/api/highlights/{bvid} 的 batch_date 仍是必填）
服务未起或账号不存在 → 整组 skip（与 test_api_contract.py 同口径）。
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pytest

BASE = os.environ.get('API_BASE', 'http://localhost:8000')
API_USER = os.environ.get('API_USER', 'admin')
API_PASS = os.environ.get('API_PASS', 'danmaku@2026')
LONG_BVID = os.environ.get('HL_LONG_BVID', 'BV1GjYe6KEGt')    # 20260914，25205s
SHORT_BVID = os.environ.get('HL_SHORT_BVID', 'BV1DnYe6aEjn')  # 20260914，154s
COOKIE = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _login():
    body = urllib.parse.urlencode({'username': API_USER, 'password': API_PASS}).encode()
    req = urllib.request.Request(BASE + '/login', data=body, method='POST')
    op = urllib.request.build_opener(_NoRedirect)
    try:
        r = op.open(req, timeout=10)
        raw = r.headers.get('Set-Cookie') or ''
    except urllib.error.HTTPError as e:
        raw = e.headers.get('Set-Cookie') or ''
    except Exception:
        return None
    return raw.split(';')[0] if raw.startswith('ads_session=') else None


def get(path, expect=200):
    req = urllib.request.Request(BASE + path)
    if COOKIE:
        req.add_header('Cookie', COOKIE)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = json.loads(r.read().decode('utf-8'))
            assert r.status == expect, f'{path}: 期望 {expect}，实际 {r.status}'
            return body
    except urllib.error.HTTPError as e:
        assert e.code == expect, f'{path}: 期望 {expect}，实际 {e.code}'
        try:
            return json.loads(e.read().decode('utf-8'))
        except Exception:
            return {}


def _probe_service():
    try:
        with urllib.request.urlopen(BASE + '/api/health', timeout=5) as r:
            if r.status != 200:
                return False
    except Exception:
        return False
    global COOKIE
    COOKIE = _login()
    return COOKIE is not None


def _probe_data():
    try:
        b = get(f'/api/video/{LONG_BVID}/highlights')
        return b.get('code') == 0
    except Exception:
        return False


_SERVICE = _probe_service()
_DATA = _SERVICE and _probe_data()

pytestmark = pytest.mark.skipif(
    not _SERVICE, reason='服务未在线，或账号不存在（先跑 tests/accept_auth.sh 建账号）')
needs_data = pytest.mark.skipif(
    not _DATA, reason='精彩时刻表未导出（先跑 make ads-highlights），或快照不含样例视频')


def _keys(d, *ks):
    for k in ks:
        assert k in d, f'缺字段 {k}: {sorted(d)[:12]}'


def _check_times(d, duration):
    for p in d['peaks']:
        assert 0 <= p['start_sec'] <= p['end_sec'] <= duration, f'峰越界: {p}'
        assert 0 <= p['peak_sec'] <= duration, f'峰中点越界: {p}'
    for s in d['critique']['segments']:
        assert 0 <= s['start_sec'] <= s['end_sec'] <= duration, f'质疑段越界: {s}'
    for s in d['slice_suggestions'] + d['highlight_positions']:
        assert 0 <= s['start_sec'] <= s['end_sec'] <= duration, f'建议越界: {s}'


def _check_shape(d):
    _keys(d, 'bvid', 'batch_date', 'title', 'tname', 'duration', 'n_danmaku', 'is_long',
          'peaks', 'critique', 'feedback', 'benchmark', 'slice_suggestions',
          'highlight_positions', 'caveats')
    assert isinstance(d['caveats'], list) and len(d['caveats']) == 4, \
        f'caveats 应为 4 条：{d["caveats"]}'
    assert all(isinstance(c, str) and c.strip() for c in d['caveats'])
    assert d['is_long'] == (d['duration'] > 1800)

    for p in d['peaks']:
        _keys(p, 'peak_id', 'start_sec', 'end_sec', 'peak_sec', 'score', 'n_repeat',
              'n_rep_texts', 'emotion', 'top_danmaku', 'sentiment', 'in_opening',
              'pctl', 'partition_n')
        _keys(p['emotion'], 'type', 'label', 'note')
        assert p['emotion']['type'] in ('moving', 'funny', 'hype', 'angry', 'neutral')
        _keys(p['sentiment'], 'pos', 'neg', 'neu')
        for t in p['top_danmaku']:
            _keys(t, 'text', 'n')
            assert isinstance(t['text'], str)
        if p['pctl'] is not None:
            assert 0 <= p['pctl'] <= 100

    c = d['critique']
    _keys(c, 'detected', 'n_segments', 'n_critique_danmaku', 'segments')
    assert c['detected'] == bool(c['segments'])
    assert c['n_segments'] == len(c['segments'])
    for s in c['segments']:
        _keys(s, 'seg_id', 'start_sec', 'end_sec', 'score', 'n', 'top_danmaku', 'neg_ratio')

    f = d['feedback']
    _keys(f, 'herd', 'neutral', 'critique', 'n_danmaku', 'n_herd', 'n_neutral', 'n_critique')
    ratios = [f['herd'], f['neutral'], f['critique']]
    if all(r is not None for r in ratios):
        assert abs(sum(ratios) - 1.0) <= 0.02, f'从众+中性+批判 比例和≈1，实际 {sum(ratios)}'
    assert f['n_herd'] + f['n_neutral'] + f['n_critique'] == f['n_danmaku']

    b = d['benchmark']
    _keys(b, 'tname', 'partition_n', 'partition_ok', 'resonance_pctl', 'resonance_z',
          'max_peak_score', 'partition_med', 'n_peers')
    assert isinstance(b['partition_ok'], bool)
    if b['resonance_pctl'] is not None:
        assert 0 <= b['resonance_pctl'] <= 100
    assert b['n_peers'] >= 0

    _check_times(d, d['duration'])


def _check_meta(m, d):
    _keys(m, 'n_peaks', 'n_critique_segments', 'engine_version', 'dict_version',
          'generated_at')
    assert m['n_peaks'] == len(d['peaks'])
    assert m['n_critique_segments'] == d['critique']['n_segments']


@needs_data
def test_long_video_slice_branch():
    b = get(f'/api/video/{LONG_BVID}/highlights')
    assert b['code'] == 0
    d = b['data']
    _check_shape(d)
    _check_meta(b['meta'], d)
    assert d['bvid'] == LONG_BVID
    assert d['is_long'] is True and d['duration'] > 1800
    assert len(d['peaks']) >= 1, '长视频应检出共振峰'
    assert 1 <= len(d['slice_suggestions']) <= 3, '长视频切片建议 top3'
    assert d['highlight_positions'] == [], '长视频不出高光定位'
    ranks = [s['rank'] for s in d['slice_suggestions']]
    assert ranks == list(range(1, len(ranks) + 1)), f'rank 应从 1 连续：{ranks}'
    # 片头噪声政策：非片头峰在前，片头回落峰只能补在后
    flags = [s['opening_only'] for s in d['slice_suggestions']]
    assert flags == sorted(flags), f'opening_only 应 False 在前：{flags}'
    for s in d['slice_suggestions']:
        _keys(s, 'rank', 'start_sec', 'end_sec', 'peak_sec', 'emotion', 'label',
              'score', 'pctl', 'opening_only', 'top_danmaku')
        assert isinstance(s['opening_only'], bool)


@needs_data
def test_short_video_position_branch():
    b = get(f'/api/video/{SHORT_BVID}/highlights')
    assert b['code'] == 0
    d = b['data']
    _check_shape(d)
    _check_meta(b['meta'], d)
    assert d['bvid'] == SHORT_BVID
    assert d['is_long'] is False and d['duration'] <= 1800
    assert len(d['peaks']) >= 1, '短视频应检出共振峰'
    assert 1 <= len(d['highlight_positions']) <= 3, '短视频高光定位 top3'
    assert d['slice_suggestions'] == [], '短视频不出切片建议'
    for s in d['highlight_positions']:
        _keys(s, 'rank', 'sec', 'start_sec', 'end_sec', 'emotion', 'label', 'score',
              'pctl', 'opening_only')
        assert isinstance(s['opening_only'], bool)
        assert 0 <= s['sec'] <= d['duration']
    flags = [s['opening_only'] for s in d['highlight_positions']]
    assert flags == sorted(flags), f'opening_only 应 False 在前：{flags}'


@needs_data
def test_explicit_batch_date_matches_default():
    b = get(f'/api/video/{LONG_BVID}/highlights')
    bd = b['data']['batch_date']
    b2 = get(f'/api/video/{LONG_BVID}/highlights?batch_date={bd}')
    assert b2['code'] == 0 and b2['data']['batch_date'] == bd


@needs_data
def test_unknown_batch_date_404():
    assert get(f'/api/video/{LONG_BVID}/highlights?batch_date=20990101', 404)['detail']


def test_unknown_video_404():
    body = get('/api/video/BV1zzzzzzzzz/highlights', 404)
    assert '快照' in json.dumps(body, ensure_ascii=False), f'404 文案要说明不在快照：{body}'


@needs_data
def test_legacy_highlights_route_kept():
    # 旧端点仍是 batch_date 必填（新路由不得把它挤掉/改签名）
    get('/api/highlights/' + LONG_BVID, 422)


def test_diagnosis_multibatch_no_500():
    """跨批视频（同一 bvid 出现在 ≥2 个批次）的诊断卡不能 500。

    2026-10-19 实测：diagnosis.diagnose() → sensitivity_payload → batch_check →
    diagnose() 未复用 snap 且未跳过 sensitivity，导致 85 支跨批视频无限递归 500。
    修复后本用例守住「库内任意视频都能打开诊断卡」。
    """
    body = get('/api/video/BV1fopY6QEmP/diagnosis')
    d = body['data']
    _keys(d, 'video', 'verdict', 'position', 'curve', 'deviations')
    assert d['video']['bvid'] == 'BV1fopY6QEmP'
