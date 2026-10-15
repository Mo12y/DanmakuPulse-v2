#!/usr/bin/env python3
"""test_api_contract.py — 服务层接口契约测试（B2.3 验收「接口契约测试过」）

运行（服务需在线）：
  ~/danmaku-venv/bin/python -m pytest tests/test_api_contract.py -q
覆盖：端点的 200 状态、统一响应结构 {code,data,meta}、关键字段存在性、404 负例。

⚠️ 2026-10-15 起内部 /api/* 需要登录（与旧前端一致：只有 /api/open/* 是公开的）。
   本文件因此先登录取会话 cookie；账号取环境变量 API_USER / API_PASS
   （默认 admin / danmaku@2026，可用 tests/accept_auth.sh 的同一套账号）。
   服务未起、或账号不存在 → 整组 skip（不阻塞无服务场景的套件）。
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
COOKIE = None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _login():
    """POST /login 拿会话 cookie（不让 urllib 跟 302，否则读不到 Set-Cookie）"""
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
        with urllib.request.urlopen(req, timeout=10) as r:
            body = json.loads(r.read().decode('utf-8'))
            assert r.status == expect, f'{path}: status {r.status}'
            return body
    except urllib.error.HTTPError as e:
        if expect == 404:
            return e.code
        raise


def probe():
    try:
        with urllib.request.urlopen(BASE + '/api/health', timeout=5) as r:
            if r.status != 200:
                return False
    except Exception:
        return False
    global COOKIE
    COOKIE = _login()
    return COOKIE is not None        # 登不进去也 skip（账号没建）


pytestmark = pytest.mark.skipif(
    not probe(), reason='服务未在线，或账号不存在（先跑 tests/accept_auth.sh 建账号）')


def _keys(d, *ks):
    for k in ks:
        assert k in d, f'缺字段 {k}: {list(d)[:10]}'


def test_health_and_unified_shape():
    b = get('/api/health')
    _keys(b, 'code', 'data')
    assert b['code'] == 0


def test_batches_and_anchors():
    b = get('/api/batches')
    assert b['code'] == 0 and isinstance(b['data'], list) and len(b['data']) >= 4
    _keys(b['data'][0], 'batch_date', 'cadence')
    a = get('/api/anchors')
    _keys(a['data'], 'n_rows', 'pool_n', 'expect')


def test_overview_contract():
    b = get('/api/overview')
    d = b['data']
    _keys(d, 'latest_batch', 'hotlist_size', 'n_videos_pool', 'cluster', 'evo_types', 'composition')
    assert d['n_videos_pool'] == 508
    # 2026-10-13：k 随数据演化（6 批 + burst 修复后 silhouette 选 k=3），改结构断言
    assert d['cluster']['k'] >= 2
    # 2026-10-13：演化样本随数据增长（B2 时 124，现 142），改下界
    assert sum(t['n'] for t in d['evo_types']) >= 120


def test_hotlist_contract_and_404():
    b = get('/api/hotlist')
    d = b['data']
    # 2026-10-13：hotlist 行数随新批次增长（当前 300），改下界断言
    assert len(d) >= 200
    _keys(d[0], 'rank_proxy', 'bvid', 'title', 'tname', 'play', 'cadence')
    # 2026-10-13：新批次引入 daily 节奏（hotlist[0] 为最新批）
    assert d[0]['cadence'] in ('daily', 'weekly_legacy')
    assert d[0]['rank_delta'] is None          # 冻结决策：legacy 不做跨批差分
    assert get('/api/hotlist?batch_date=20990101', 404) == 404


def test_rhythm_100_and_5s():
    b = get('/api/hotlist')
    bvid = b['data'][0]['bvid']
    bd = b['meta']['batch_date']
    c = get(f'/api/rhythm/{bvid}?batch_date={bd}')
    _keys(c['data'], 'points')
    _keys(c['data']['points'][0], 'bucket', 'cnt', 'wcnt', 'cal_cnt', 'sent_avg', 'event_id')
    assert c['meta']['granularity'] == '100'
    c5 = get(f'/api/rhythm/{bvid}?batch_date={bd}&granularity=5s')
    _keys(c5['data']['points'][0], 'bucket5s', 'cnt', 'wcnt', 'cal_cnt')
    assert c5['meta']['granularity'] == '5s'


def test_fingerprint_contract():
    b = get('/api/hotlist')
    bvid = b['data'][0]['bvid']
    bd = b['meta']['batch_date']
    f = get(f'/api/fingerprint/{bvid}?batch_date={bd}')['data']
    _keys(f, 'rhythm_entropy', 'burst_cv', 'opening_5pct', 'cluster_id', 'cluster_method')
    assert f['cluster_method'] == 'kmeans'


def test_partitions_and_detail():
    parts = get('/api/partitions')['data']
    # 2026-10-13：分区数随新批次增长（当前 76 命名 + 1 未知），改下界断言
    assert len(parts) >= 70, f'命名分区 ≥70（当前 {len(parts)}）'
    assert all(p['tname'] for p in parts)
    tname = parts[0]['tname']
    p = get(f'/api/partition/{urllib.parse.quote(tname)}')['data']
    _keys(p, 'curve_med', 'curve_p25', 'curve_p75', 'feat_med', 'feat_mad', 'n_videos')
    assert len(p['curve_med']) == 100
    v = get(f'/api/partition/{urllib.parse.quote(tname)}/videos')
    assert v['meta']['n'] >= 1 and 'type_dist' in v['meta']
    assert get('/api/partition/' + urllib.parse.quote('不存在的分区XYZ'), 404) == 404


def test_highlights_and_dualaxis():
    # 找一个有事件的视频（从 highlights 可 404 容忍：至少抽查前 5 名榜单）
    b = get('/api/hotlist')
    bd = b['meta']['batch_date']
    hit = None
    for r in b['data'][:5]:
        try:
            h = get(f"/api/highlights/{r['bvid']}?batch_date={bd}")['data']
            hit = (r['bvid'], h)
            break
        except urllib.error.HTTPError:
            continue
    assert hit, '前 5 名均无事件？异常'
    bvid, h = hit
    _keys(h[0], 'event_id', 'start_sec', 'end_sec', 'score_fused', 'sync_p',
          'top_danmaku', 'n_pos', 'n_neg', 'n_neu', 'n_obj', 'summary')
    da = get(f'/api/dualaxis/{bvid}')['data']
    _keys(da[0], 'offset_bin', 'lag_bin', 'cnt')
