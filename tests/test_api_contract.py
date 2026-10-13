#!/usr/bin/env python3
"""test_api_contract.py — 服务层接口契约测试（B2.3 验收「接口契约测试过」）

运行（服务需在线）：
  ~/danmaku-venv/bin/python -m pytest tests/test_api_contract.py -q
覆盖：12 个端点的 200 状态、统一响应结构 {code,data,meta}、关键字段存在性、404 负例。
服务未起时整组 skip（不阻塞无服务场景的套件）。
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pytest

BASE = os.environ.get('API_BASE', 'http://localhost:8000')


def get(path, expect=200):
    url = BASE + path
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
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
            return r.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not probe(), reason='服务未在线（先起 fastapi）')


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
    assert d['cluster']['k'] == 4
    assert sum(t['n'] for t in d['evo_types']) == 124


def test_hotlist_contract_and_404():
    b = get('/api/hotlist')
    d = b['data']
    assert len(d) == 200
    _keys(d[0], 'rank_proxy', 'bvid', 'title', 'tname', 'play', 'cadence')
    assert d[0]['cadence'] == 'weekly_legacy'
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
    assert len(parts) == 71, f'命名分区应为 71（全表 72 含 1 行 tname=NULL 未知分区）: {len(parts)}'
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
