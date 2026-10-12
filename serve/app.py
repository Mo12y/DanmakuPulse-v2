"""FastAPI 只读服务（B0 最小版）——读 PG ads 库，无写操作（07 §8 只读约束）"""
import os

import psycopg
from fastapi import FastAPI, HTTPException, Query

app = FastAPI(title='danmaku-b0', version='b0-min-v1')
DSN = os.environ['ADS_PG_DSN']

METRICS_COLS = ('bvid, title, tname, duration, play, danmaku_count, '
                'peak_count, main_peak_pct, peak_fwhm, cold_pct, '
                'front_ratio, mid_std, end_rebound, cv, engine_version')


def q(sql, params=()):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def q1(sql, params=()):
    rows = q(sql, params)
    return rows[0] if rows else None


@app.get('/api/health')
def health():
    q('SELECT 1')
    return {'code': 0, 'data': 'ok'}


@app.get('/api/anchors')
def anchors():
    """池级对账锚点（B0 验收数字，与 parity_report.json 同源）"""
    row = q1('''SELECT COUNT(*) AS n_rows, SUM(danmaku_count) AS sum_danmaku
                FROM ads_video_metrics''')
    pool = q1('''SELECT COUNT(*) AS pool_n, SUM(danmaku_count) AS pool_sum FROM (
                   SELECT bvid, danmaku_count,
                          ROW_NUMBER() OVER (PARTITION BY bvid ORDER BY batch_date DESC) rn
                   FROM ads_video_metrics) t WHERE rn = 1''')
    return {'code': 0, 'data': {**row, **pool,
                                'expect': {'n_rows': 509, 'sum_danmaku': 672439,
                                           'pool_n': 508, 'pool_sum': 669995}}}


@app.get('/api/batches')
def batches():
    rows = q('''SELECT batch_date, cadence, real_list_date, snapshot_present, status,
                       coverage, list_fetched_at, n_videos, n_danmaku, known_issues
                FROM ads_batch_meta ORDER BY batch_date''')
    return {'code': 0, 'data': rows, 'meta': {'n': len(rows)}}


@app.get('/api/videos')
def videos(batch_date: str = Query(..., min_length=8, max_length=8)):
    rows = q(f'''SELECT {METRICS_COLS} FROM ads_video_metrics
                 WHERE batch_date = %s ORDER BY danmaku_count DESC''', (batch_date,))
    if not rows:
        raise HTTPException(404, f'batch_date={batch_date} 无数据')
    return {'code': 0, 'data': rows, 'meta': {'n': len(rows)}}


@app.get('/api/video/{bvid}')
def video(bvid: str, batch_date: str = Query(..., min_length=8, max_length=8)):
    m = q1(f'''SELECT {METRICS_COLS} FROM ads_video_metrics
               WHERE bvid = %s AND batch_date = %s''', (bvid, batch_date))
    if not m:
        raise HTTPException(404, f'{bvid}@{batch_date} 无数据')
    curve = q('''SELECT bucket, c FROM ads_video_curve
                 WHERE bvid = %s AND batch_date = %s ORDER BY bucket''', (bvid, batch_date))
    m['curve'] = [p['c'] for p in curve]
    return {'code': 0, 'data': m}
