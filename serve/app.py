"""FastAPI 只读服务（B2.3：3 页数据流 + M4；08 §10.2 统一响应 {code,data,meta}）

端点约定：统一响应 {code, data, meta:{n, engine_version, quality_flag}}；只读；缓存 300s。
B0 验收端点（/api/anchors、/api/videos、/api/video/{bvid}）保留不动——锚点页依赖。
"""
import os
from urllib.parse import parse_qs, quote

import psycopg
import auth
import diagnosis
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

app = FastAPI(title='danmaku-b2', version='b2-v1')
DSN = os.environ['ADS_PG_DSN']
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
LOGIN_HTML = os.path.join(STATIC_DIR, 'login.html')

METRICS_COLS = ('bvid, title, tname, duration, play, danmaku_count, '
                'peak_count, main_peak_pct, peak_fwhm, cold_pct, '
                'front_ratio, mid_std, end_rebound, cv, engine_version')

FP_FEATS = ['dm_per_min', 'wdm_per_min', 'rhythm_entropy', 'burst_cv', 'w_burst_cv',
            'early_lag_ratio', 'color_ratio', 'top_mode_ratio', 'pos_ratio', 'neg_ratio',
            'neu_ratio', 'no_hit_ratio', 'avg_text_len', 'opening_5pct', 'ending_5pct', 'mid_ratio']


@app.on_event('startup')
def _startup():
    auth.ensure_tables()
    print('[app] 用户表就绪' if True else '', flush=True)


def _safe_next(nxt):
    """next 必须是站内绝对路径（照抄旧系统两行判断：挡 //host 与 /\\host 开放重定向）"""
    if not nxt or not nxt.startswith('/') or nxt[1:2] in ('/', '\\'):
        return '/'
    return nxt


@app.middleware('http')
async def _gate(request: Request, call_next):
    """复刻旧 app.py 的 before_request 三分支。
    注意：nginx 只把 /api/ 和 /login 反代进来，静态页面的门禁在 nginx 层（auth_request），
    这里挡的是 API——保证未登录/越权时返回的仍是 JSON，而不是被 nginx 换成 302。"""
    path = request.url.path
    if path in auth.public_paths() or path.startswith('/api/session'):
        return await call_next(request)
    user = auth.read_session(request.cookies.get(auth.COOKIE_NAME))
    request.state.user = user
    if not user:
        if path.startswith('/api/'):
            return JSONResponse({'code': 401, 'error': '未登录，请先登录', 'login_required': True},
                                status_code=401)
        return RedirectResponse('/login?next=' + quote(path), status_code=302)
    if path.startswith(auth.ADMIN_PREFIXES) and user.get('role') != 'admin':
        if path.startswith('/api/'):
            return JSONResponse({'code': 403, 'error': '需要管理员权限', 'forbidden': True},
                                status_code=403)
        return FileResponse(os.path.join(STATIC_DIR, '403.html'), status_code=403)
    return await call_next(request)


def _login_html(next_val, error=None):
    html = open(LOGIN_HTML, encoding='utf-8').read()
    esc = (next_val or '/').replace('&', '&amp;').replace('"', '&quot;').replace('<', '&lt;')
    html = html.replace('<!--NEXT-->', esc)
    html = html.replace('<!--ERR-->', ('<div class="lerr">%s</div>' % error) if error else '')
    return HTMLResponse(html, headers={'Cache-Control': 'no-store'})


@app.get('/login')
def login_page(request: Request, next: str = Query('/')):
    nxt = _safe_next(next)
    if auth.read_session(request.cookies.get(auth.COOKIE_NAME)):
        return RedirectResponse(nxt, status_code=302)
    return _login_html(nxt)


@app.post('/login')
async def login_submit(request: Request):
    """自己解 form-urlencoded：FastAPI 的 Form() 需要 python-multipart，
    为了解析一个登录表单给镜像加依赖不划算（标准库 3 行搞定）。"""
    raw = (await request.body()).decode('utf-8', 'ignore')
    f = {k: v[0] for k, v in parse_qs(raw, keep_blank_values=True).items()}
    nxt = _safe_next(f.get('next') or '/')
    uname = (f.get('username') or '').strip()
    password = f.get('password') or ''
    u = auth.user_by_name(uname) if uname else None
    if u and auth.check_password(u['password_hash'], password):
        token = auth.sign_session({'id': u['id'], 'username': u['username'], 'role': u['role']})
        resp = RedirectResponse(nxt, status_code=302)
        resp.headers['Set-Cookie'] = auth.cookie_header(token)
        return resp
    # 失败：不走 302，直接回 200 + 错误文案（与旧系统一致；且不区分"用户不存在/密码错"）
    return _login_html(nxt, '用户名或密码错误')


@app.get('/logout')
def logout():
    resp = RedirectResponse('/login', status_code=302)
    resp.headers['Set-Cookie'] = auth.cookie_header('', max_age=0)
    return resp


@app.get('/api/session')
def api_session(request: Request, need: str = Query(None)):
    """给 nginx auth_request 用的门禁探针：200=放行 / 401=未登录 / 403=权限不足。
    绝不能缓存，否则登出后还能进。"""
    u = auth.read_session(request.cookies.get(auth.COOKIE_NAME))
    h = {'Cache-Control': 'no-store'}
    if not u:
        return JSONResponse({'code': 401, 'error': '未登录', 'login_required': True},
                            status_code=401, headers=h)
    if need == 'admin' and u.get('role') != 'admin':
        return JSONResponse({'code': 403, 'error': '需要管理员权限', 'forbidden': True},
                            status_code=403, headers=h)
    return JSONResponse({'code': 0, 'data': {'id': u['id'], 'username': u['username'],
                                             'role': u['role']}}, headers=h)


@app.get('/api/me')
def api_me(request: Request, resp: Response):
    u = getattr(request.state, 'user', None)
    return ok(resp, {'username': u['username'], 'role': u['role'],
                     'is_admin': u.get('role') == 'admin'})


# ── 用户管理（仅管理员；门禁在 auth.ADMIN_PREFIXES）──

@app.get('/api/users')
def api_users(resp: Response):
    rows = auth.list_users()
    return ok(resp, rows, meta={'n': len(rows)})


@app.post('/api/users')
async def api_user_create(request: Request, resp: Response):
    d = await request.json()
    username = (d.get('username') or '').strip()
    password = d.get('password') or ''
    role = d.get('role') if d.get('role') in auth.ROLES else 'user'
    if not username or not password:
        raise HTTPException(400, '用户名和密码都不能为空')
    if len(username) > 60 or len(password) < 4:
        raise HTTPException(400, '用户名 ≤60 字符，密码 ≥4 位')
    try:
        u = auth.create_user(username, password, role)
    except Exception as e:
        if 'duplicate key' in str(e) or 'unique' in str(e).lower():
            raise HTTPException(409, '用户名已存在')
        raise HTTPException(500, '创建失败：%s' % e)
    return ok(resp, u)


@app.post('/api/users/{uid}/role')
async def api_user_role(uid: int, request: Request, resp: Response):
    d = await request.json()
    role = d.get('role')
    if role not in auth.ROLES:
        raise HTTPException(400, 'role 只能是 admin 或 user')
    if not auth.user_by_id(uid):
        raise HTTPException(404, '用户不存在')
    # 旧系统没有这条保护（审计已记为缺口）：最后一个管理员降级 = 没人能进管理域
    if role != 'admin' and auth.is_last_admin(uid):
        raise HTTPException(409, '这是最后一个管理员，不能降级——请先指定另一个管理员')
    auth.set_role(uid, role)
    me = getattr(request.state, 'user', None)
    if me and me.get('id') == uid:      # 改自己 → 立刻刷新会话（旧系统同款）
        token = auth.sign_session({'id': uid, 'username': me['username'], 'role': role})
        r = JSONResponse({'code': 0, 'data': {'id': uid, 'role': role}, 'meta': {'n': 1}})
        r.headers['Set-Cookie'] = auth.cookie_header(token)
        return r
    return ok(resp, {'id': uid, 'role': role})


# ── 收藏（登录即可；uid 一律取自会话）──

def _title_of(bvid):
    try:
        for r in diagnosis.index():
            if r['bvid'] == bvid:
                return r.get('title') or '', r.get('tname') or ''
    except Exception:
        pass
    return '', ''


@app.get('/api/favorites')
def api_favorites(request: Request, resp: Response):
    uid = request.state.user['id']
    rows = auth.q('SELECT bvid, created_at FROM favorites WHERE user_id = %s '
                  'ORDER BY created_at DESC', (uid,))
    for r in rows:
        t, tn = _title_of(r['bvid'])
        r['title'], r['tname'] = t, tn
        r['created_at'] = r['created_at'].strftime('%Y-%m-%d %H:%M') if r['created_at'] else ''
    return ok(resp, rows, meta={'n': len(rows)})


@app.post('/api/favorites')
async def api_fav_toggle(request: Request, resp: Response):
    uid = request.state.user['id']
    d = await request.json()
    bvid = (d.get('bvid') or '').strip()
    if not bvid:
        raise HTTPException(400, 'bvid 不能为空')
    hit = auth.q1('SELECT id FROM favorites WHERE user_id = %s AND bvid = %s', (uid, bvid))
    if hit:
        auth.q('DELETE FROM favorites WHERE id = %s', (hit['id'],), fetch=False)
        return ok(resp, {'bvid': bvid, 'favorited': False})
    auth.q('INSERT INTO favorites (user_id, bvid) VALUES (%s,%s)', (uid, bvid), fetch=False)
    return ok(resp, {'bvid': bvid, 'favorited': True})


@app.get('/api/favorites/status')
def api_fav_status(request: Request, resp: Response, bvid: str = Query(..., min_length=4)):
    uid = request.state.user['id']
    hit = auth.q1('SELECT 1 AS x FROM favorites WHERE user_id = %s AND bvid = %s', (uid, bvid))
    return ok(resp, {'bvid': bvid, 'favorited': bool(hit)})


def q(sql, params=()):
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def q1(sql, params=()):
    rows = q(sql, params)
    return rows[0] if rows else None


def ok(resp: Response, data, meta=None):
    resp.headers['Cache-Control'] = 'public, max-age=300'
    return {'code': 0, 'data': data, 'meta': meta or {'n': len(data) if isinstance(data, list) else 1}}


# ─────────────── B0 验收端点（保留） ───────────────

@app.get('/api/health')
def health():
    q('SELECT 1')
    return {'code': 0, 'data': 'ok'}


@app.get('/api/anchors')
def anchors():
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


# ─────────────── B2.3 生产链端点（3 页数据流） ───────────────

@app.get('/api/overview')
def overview(resp: Response):
    """① 运营总览：卡片 + 演化分型 + 聚类概览"""
    latest = q1('SELECT MAX(batch_date) AS bd FROM ads_hotlist_history')
    bd = latest['bd']
    pool = q1('SELECT COUNT(*) AS n_videos, SUM(danmaku_count) AS n_danmaku FROM ('
              ' SELECT bvid, danmaku_count, ROW_NUMBER() OVER ('
              '   PARTITION BY bvid ORDER BY batch_date DESC) rn FROM ads_video_metrics) t'
              ' WHERE rn = 1')
    hl = q1('SELECT COUNT(*) AS n FROM ads_hotlist_history WHERE batch_date = %s', (bd,))
    clus = q1('SELECT k, silhouette FROM ads_video_fingerprint '
              'WHERE cluster_id IS NOT NULL LIMIT 1')
    sizes = q('SELECT cluster_id, COUNT(*) AS n FROM ads_video_fingerprint '
              'WHERE cluster_id IS NOT NULL GROUP BY cluster_id ORDER BY cluster_id')
    evo = q('SELECT cluster_type, COUNT(*) AS n FROM ads_evolution_feats '
            'GROUP BY cluster_type ORDER BY n DESC')
    comp = q('''SELECT batch_date, tname, COUNT(*) AS n, SUM(play) AS play_sum
                FROM ads_hotlist_history GROUP BY batch_date, tname
                ORDER BY batch_date, n DESC''')
    return ok(resp, {
        'latest_batch': bd, 'hotlist_size': hl['n'],
        'n_videos_pool': pool['n_videos'], 'n_danmaku_pool': int(pool['n_danmaku'] or 0),
        'cluster': {'k': clus['k'] if clus else None,
                    'silhouette': clus['silhouette'] if clus else None,
                    'sizes': sizes},
        'evo_types': evo, 'composition': comp,
    }, meta={'n_batches': len({r['batch_date'] for r in comp}), 'engine_version': 'b2-v1',
             'quality_flag': 'ok'})


@app.get('/api/hotlist')
def hotlist(resp: Response, batch_date: str = Query(None, min_length=8, max_length=8)):
    if batch_date is None:
        batch_date = q1('SELECT MAX(batch_date) AS bd FROM ads_hotlist_history')['bd']
    rows = q('''SELECT rank_proxy, bvid, title, tname, up_name, play, danmaku_reported,
                       source, cadence, prev_batch_date, rank_delta, play_delta, danmaku_delta
                FROM ads_hotlist_history WHERE batch_date = %s ORDER BY rank_proxy''',
             (str(batch_date),))
    if not rows:
        raise HTTPException(404, f'batch_date={batch_date} 无榜单数据')
    return ok(resp, rows, meta={'n': len(rows), 'batch_date': str(batch_date)})


@app.get('/api/partitions')
def partitions(resp: Response):
    # tname IS NULL 行 = 未知分区（部分视频无分区名）——不可导航，不进选择器
    rows = q('''SELECT tname, n_videos, low_support FROM ads_partition_fingerprint
                WHERE tname IS NOT NULL ORDER BY n_videos DESC''')
    return ok(resp, rows)


@app.get('/api/partition/{tname}')
def partition(tname: str, resp: Response):
    m = q1('SELECT * FROM ads_partition_fingerprint WHERE tname = %s', (tname,))
    if not m:
        raise HTTPException(404, f'分区 {tname} 无基准数据')
    m['feat_med'] = {c: m.get(f'{c}_med') for c in FP_FEATS}
    m['feat_mad'] = {c: m.get(f'{c}_mad') for c in FP_FEATS}
    return ok(resp, m)


@app.get('/api/partition/{tname}/videos')
def partition_videos(tname: str, resp: Response,
                     batch_date: str = Query(None, min_length=8, max_length=8)):
    cond = 'AND f.batch_date = %s' if batch_date else ''
    params = (tname, str(batch_date)) if batch_date else (tname,)
    rows = q(f'''SELECT f.bvid, f.batch_date, f.n_danmaku, f.dm_per_min, f.rhythm_entropy,
                        f.cluster_id, f.evo_type, f.silhouette,
                        h.title, h.play, h.up_name
                 FROM ads_video_fingerprint f
                 LEFT JOIN ads_hotlist_history h
                   ON f.bvid = h.bvid AND f.batch_date = h.batch_date
                 WHERE f.tname = %s {cond}
                 ORDER BY f.n_danmaku DESC''', params)
    dist = {}
    for r in rows:
        t = r['evo_type'] or '未分型'
        dist[t] = dist.get(t, 0) + 1
    return ok(resp, rows, meta={'n': len(rows), 'type_dist': dist})


@app.get('/api/rhythm/{bvid}')
def rhythm(bvid: str, resp: Response,
           batch_date: str = Query(..., min_length=8, max_length=8),
           granularity: str = Query('100')):
    if granularity == '5s':
        pts = q('''SELECT bucket5s, cnt, wcnt, cal_cnt FROM ads_video_curve_5s
                   WHERE bvid = %s AND batch_date = %s ORDER BY bucket5s''',
                (bvid, str(batch_date)))
    else:
        pts = q('''SELECT bucket, cnt, wcnt, cal_cnt, sent_avg, event_id
                   FROM ads_video_rhythm_curve
                   WHERE bvid = %s AND batch_date = %s ORDER BY bucket''',
                (bvid, str(batch_date)))
    if not pts:
        raise HTTPException(404, f'{bvid}@{batch_date} 无曲线数据')
    return ok(resp, {'bvid': bvid, 'batch_date': str(batch_date), 'points': pts},
              meta={'n': len(pts), 'granularity': granularity})


@app.get('/api/highlights/{bvid}')
def highlights(bvid: str, resp: Response,
               batch_date: str = Query(..., min_length=8, max_length=8)):
    rows = q('''SELECT event_id, start_sec, end_sec, score_fused, n_methods,
                       sync_score, sync_p, n_window, delta_hat_sec, calibrated_on,
                       summary, top_danmaku, n_pos, n_neg, n_neu, n_obj
                FROM ads_video_highlights
                WHERE bvid = %s AND batch_date = %s ORDER BY event_id''',
             (bvid, str(batch_date)))
    if not rows:
        raise HTTPException(404, f'{bvid}@{batch_date} 无事件数据')
    return ok(resp, rows, meta={'n': len(rows)})


@app.get('/api/fingerprint/{bvid}')
def fingerprint(bvid: str, resp: Response,
                batch_date: str = Query(..., min_length=8, max_length=8)):
    cols = ', '.join(FP_FEATS)
    zcols = ', '.join(f'{c}_z_part' for c in FP_FEATS)
    m = q1(f'''SELECT bvid, batch_date, tname, duration, n_danmaku, {cols}, {zcols},
                      cluster_id, cluster_method, silhouette, cluster_id_kshape,
                      ari_kmeans_kshape, evo_type
               FROM ads_video_fingerprint
               WHERE bvid = %s AND batch_date = %s''', (bvid, str(batch_date)))
    if not m:
        raise HTTPException(404, f'{bvid}@{batch_date} 无指纹数据')
    base = q1('''SELECT tname, n_videos, low_support, curve_med FROM ads_partition_fingerprint
                 WHERE tname = %s''', (m['tname'],))
    m['baseline'] = base
    return ok(resp, m)


@app.get('/api/dualaxis/{bvid}')
def dualaxis(bvid: str, resp: Response):
    rows = q('''SELECT offset_bin, lag_bin, cnt FROM ads_dual_axis
                WHERE bvid = %s ORDER BY offset_bin, lag_bin''', (bvid,))
    if not rows:
        raise HTTPException(404, f'{bvid} 无双时间轴数据')
    return ok(resp, rows)


# ─────────────── v3 对标诊断（33 号四视图；数据源 = serve/v3data 只读快照） ───────────────

@app.get('/api/diagnosis/meta')
def diagnosis_meta(resp: Response):
    '''快照元信息 + 词典版本（前端页脚展示口径）'''
    return ok(resp, diagnosis.snapshot_meta())


@app.get('/api/diagnosis/index')
def diagnosis_index(resp: Response):
    '''选片索引（803 支，按分区 + 互动热度排序）'''
    rows = diagnosis.index()
    return ok(resp, rows, meta={'n': len(rows)})


@app.get('/api/video/{bvid}/diagnosis')
def video_diagnosis(bvid: str, resp: Response,
                    batch_date: str = Query(None, min_length=8, max_length=8)):
    '''一张诊断卡：定位 / 偏离 / 相似对标 / 一句话结论 + 行动（33 号 §3）'''
    d = diagnosis.diagnose(bvid, str(batch_date) if batch_date else None)
    if not d:
        raise HTTPException(404, f'{bvid} 不在节奏诊断快照里（快照覆盖 803 支视频）')
    return ok(resp, d, meta={'n': 1, 'engine_version': 'b2-v1', 'quality_flag': 'ok',
                             'dict_version': d['tech']['dict_version']})


@app.get('/api/v3/overview')
def v3_overview(resp: Response):
    '''市场总览：池子规模 + 各分区性格 + 全市场最不典型的视频'''
    return ok(resp, diagnosis.overview_view())


@app.get('/api/v3/partitions')
def v3_partitions(resp: Response):
    rows = diagnosis.partition_list()
    return ok(resp, rows, meta={'n': len(rows)})


@app.get('/api/v3/partition/{tname}')
def v3_partition(tname: str, resp: Response):
    '''分区画像：这个分区的观众习惯 + 标杆 / 最特别'''
    d = diagnosis.partition_view(tname)
    if not d:
        raise HTTPException(404, f'分区 {tname} 无基准数据')
    return ok(resp, d, meta={'n': 1})


@app.get('/api/lifecycle/{bvid}')
def lifecycle(bvid: str, resp: Response):
    """生命周期（M4；来自 dws_lifecycle → ads 未建表则回退 404）"""
    r = q1('SELECT * FROM ads_lifecycle WHERE bvid = %s', (bvid,))
    if not r:
        raise HTTPException(404, f'{bvid} 无生命周期数据')
    return ok(resp, r)
