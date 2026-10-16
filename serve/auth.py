# -*- coding: utf-8 -*-
"""账号与会话（沿用旧前端语义；旧实现见 ~/projects/danmaku/web/blueprints/account.py）

为什么不引第三方库：新栈 requirements 只有 fastapi / uvicorn / psycopg。
  · 密码：标准库 hashlib.scrypt —— 与旧库 werkzeug 的 scrypt:N:R:P$salt$hex
    **格式完全兼容**，旧账号哈希可以原样搬进 PG，不需要重置密码。
    ⚠️ 必须传 maxmem：OpenSSL 默认上限 32MB，而 n=32768,r=8 需要约 33.5MB，
       不传会直接 ValueError: memory limit exceeded。
  · 会话：标准库 HMAC-SHA256 签名 cookie（不装 itsdangerous）。

必须照抄的语义（旧仓库测试已固化，见 tests/test_error_pages.py 等）：
  · 角色只有 admin / user，默认 user；不做公开注册
  · 会话三件套 {id, username, role}
  · 未登录：页面 302 /login?next=<path>；/api/* 401 JSON（带 login_required 标志）
  · 非 admin 命中管理前缀：页面 403；/api/* 403 JSON（带 forbidden 标志）
  · next 必须站内绝对路径（挡双斜杠与反斜杠绕过），否则回落 /
  · 登出是 GET /logout
  · 收藏一律用会话里的 uid，绝不接受客户端传 user_id
  · 密码只存哈希；登录失败文案不区分"用户不存在/密码错"
  · 未匹配路径在未登录时仍返回 404（不因门禁多一次跳转）
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time

import psycopg

COOKIE_NAME = 'ads_session'
MAX_AGE = 31 * 24 * 3600                     # 对齐旧系统 31 天
SCRYPT_N, SCRYPT_R, SCRYPT_P = 32768, 8, 1
SALT_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
ROLES = ('admin', 'user')

# 仅管理员可访问的路径前缀（照抄旧 ADMIN_PREFIXES 语义；v3 把"原始细节"技术页也划进来）
ADMIN_PREFIXES = ('/users', '/api/users')   # rhythm.html 已下线（2026-10-16）

# 无需登录的路径（nginx 侧另有静态白名单；这里是 API 侧的，含义一致）
_PUBLIC = ('/login', '/logout', '/api/session', '/api/health', '/docs', '/openapi.json')

_SECRET = None


def public_paths():
    return _PUBLIC


def secret():
    global _SECRET
    if _SECRET is None:
        s = os.environ.get('ADS_SESSION_SECRET')
        if not s:
            s = secrets.token_hex(32)
            print('[auth] WARN: ADS_SESSION_SECRET 未设置，本次随机生成——重启后所有登录失效',
                  flush=True)
        _SECRET = s.encode()
    return _SECRET


def dsn():
    return os.environ['ADS_PG_DSN']


def q(sql, params=(), fetch=True):
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if not fetch:
            conn.commit()
            return None
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def q1(sql, params=()):
    r = q(sql, params)
    return r[0] if r else None


# ── 密码 ────────────────────────────────────────────────────

def hash_password(pw):
    salt = ''.join(secrets.choice(SALT_CHARS) for _ in range(16))
    h = hashlib.scrypt(pw.encode(), salt=salt.encode(), n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                       maxmem=132 * SCRYPT_N * SCRYPT_R * SCRYPT_P).hex()
    return 'scrypt:%d:%d:%d$%s$%s' % (SCRYPT_N, SCRYPT_R, SCRYPT_P, salt, h)


def check_password(pwhash, pw):
    try:
        method, salt, hashval = pwhash.split('$', 2)
        parts = method.split(':')
        if parts[0] != 'scrypt':
            return False
        n, r, p = (int(x) for x in parts[1:])
        calc = hashlib.scrypt(pw.encode(), salt=salt.encode(), n=n, r=r, p=p,
                              maxmem=132 * n * r * p).hex()
        return hmac.compare_digest(calc, hashval)
    except Exception:
        return False


# ── 会话 cookie（标准库签名） ────────────────────────────────

def _b64e(b):
    return base64.urlsafe_b64encode(b).rstrip(b'=')


def _b64d(s):
    return base64.urlsafe_b64decode(s + b'=' * (-len(s) % 4))


def sign_session(payload):
    payload = dict(payload)
    payload['exp'] = int(time.time()) + MAX_AGE
    body = _b64e(json.dumps(payload, separators=(',', ':')).encode())
    sig = _b64e(hmac.new(secret(), body, hashlib.sha256).digest())
    return (body + b'.' + sig).decode()


def read_session(token):
    if not token:
        return None
    try:
        body, sig = token.encode().split(b'.', 1)
        expect = _b64e(hmac.new(secret(), body, hashlib.sha256).digest())
        if not hmac.compare_digest(expect, sig):      # 必须 compare_digest
            return None
        data = json.loads(_b64d(body))
        return data if data.get('exp', 0) > time.time() else None
    except Exception:
        return None


def cookie_header(token, max_age=MAX_AGE):
    return '%s=%s; Path=/; HttpOnly; SameSite=Lax; Max-Age=%d' % (COOKIE_NAME, token, max_age)


# ── 建表 ────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS users (
  id            bigserial PRIMARY KEY,
  username      varchar(64)  NOT NULL UNIQUE,
  password_hash varchar(255) NOT NULL,
  role          varchar(8)   NOT NULL DEFAULT 'user' CHECK (role IN ('admin','user')),
  created_at    timestamptz  NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS favorites (
  id         bigserial PRIMARY KEY,
  user_id    bigint      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  bvid       varchar(32) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT uniq_user_bvid UNIQUE (user_id, bvid)
);
"""


def ensure_tables():
    """幂等建表；库没就绪时不阻塞启动（旧系统同款：失败只 warning）"""
    try:
        with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
            cur.execute(DDL)
            conn.commit()
        return True
    except Exception as e:
        print('[auth] 用户表初始化失败（不阻塞启动）：%s' % e, flush=True)
        return False


# ── 用户 ────────────────────────────────────────────────────

def user_by_name(username):
    return q1('SELECT id, username, password_hash, role FROM users WHERE username = %s', (username,))


def user_by_id(uid):
    return q1('SELECT id, username, role, created_at FROM users WHERE id = %s', (uid,))


def list_users():
    rows = q('SELECT id, username, role, created_at FROM users ORDER BY id')
    for r in rows:
        r['created_at'] = r['created_at'].strftime('%Y-%m-%d %H:%M') if r['created_at'] else ''
    return rows


def create_user(username, password, role='user'):
    role = role if role in ROLES else 'user'
    with psycopg.connect(dsn()) as conn, conn.cursor() as cur:
        cur.execute('INSERT INTO users (username, password_hash, role) VALUES (%s,%s,%s) '
                    'RETURNING id, username, role',
                    (username, hash_password(password), role))
        row = cur.fetchone()
        conn.commit()
        return {'id': row[0], 'username': row[1], 'role': row[2]}


def set_role(uid, role):
    q('UPDATE users SET role = %s WHERE id = %s', (role, uid), fetch=False)


def count_admins():
    r = q1("SELECT COUNT(*) AS n FROM users WHERE role = 'admin'")
    return int(r['n']) if r else 0


def is_last_admin(uid):
    """旧系统没有这条保护（审计已记为缺口）——最后一个管理员降级后就没人能进管理域了"""
    u = user_by_id(uid)
    return bool(u) and u['role'] == 'admin' and count_admins() <= 1

