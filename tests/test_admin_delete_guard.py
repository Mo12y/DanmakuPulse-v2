#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''test_admin_delete_guard.py — 「删除账号」的两条红线（2026-10-22）

为什么钉这条：
  删除是不可逆操作。真正会出事的不是"删错一个普通用户"，而是
    ① 把**最后一个管理员**删掉 → 再没人能进管理域（旧系统降级保护就是为此补的）；
    ② 把自己删掉 → 会话还挂着一个不存在的 uid。
  两条都必须由**后端**拒绝，不能只靠前端不显示按钮。

判据（源码级，不需要数据库；与 test_v3_ai_gate.py 同风格）：
  · 路由是 DELETE /api/users/{uid}，且 /api/users 在 auth.ADMIN_PREFIXES 里（越权由中间件拦）
  · 先判「最后一个管理员」再判「不能删自己」（自己正好是最后管理员时，先给真正的风险）
  · 删除成功后写 admin_audit（谁、何时、删了谁，可追溯）
  · 前端：确认框写清用户名；当前账号不显示删除按钮

行为级用例在 tests/accept_auth.sh 的 ⑦b 段（删普通用户 / 删自己 / 删最后管理员 / 越权）。
用法：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python -m pytest tests/ -q
'''
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(*p):
    return io.open(os.path.join(ROOT, *p), encoding='utf-8').read()


def _endpoint_body(app):
    i = app.index("@app.delete('/api/users/{uid}')")
    j = app.index('# ── 收藏', i)
    return app[i:j]


def test_delete_route_exists_and_is_admin_only():
    app = _src('serve', 'app.py')
    body = _endpoint_body(app)
    assert 'HTTPException(404' in body, '用户不存在要 404'
    auth = _src('serve', 'auth.py')
    assert "'/api/users'" in auth and 'ADMIN_PREFIXES' in auth, \
        '/api/users 必须在 ADMIN_PREFIXES 里（DELETE 也走同一前缀门禁）'


def test_last_admin_checked_before_self():
    body = _endpoint_body(_src('serve', 'app.py'))
    i_last = body.index('count_admins() <= 1')
    i_self = body.index("me.get('id')")
    assert i_last < i_self, '必须先判「最后一个管理员」再判「不能删自己」'


def test_delete_is_audited():
    body = _endpoint_body(_src('serve', 'app.py'))
    assert "auth.audit(me, 'delete_user'" in body, '删除成功必须写审计日志'
    auth = _src('serve', 'auth.py')
    assert 'CREATE TABLE IF NOT EXISTS admin_audit' in auth, '缺少 admin_audit 表'
    assert re.search(r'def audit\(', auth), '缺少 auth.audit()'
    assert re.search(r'def delete_user\(', auth), '缺少 auth.delete_user()'


def test_frontend_confirms_and_hides_self_delete():
    h = _src('serve', 'static', 'users.html')
    assert 'data-del' in h, '用户管理页要有删除按钮'
    assert "confirm('删除账号「" in h, '删除前必须弹确认，并写清删的是谁'
    assert '当前账号' in h, '当前登录账号不应显示删除按钮'


def test_accept_auth_covers_delete_cases():
    sh = _src('tests', 'accept_auth.sh')
    for kw in ('删普通用户', '最后一个管理员删自己', '有两个管理员时删自己', '普通用户删账号'):
        assert kw in sh, 'accept_auth 缺少用例：%s' % kw
