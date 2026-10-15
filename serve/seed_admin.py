#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""首个管理员种子（容器内执行）

旧系统把默认口令硬编码在脚本里（审计已记为问题：scripts/init_users.py:28
DEFAULT_PASSWORD = 'danmaku2026'）。这里不照抄：
  优先读环境变量 ADS_ADMIN_PASSWORD；没有就**随机生成并打印一次**，让管理员自己记下来。

用法：
  docker exec fastapi python seed_admin.py                      # 用户名 admin，随机口令
  docker exec -e ADS_ADMIN_PASSWORD=xxx fastapi python seed_admin.py
  docker exec fastapi python seed_admin.py --username alice --role user
"""
import argparse
import secrets
import sys

import auth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--username', default='admin')
    ap.add_argument('--role', default='admin', choices=list(auth.ROLES))
    ap.add_argument('--password', default=None,
                    help='不传则读 ADS_ADMIN_PASSWORD，再没有就随机生成并打印')
    args = ap.parse_args()

    if not auth.ensure_tables():
        print('建表失败：数据库没就绪', file=sys.stderr)
        return 1

    import os
    pw = args.password or os.environ.get('ADS_ADMIN_PASSWORD')
    generated = False
    if not pw:
        pw = secrets.token_urlsafe(9)
        generated = True

    exist = auth.user_by_name(args.username)
    if exist:
        auth.q('UPDATE users SET password_hash = %s, role = %s WHERE id = %s',
               (auth.hash_password(pw), args.role, exist['id']), fetch=False)
        print('已重置账号：%s（角色 %s）' % (args.username, args.role))
    else:
        auth.create_user(args.username, pw, args.role)
        print('已创建账号：%s（角色 %s）' % (args.username, args.role))
    if generated:
        print('随机口令：%s   ← 只显示这一次，请立即记下' % pw)
    print('现有账号：', [(u['id'], u['username'], u['role']) for u in auth.list_users()])
    return 0


if __name__ == '__main__':
    sys.exit(main())

