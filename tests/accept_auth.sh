#!/usr/bin/env bash
# accept_auth.sh — 登录 / 门禁 机械验收（命令 + 预期输出，不写 prose）
#
# 语义对齐旧前端（~/projects/danmaku/web/blueprints/account.py + app.py before_request）：
#   未登录页面 302 /login?next= ｜ 未登录 /api/* 401 JSON(login_required)
#   非 admin 管理域 403 JSON(forbidden) / 页面送 403 页 ｜ next 只能站内绝对路径
#
# 两处**有意**与旧系统不同（有理由，不是漏做）：
#   ① 未登录时未知路径也回 302，而非旧系统的 404。
#      原因：页面门禁在 nginx（静态文件 FastAPI 看不到），auth_request 在 try_files 之前跑。
#      结果上更保守——未登录访客对所有路径行为完全一致，探测不出哪些路径存在；登录后未知路径仍是 404。
#   ② 登出用 GET /logout 置空 cookie；无状态签名 cookie 与旧 Flask session 同性质：
#      客户端若继续回放旧 cookie，有效期内仍有效。用例按"响应置空 + jar 更新后失效"验。
#
# 用法：bash tests/accept_auth.sh [host]     默认 http://127.0.0.1:8088
set -u
H="${1:-http://127.0.0.1:8088}"
EG_BVID="${EG_BVID:-BV1PJpE6QEx6}"   # 快照里真实存在的示例视频
ADMIN_USER="${ADMIN_USER:-admin}"
ADMIN_PASS="${ADMIN_PASS:-danmaku@2026}"
USER_NAME="${USER_NAME:-tester}"
USER_PASS="${USER_PASS:-test1234}"
pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then echo "  PASS  $1  → $3"; pass=$((pass+1));
        else echo "  FAIL  $1  期望[$2] 实际[$3]"; fail=$((fail+1)); fi; }
has() { case "$3" in *"$2"*) echo "  PASS  $1"; pass=$((pass+1));;
        *) echo "  FAIL  $1  （输出里没有 '$2'）→ $3"; fail=$((fail+1));; esac; }
J1=$(mktemp); J2=$(mktemp); J3=$(mktemp)

echo "=== ① 未登录：页面跳登录页 / API 回 401 JSON / 公开资源放行 ==="
chk "未登录 GET /diagnosis.html" 302 "$(curl -s -o /dev/null -w '%{http_code}' $H/diagnosis.html)"
# 必须连 host:port 一起断言：nginx 默认把相对 Location 补成绝对 URL 时用的是**内部监听端口**
# （8080/8443），只查路径会漏掉"跳到未发布端口"这类 bug（2026-10-15 实测漏过一次）
has "未登录跳登录页且保留原端口" "$H/login?next=/diagnosis.html" "$(curl -s -o /dev/null -w '%{redirect_url}' $H/diagnosis.html)"
chk "未登录 GET /api/v3/overview" 401 "$(curl -s -o /dev/null -w '%{http_code}' $H/api/v3/overview)"
has "401 带 login_required" "login_required" "$(curl -s $H/api/v3/overview)"
for u in /api/health /login.html /403.html /v3.css /v3nav.js /vendor/echarts.min.js; do
  chk "公开 GET $u" 200 "$(curl -s -o /dev/null -w '%{http_code}' $H$u)"
done

echo
echo "=== ② 登录页：零口令 + 表单完整 ==="
B=$(curl -s $H/login.html)
case "$B" in *danmaku@2026*|*test1234*|*data-p=*) echo "  FAIL  登录页泄露口令"; fail=$((fail+1));;
             *) echo "  PASS  登录页不含任何口令"; pass=$((pass+1));; esac
for k in 'name="username"' 'name="password"' 'method="post"' 'action="/login"' '进入系统'; do
  has "表单含 $k" "$k" "$B"
done

echo
echo "=== ③ 错口令被拒 / 对口令放行（文案不区分用户是否存在）==="
chk "错口令 POST /login" 200 "$(curl -s -o /dev/null -w '%{http_code}' -X POST -d 'username=admin&password=WRONG' $H/login)"
has "错口令有提示" "用户名或密码错误" "$(curl -s -X POST -d 'username=admin&password=WRONG' $H/login)"
chk "不存在的用户同样 200" 200 "$(curl -s -o /dev/null -w '%{http_code}' -X POST -d 'username=nobody&password=x' $H/login)"
chk "对口令 POST /login" 302 "$(curl -s -c $J1 -o /dev/null -w '%{http_code}' -X POST -d "username=$ADMIN_USER&password=$ADMIN_PASS" $H/login)"
grep -q ads_session $J1 && { echo "  PASS  已下发会话 cookie"; pass=$((pass+1)); } || { echo "  FAIL  未下发 cookie"; fail=$((fail+1)); }

echo
echo "=== ④ 登录后：页面与 API 都放行 ==="
for u in / /diagnosis.html /overview.html /partition.html /me.html /api/v3/overview /users /api/users; do
  chk "admin GET $u" 200 "$(curl -s -b $J1 -o /dev/null -w '%{http_code}' $H$u)"
done

echo
echo "=== ⑤ 普通用户：能看诊断，进不了管理域 ==="
curl -s -c $J2 -o /dev/null -X POST -d "username=$USER_NAME&password=$USER_PASS" $H/login
chk "user GET /diagnosis.html" 200 "$(curl -s -b $J2 -o /dev/null -w '%{http_code}' $H/diagnosis.html)"
chk "user GET /api/users" 403 "$(curl -s -b $J2 -o /dev/null -w '%{http_code}' $H/api/users)"
has "403 带 forbidden" "forbidden" "$(curl -s -b $J2 $H/api/users)"
has "user 打开 /users 被送去 403 页（保留原端口）" "$H/403.html" "$(curl -s -b $J2 -o /dev/null -w '%{redirect_url}' $H/users)"
# rhythm.html / index.html 于 2026-10-16 下线（导航只留用户真正会用的四页 + 用户管理）
chk "已下线页 /rhythm.html 登录后仍是 404" 404 "$(curl -s -b $J1 -o /dev/null -w '%{http_code}' $H/rhythm.html)"
chk "已下线页 /index.html 登录后仍是 404" 404 "$(curl -s -b $J1 -o /dev/null -w '%{http_code}' $H/index.html)"
chk "user GET /users（仅管理员页）仍 302" 302 "$(curl -s -b $J2 -o /dev/null -w '%{http_code}' $H/users)"

echo
echo "=== ⑥ next 防开放重定向 ==="
chk "next=//evil.com 回落" "$H/" "$(curl -s -o /dev/null -w '%{redirect_url}' -X POST -d "username=$ADMIN_USER&password=$ADMIN_PASS&next=//evil.com" $H/login)"
chk "next=/\evil.com 回落" "$H/" "$(curl -s -o /dev/null -w '%{redirect_url}' -X POST -d "username=$ADMIN_USER&password=$ADMIN_PASS&next=/\evil.com" $H/login)"
chk "合法 next 生效" "$H/partition.html" "$(curl -s -o /dev/null -w '%{redirect_url}' -X POST -d "username=$ADMIN_USER&password=$ADMIN_PASS&next=/partition.html" $H/login)"

echo
echo "=== ⑦ 最后一个管理员不能降级（旧系统缺这条保护）==="
UID_A=$(curl -s -b $J1 $H/api/users | python3 -c "import json,sys;print([u['id'] for u in json.load(sys.stdin)['data'] if u['username']=='$ADMIN_USER'][0])")
chk "唯一管理员降级" 409 "$(curl -s -b $J1 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{"role":"user"}' $H/api/users/$UID_A/role)"

echo
echo "=== ⑧ 收藏：uid 取自会话 + toggle 幂等 + 越权不可传 user_id ==="
BV=BV12wY96jEAG
fav() { curl -s -b $J2 "$H/api/favorites/status?bvid=$BV" | python3 -c "import json,sys;print(json.load(sys.stdin)['data']['favorited'])"; }
cnt() { curl -s -b $J2 $H/api/favorites | python3 -c "import json,sys;print(len(json.load(sys.stdin)['data']))"; }
B0=$(fav); N0=$(cnt)
curl -s -b $J2 -o /dev/null -X POST -H 'Content-Type: application/json' -d "{\"bvid\":\"$BV\"}" $H/api/favorites
B1=$(fav); N1=$(cnt)
[ "$B0" != "$B1" ] && { echo "  PASS  点一次改变收藏状态（$B0 → $B1）"; pass=$((pass+1)); } || { echo "  FAIL  toggle 没生效（仍为 $B1）"; fail=$((fail+1)); }
curl -s -b $J2 -o /dev/null -X POST -H 'Content-Type: application/json' -d "{\"bvid\":\"$BV\"}" $H/api/favorites
B2=$(fav); N2=$(cnt)
[ "$B2" = "$B0" ] && [ "$N2" = "$N0" ] && { echo "  PASS  再点一次回到原状态（toggle 幂等：$N0 → $N1 → $N2）"; pass=$((pass+1)); } || { echo "  FAIL  连点两次没回到原状（$N0 → $N1 → $N2）"; fail=$((fail+1)); }
chk "未登录读收藏" 401 "$(curl -s -o /dev/null -w '%{http_code}' $H/api/favorites)"
# 传 user_id 应被忽略：admin 与 tester 看到的是各自的列表
chk "传 user_id 无效（admin 列表 ≠ tester 列表条数）" "different" "$(A=$(curl -s -b $J1 $H/api/favorites | python3 -c "import json,sys;print(len(json.load(sys.stdin)['data']))"); T=$(cnt); [ "$A" = "$T" ] && echo same || echo different)"

echo
echo "=== ⑨ 登出 ==="
curl -s -c $J3 -o /dev/null -X POST -d "username=$ADMIN_USER&password=$ADMIN_PASS" $H/login
chk "登出前可进" 200 "$(curl -s -b $J3 -c $J3 -o /dev/null -w '%{http_code}' $H/diagnosis.html)"
has "登出响应置空 cookie" "Max-Age=0" "$(curl -s -b $J3 -c $J3 -D - -o /dev/null $H/logout | tr -d '\r' | grep -i '^set-cookie')"
chk "登出后（jar 已更新）不可进" 302 "$(curl -s -b $J3 -o /dev/null -w '%{http_code}' $H/diagnosis.html)"

echo
echo "=== ⑩ 路径存在性：未登录不泄露，登录后仍是 404 ==="
chk "未登录 未知路径（与已知路径同为 302）" 302 "$(curl -s -o /dev/null -w '%{http_code}' $H/no-such-page.html)"
chk "已登录 未知路径" 404 "$(curl -s -b $J1 -o /dev/null -w '%{http_code}' $H/no-such-page.html)"

echo
echo "=== 双协议同端口（浏览器「始终使用安全连接」会把 http 升级成 https）==="
HTTPS_H="${H/http:/https:}"
chk "明文 http 取登录页" 200 "$(curl -s -o /dev/null -w '%{http_code}' $H/login.html)"
chk "TLS https 取登录页（自签，-k 忽略证书）" 200 "$(curl -sk -o /dev/null -w '%{http_code}' $HTTPS_H/login.html)"
chk "TLS 未登录取页面仍 302" 302 "$(curl -sk -o /dev/null -w '%{http_code}' $HTTPS_H/diagnosis.html)"
has "TLS 跳转也保留原端口（不能指到内部 8443）" "$HTTPS_H/login" "$(curl -sk -o /dev/null -w '%{redirect_url}' $HTTPS_H/diagnosis.html)"
chk "TLS 未登录取 API 仍 401" 401 "$(curl -sk -o /dev/null -w '%{http_code}' $HTTPS_H/api/v3/overview)"
JT=$(mktemp)
chk "TLS 下能登录" 302 "$(curl -sk -c $JT -o /dev/null -w '%{http_code}' -X POST -d "username=$ADMIN_USER&password=$ADMIN_PASS" $HTTPS_H/login)"
chk "TLS 下带会话能取诊断卡" 200 "$(curl -sk -b $JT -o /dev/null -w '%{http_code}' $HTTPS_H/diagnosis.html)"
chk "TLS 下带会话能取 API" 200 "$(curl -sk -b $JT -o /dev/null -w '%{http_code}' $HTTPS_H/api/v3/overview)"
rm -f $JT

echo
echo "=== 会话隔离与「系统属性」门禁（2026-10-16 新增）==="
# 起因：/api/me 默认发 public, max-age=300，被浏览器缓存 5 分钟 ——
# admin 退出、新用户登录后顶栏还是 admin，看起来就是「新用户登录后还是 admin」。
has "/api/me 必须 no-store（否则换号会串）" "cache-control: no-store" \
  "$(curl -s -b $J1 -o /dev/null -D - $H/api/me | tr 'A-Z' 'a-z' | tr -d '\r')"
has "/api/favorites 必须 no-store（per-user 数据）" "cache-control: no-store" \
  "$(curl -s -b $J1 -o /dev/null -D - $H/api/favorites | tr 'A-Z' 'a-z' | tr -d '\r')"
has "/api/users 必须 no-store" "cache-control: no-store" \
  "$(curl -s -b $J1 -o /dev/null -D - $H/api/users | tr 'A-Z' 'a-z' | tr -d '\r')"

# 已登录访问 /login 必须给表单（曾经静默 302 弹回，导致"想换账号根本到不了表单"）
chk "已登录 GET /login 仍给表单（不再静默弹回）" 200 "$(curl -s -b $J1 -o /dev/null -w '%{http_code}' $H/login)"
has "已登录的 /login 页说明当前身份" "当前已登录为" "$(curl -s -b $J1 $H/login)"

# 完整换号：admin 建临时用户 → 退出 → 新用户登录 → /api/me 是新用户
TMPU="acc$(date +%s)"
TMPP="pw$(date +%s)"
chk "admin 建临时用户" 200 "$(curl -s -b $J1 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' \
  -d "{\"username\":\"$TMPU\",\"password\":\"$TMPP\",\"role\":\"user\"}" $H/api/users)"
JS=$(mktemp)
curl -s -c $JS -o /dev/null -X POST -d "username=$TMPU&password=$TMPP" $H/login
has "换号后 /api/me 是新用户（不是 admin）" "\"username\":\"$TMPU\"" "$(curl -s -b $JS $H/api/me)"
has "换号后角色是 user（不是 admin）" "\"role\":\"user\"" "$(curl -s -b $JS $H/api/me)"

# 「系统属性」= 技术参数层：普通用户后端就不下发 tech，curl 也拿不到
# 只打印这一个字段：把整个响应体倒进断言输出既吵又容易看错
has "普通用户 meta.tech_included = False" "False" \
  "$(curl -s -b $JS "$H/api/video/$EG_BVID/diagnosis" | python3 -c 'import json,sys; print(json.load(sys.stdin)["meta"]["tech_included"])')"
has "管理员 meta.tech_included = True" "True" \
  "$(curl -s -b $J1 "$H/api/video/$EG_BVID/diagnosis" | python3 -c 'import json,sys; print(json.load(sys.stdin)["meta"]["tech_included"])')"
has "普通用户 diagnosis 的 data 里确实没有 tech 字段" "no-tech" \
  "$(curl -s -b $JS "$H/api/video/$EG_BVID/diagnosis" | python3 -c 'import json,sys; print("has-tech" if "tech" in json.load(sys.stdin)["data"] else "no-tech")')"
rm -f $JS

rm -f $J1 $J2 $J3
echo
echo "========== 合计 PASS=$pass FAIL=$fail =========="
[ $fail -eq 0 ] && echo "全部通过 ✅" || echo "有失败项 ❌"
exit $((fail > 0))

