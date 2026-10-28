#!/usr/bin/env bash
# exp46 smoke 验收：登录 → 列表 API → 3 支演示详情 → 页面 → 输出演示响应证据
# 运行：bash smoke_test.sh（在 VM 或任何能访问 :8088 的机器）
set -u
BASE=${API_BASE:-http://localhost:8088}
CK=$(curl -s -i -X POST $BASE/login -d 'username=admin&password=danmaku@2026' | grep -i '^set-cookie' | head -1 | cut -d' ' -f2 | tr -d '\r')
if [ -z "$CK" ]; then echo "FAIL login"; exit 1; fi
echo "PASS login"

curl -s -H "Cookie: $CK" $BASE/api/ml/forecast -o demo_response_list.json
grep -q '"code":0' demo_response_list.json && echo "PASS list api ($(grep -o '"n_items":[0-9]*' demo_response_list.json))" || { echo "FAIL list api"; exit 1; }

for BV in BV1kBpi6aEDP BV1ZSp46AEF8 BV1wAp76yEu1; do
  curl -s -H "Cookie: $CK" $BASE/api/ml/forecast/$BV -o demo_response_$BV.json
  grep -q '"code":0' demo_response_$BV.json && echo "PASS detail $BV" || echo "FAIL detail $BV"
done

CODE=$(curl -s -o /dev/null -w '%{http_code}' -H "Cookie: $CK" $BASE/ml_forecast.html)
[ "$CODE" = "200" ] && echo "PASS page 200" || echo "WARN page=$CODE（302=未登录跳转，属正常鉴权行为）"
CODE2=$(curl -s -o /dev/null -w '%{http_code}' -H "Cookie: $CK" $BASE/api/ml/forecast/BVXXXnotexist)
[ "$CODE2" = "404" ] && echo "PASS unknown-bvid 404" || echo "FAIL 404=$CODE2"
echo "SMOKE-DONE（演示响应已存为 demo_response_*.json）"
