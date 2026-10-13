#!/bin/bash
# daily_cron_guard.sh — 日频补跑守卫（幂等）
# 场景：VM 夜间挂起错过 01:00 crontab → 开机后补跑当天批次（@reboot 触发）
# 幂等三态：
#   ① 今天批次已完成（logs/collect_<YYYYMMDD>.log 含 "✅ 完成"）→ 退出
#   ② 已有采集进程在跑（pgrep bili.py hotlist）→ 退出（编排链会串行接上今天批次）
#   ③ 否则 → 跑完整 daily_cron.sh（采集 → stage → ingest）
set -uo pipefail
cd "$HOME/projects/danmaku-v2" || exit 1
STAMP=$(date +%Y%m%d)
LOG="logs/collect_$STAMP.log"

if [ -f "$LOG" ] && grep -q "✅ 完成" "$LOG"; then
  echo "[$(date +'%F %T')] guard: 今天批次已完成，跳过"
  exit 0
fi
if pgrep -f "bili.py hotlist" >/dev/null 2>&1; then
  echo "[$(date +'%F %T')] guard: 已有采集进程在跑，跳过（编排链会串行接上今天批次）"
  exit 0
fi
echo "[$(date +'%F %T')] guard: 补跑今日批次"
bash collect/daily_cron.sh
