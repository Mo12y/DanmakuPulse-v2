#!/bin/bash
# daily_cron_guard.sh — 日频补跑守卫（幂等；2026-10-14 扩展：覆盖 VM 挂起场景）
# 场景一：VM 夜间挂起/关机错过 01:00 crontab → 恢复后由 @reboot + 每小时兜底
#         （crontab: 17 * * * *）触发补跑（挂起不触发 @reboot——2026-10-14 实测缺口）
# 场景二：当天批次中断（aborted/部分）→ 补跑时自动「续采」而不是重开新榜
# 幂等状态机：
#   ① 今天批次已完成（logs/collect_<today>.log 含 "✅ 完成" 或 meta status=complete）→ 退出
#   ② 已有采集/日链进程在跑（pgrep bili.py hotlist / daily_cron）→ 退出
#   ③ 有当天未完成批次（meta 存在）→ 续采（--resume）；否则 → 新采
#   ④ 采集后：stage_raw_local → ingest（HDFS 未起则暂存告警）
# 注：ETL 全链（make daily）仍为手动/后续 DS 编排，本守卫只负责「批次到 raw」。
set -uo pipefail
cd "$HOME/projects/danmaku-v2" || exit 1
STAMP=$(date +%Y%m%d)
LOG="logs/collect_$STAMP.log"
META="data/cli_collected/hotlist_top200_$STAMP.meta.json"

if { [ -f "$LOG" ] && grep -q "✅ 完成" "$LOG"; } \
   || { [ -f "$META" ] && grep -q '"status": *"complete"' "$META"; }; then
  echo "[$(date +'%F %T')] guard: 今天批次已完成，跳过"
  exit 0
fi
# 2026-10-15 修：原第二段模式 "daily_cron" 会匹配守卫自身（cron 的 sh -c 命令行含 daily_cron_guard.sh）
# ⇒ 每次运行都在此分支自我跳过，补跑永不触发。改为精确匹配 collect/daily_cron.sh。
if pgrep -f "bili.py hotlist" >/dev/null 2>&1 || pgrep -f "collect/daily_cron.sh" >/dev/null 2>&1; then
  echo "[$(date +'%F %T')] guard: 已有采集/日链进程在跑，跳过（编排链会串行接上今天批次）"
  exit 0
fi

if [ -f "$META" ]; then
  echo "[$(date +'%F %T')] guard: 发现当天未完成批次（中断态）→ 续采 $STAMP"
  bash collect/bili_daily.sh --resume "$STAMP"
else
  echo "[$(date +'%F %T')] guard: 补跑今日批次（新采）"
  bash collect/bili_daily.sh
fi

# 采集后：暂存 raw-local + 灌 HDFS（与 daily_cron.sh 同构；失败不致命）
python3 tools/stage_raw_local.py || echo "[$(date +'%F %T')] [error] stage 失败，手动重跑 tools/stage_raw_local.py"
if docker compose -f compose/docker-compose.b0.yml ps --status running 2>/dev/null | grep -q namenode; then
  docker compose -f compose/docker-compose.b0.yml exec -T namenode bash /mnt/assets/ingest_raw.sh
  echo "[$(date +'%F %T')] guard：灌入完成（ETL 全链 make daily 仍手动 / 待 DS 编排）"
else
  echo "[$(date +'%F %T')] [warn] HDFS 未运行，新批次已暂存 data/raw-local/"
fi
