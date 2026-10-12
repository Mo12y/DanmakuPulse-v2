#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# daily_cron.sh — 每日 D+1 全流程（B0 期间宿主 crontab 入口）
#   01:00 采集（bili_daily.sh，约 2-4 小时）→ 完成后自动灌新批次入 HDFS
#   ETL/ADS 仍手动（make etl）——B2.4 起由 DolphinScheduler DAG T1-T9 接管
# crontab 条目（molly 用户，免 sudo）：
#   0 1 * * * cd ~/projects/danmaku-v2 && bash collect/daily_cron.sh >> logs/cron_daily.log 2>&1
# ═══════════════════════════════════════════════════════════════
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
LOGS="$PROJECT_ROOT/logs"; mkdir -p "$LOGS"

echo "[$(date +'%F %T')] ════ daily_cron 开始 ════"

# 1) 采集（失败则终止——没新数据就不用灌）
if bash collect/bili_daily.sh; then
  # 2) 灌入 HDFS（ingest_raw.sh 自动发现新批次；Docker 未起时跳过并告警）
  if docker compose -f compose/docker-compose.b0.yml ps --status running 2>/dev/null | grep -q namenode; then
    docker compose -f compose/docker-compose.b0.yml exec -T namenode bash /mnt/assets/ingest_raw.sh
    echo "[$(date +'%F %T')] 灌入完成。ETL 请手动：make etl"
  else
    echo "[$(date +'%F %T')] [warn] HDFS 未运行，新批次暂存 data/cli_collected/，起服务后手动 make ingest"
  fi
else
  echo "[$(date +'%F %T')] [error] 采集失败，本日跳过灌入（详见 logs/collect_*.log）"
fi
echo "[$(date +'%F %T')] ════ daily_cron 结束 ════"
