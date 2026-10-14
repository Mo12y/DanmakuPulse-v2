#!/bin/bash
# await_daily_today.sh — 守望：等今日采集/灌入链结束 → make daily 全链 → T3 探针 → 备份（2026-10-14 新增，可复用）
# 用法（脱离会话）：
#   setsid nohup bash collect/await_daily_today.sh [BATCHES] >> logs/daily_run_$(date +%Y%m%d_%H%M).log 2>&1 &
#   BATCHES 缺省 = 今日日期 YYYYMMDD；幂等：等链静默后校验 /raw 完整性，缺则中止留日志。
set -uo pipefail
cd "$HOME/projects/danmaku-v2" || exit 1
BD="${1:-$(date +%Y%m%d)}"
LOG="logs/daily_run_$(date +%Y%m%d_%H%M).log"
log() { echo "[$(date +'%F %T')] $1" | tee -a "$LOG"; }

log "守望启动：等待采集/灌入链退出（batch=$BD）..."
while pgrep -f "bili.py hotlist|daily_cron" >/dev/null 2>&1; do sleep 120; done
log "现场已静默；校验 /raw/batch_date=$BD 完整性..."
if ! docker exec namenode hdfs dfs -test -e "/raw/batch_date=$BD/danmaku.csv" 2>/dev/null; then
  log "❌ /raw/batch_date=$BD/danmaku.csv 缺失——中止（查 logs/collect_*.log）"
  exit 1
fi
log "raw 就位。启动 make daily BATCHES=$BD"
{
  echo "=== daily 全链开始 $(date +'%F %T') ==="
  time make daily BATCHES="$BD"
  rc=$?
  echo "=== daily 退出码 $rc ==="
} >> "$LOG" 2>&1

log "T3 探针（日频预检）..."
docker cp tools/probe_t3_daily.py spark-master:/tmp/p3.py >> "$LOG" 2>&1
docker exec spark-master /opt/spark/bin/spark-submit --master local[2] /tmp/p3.py >> "$LOG" 2>&1

log "刷新备份（含新批次 /raw）..."
bash tools/backup.sh >> "$LOG" 2>&1

log "转录 M4/演化报告摘要到日志（供交接引用）..."
for f in m4_track_report.json m4_evolution_report.json m4_onlist_report.json t3_probe_report.json; do
  echo "----- $f -----" >> "$LOG"
  cat "data-out/$f" >> "$LOG" 2>/dev/null
done
log "✅ 守望任务全部结束（日志：$LOG）"
