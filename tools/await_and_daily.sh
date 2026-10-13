#!/bin/bash
# await_and_daily.sh — 等待中断批次收尾链完成后，自动执行日频全链（2026-10-13 一次性运维脚本）
# 场景：20261012 续采 + 20261013 采集由 finish_pending_batches.sh 串行收尾（约数小时）；
#       完成后本脚本接续：守卫装载（make daily）→ T3 探针 → 备份，无需人工守候。
# 启动（脱离会话）：
#   setsid nohup bash tools/await_and_daily.sh < /dev/null > /dev/null 2>&1 &
# 幂等：收尾链退出后若 /raw 两批缺失则中止（留日志人工查）；跑完即退出（一次性）。
set -uo pipefail
cd "$HOME/projects/danmaku-v2" || exit 1
LOG="logs/daily_run_$(date +%Y%m%d_%H%M).log"
log() { echo "[$(date +'%F %T')] $1" | tee -a "$LOG"; }

log "守望启动：等待 finish_pending_batches.sh 退出（串行纪律，不抢采集）..."
while pgrep -f "finish_pending_batches" >/dev/null 2>&1; do sleep 120; done
log "收尾链已退出；校验 /raw 新批次完整性..."
for bd in 20261012 20261013; do
  if docker exec namenode bash -c "hdfs dfs -test -e /raw/batch_date=$bd/danmaku.csv" 2>/dev/null; then
    log "  /raw/batch_date=$bd/danmaku.csv ✓"
  else
    log "  ❌ /raw/batch_date=$bd/danmaku.csv 缺失——中止（查 logs/collect_chain_*.log）"
    exit 1
  fi
done
log "启动日频全链：make daily BATCHES=20261012,20261013（守卫→装载→生产 dwd→对账→情感→L1→L2→L3→M4→ADS）"
{
  echo "=== daily 全链开始 $(date +'%F %T') ==="
  time make daily BATCHES=20261012,20261013
  rc=$?
  echo "=== daily 退出码 $rc ==="
} >> "$LOG" 2>&1

log "跑 T3 探针（日频首批预检）..."
docker cp tools/probe_t3_daily.py spark-master:/tmp/p3.py >> "$LOG" 2>&1
docker exec spark-master /opt/spark/bin/spark-submit --master local[2] /tmp/p3.py >> "$LOG" 2>&1

log "刷新备份（含新批次 /raw）..."
bash tools/backup.sh >> "$LOG" 2>&1
log "✅ 守望任务全部结束（结果见本日志）"
