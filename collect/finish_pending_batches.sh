#!/bin/bash
# finish_pending_batches.sh — 中断批次收尾链（2026-10-13 一次性运维脚本）
# 场景：VM 挂起/关机打断采集；@reboot guard 只补"今天"批次（--fresh），
#       被打断的上一批次（20261012, status=aborted 155/200）无人续采。
# 本脚本串行收尾（不得与采集并发——限流纪律）：
#   ① 等 daily_cron 全链退出 → ② 续采 20261012 → ③ stage → ④ ingest（幂等）
# 启动（VM 会挂起，必须脱离会话）：
#   setsid nohup bash collect/finish_pending_batches.sh >> logs/collect_chain_$(date +%Y%m%d).log 2>&1 &
# 注意：bili_daily.sh 内部日志文件按"当天日期"命名（collect_<today>.log），
#       20261012 的续采输出会追加进 logs/collect_<today>.log——已知，不影响语义。
set -uo pipefail
cd "$HOME/projects/danmaku-v2" || exit 1
LOG="logs/collect_chain_$(date +%Y%m%d).log"
log() { echo "[$(date +'%F %T')] $1" | tee -a "$LOG"; }

log "链路启动：等待当前采集/日链退出（串行纪律）..."
while pgrep -f "bili.py hotlist|daily_cron" >/dev/null 2>&1; do
  sleep 120
done
log "现场已静默，开始续采 20261012（只给 --batch-date，读已存快照增量续采）"
bash collect/bili_daily.sh --resume 20261012
log "续采结束，开始 stage（cli_collected → raw-local）"
python3 tools/stage_raw_local.py || log "[error] stage 失败，手动重跑 tools/stage_raw_local.py"
if docker compose -f compose/docker-compose.b0.yml ps --status running 2>/dev/null | grep -q namenode; then
  log "ingest：raw-local → HDFS（幂等覆盖，含 20261012 + 20261013 两批）"
  docker compose -f compose/docker-compose.b0.yml exec -T namenode bash /mnt/assets/ingest_raw.sh
  log "✅ 收尾链完成（两批已灌 HDFS /raw；ETL 请后续 make 链手动跑）"
else
  log "[warn] HDFS 未运行，批次已暂存 data/raw-local/，起服务后手动 make ingest"
fi