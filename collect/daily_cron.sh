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

# 1) 采集（0=成功→灌入；143=SIGTERM 中断→跳过（守卫续采后补）；其他=失败→跳过）
bash collect/bili_daily.sh
RC=$?
if [ "$RC" -eq 0 ]; then
  # 2) 暂存 raw-local（批次映射唯一事实源=stage_raw_local.py；纯本地、不依赖 Docker。
  #    ⚠️ 2026-10-12 修复：此前缺本环——新批次停留在 cli_collected/，ingest 只见旧 raw-local，
  #    "自动灌 raw" 实际断链；且本步必须先于 HDFS 判断（HDFS 未起时也先把批次暂存好）。
  python3 tools/stage_raw_local.py || echo "[$(date +'%F %T')] [error] stage 失败，检查后手动重跑 tools/stage_raw_local.py"
  # 3) 灌入 HDFS（ingest_raw.sh 幂等覆盖 raw-local 全部批次；Docker 未起时跳过并告警）
  if docker compose -f compose/docker-compose.b0.yml ps --status running 2>/dev/null | grep -q namenode; then
    docker compose -f compose/docker-compose.b0.yml exec -T namenode bash /mnt/assets/ingest_raw.sh
    echo "[$(date +'%F %T')] 灌入完成。ETL 请手动：make etl"
  else
    echo "[$(date +'%F %T')] [warn] HDFS 未运行，新批次已暂存 data/raw-local/，起服务后手动 make ingest"
  fi
elif [ "$RC" -eq 143 ]; then
  echo "[$(date +'%F %T')] [warn] 采集中断（SIGTERM；已采部分停留 cli_collected）——跳过灌入；守卫（每小时兜底）将自动续采并灌入"
else
  echo "[$(date +'%F %T')] [error] 采集失败（rc=$RC），本日跳过灌入（详见 logs/collect_*.log）"
fi
echo "[$(date +'%F %T')] ════ daily_cron 结束 ════"
