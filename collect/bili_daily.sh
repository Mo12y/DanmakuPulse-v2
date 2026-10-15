#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# bili_daily.sh — 新系统日频采集 wrappe
# bili.py 零改动（09 §3.1 冻结：限流定标/批次身份/SIGTERM 收尾全部沿用）
# 与旧 collect_schedule.sh 的区别：不跑旧 ETL / 不写旧 SESSION_LOG / 不 git commit
# 编排：B0 期间由宿主 crontab 每日 01:00 触发（collect/daily_cron.sh）；B1 起 DolphinScheduler 接管
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 依赖：requests（B0 暂用旧仓库 venv——它只装 requests/pyspark，不污染；B1 建独立 venv）
source "$HOME/danmaku-venv/bin/activate"

LOGS="$PROJECT_ROOT/logs"
mkdir -p "$LOGS"
STAMP=$(date +%Y%m%d)
LOG_FILE="$LOGS/collect_$STAMP.log"

# 采集参数：Top 200（collect_schedule.sh 旧默认），pages=ceil(200/50)+2=6，--fresh 开新批次
# 断点续采同批：./bili_daily.sh --resume 20260101
# ⚠️ 2026-10-12 修复：bili.py 实际没有 --no-fresh 参数（其用法提示/报错文案误写，bili.py 按
#    09 §3.1 冻结零改动，问题记入 STATUS）。续采 = 不给 --fresh、只给 --batch-date（读已存快照
#    增量续采，实测有效：20261012 批次即以此方式续采）。
COUNT=200
PAGES=6
FRESH="--fresh"
if [ "${1:-}" = "--resume" ] && [ -n "${2:-}" ]; then
  FRESH="--batch-date $2"
fi

log() { echo "[$(date +'%F %T')] $1" | tee -a "$LOG_FILE"; }

log "════ 日频采集开始：Top $COUNT (pages=$PAGES, args=$FRESH) ════"
export PYTHONUNBUFFERED=1   # 坑16：stdout 接管道必须 -u + unbuffered，否则 kill 丢缓冲
# 2026-10-15 修复：pipefail + set -e 会让 SIGTERM(143) 在管道行直接终止脚本，
# 下方「需续采」提示不可达（2026-10-15 11:48 实测）。临时关 -e 捕获退出码后恢复。
set +e
python -u collect/bili.py hotlist -n "$COUNT" --pages "$PAGES" $FRESH 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

# 143=SIGTERM（用户掐断/关机）：已采部分按批次 C 仍有效，不算失败
# ⚠️ 2026-10-13 修复：143 不得再写 "✅ 完成"——daily_cron_guard.sh 以该标记判"今天已完成"，
#    若中断也写 ✅ 会让守卫误判"已完成"而静默跳过补跑（当天批次永久停在 aborted）。
#    中断态显式写"需续采"提示；仅真实完成（退出码 0）写 ✅。
log "采集结束（退出码 $EXIT_CODE）"
if [ "$EXIT_CODE" -ne 0 ] && [ "$EXIT_CODE" -ne 143 ]; then
  log "❌ 采集失败，退出码 $EXIT_CODE"
  exit 1
fi
if [ "$EXIT_CODE" -eq 143 ]; then
  log "⚠️ 采集被中断（SIGTERM）——已采部分有效但未完成；续采 = bash collect/bili_daily.sh --resume <batch_date>（不给 --fresh）"
  exit 143
else
  log "✅ 完成。产物落 data/cli_collected/（灌 HDFS 由 make ingest / daily_cron.sh 负责）"
fi
