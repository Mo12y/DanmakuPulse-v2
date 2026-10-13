#!/bin/bash
# perf_suite.sh — B5.1 性能套件（2026-10-13 排期：等夜间链结束→放大→网格）幂等可重跑
# 运行：setsid bash tools/perf_suite.sh < /dev/null > /dev/null 2>&1 &
# 产物：logs/perf_suite_*.log + data-out/perf_results.jsonl（追加）
set -uo pipefail
cd "$HOME/projects/danmaku-v2" || exit 1
LOG="logs/perf_suite_$(date +%Y%m%d_%H%M).log"
log() { echo "[$(date +'%F %T')] $1" >> "$LOG"; }

run() {  # par  pyfiles  mode  amp
  local par="$1" pf="$2" mode="$3" amp="$4"
  log "run $mode amp=$amp par=$par"
  docker exec spark-master /opt/spark/bin/spark-submit --master "local[$par]" --driver-memory 2g \
    $pf /tmp/perf_harness.py --mode "$mode" --amp "$amp" >> "$LOG" 2>&1
}

log "套件启动：等待夜间链结束（守望结束标记 / 守望进程退出）..."
while true; do
  if grep -q "守望任务全部结束" logs/daily_run_*.log 2>/dev/null; then
    log "检测到守望结束标记"; break
  fi
  if ! pgrep -f "await_and_daily|finish_pending" >/dev/null 2>&1; then
    log "守望/收尾进程不在——5 分钟后复核"; sleep 300
    if ! pgrep -f "await_and_daily|finish_pending" >/dev/null 2>&1; then
      log "确认无人活动，直接开始"; break
    fi
  fi
  sleep 120
done

# ① 放大数据集（若已存在则跳过）
for m in 1 4 8; do
  if docker exec namenode bash -c "hdfs dfs -test -e /warehouse/perf_amp_${m}x/_SUCCESS" 2>/dev/null; then
    log "amp ${m}x 已存在，跳过"
  else
    log "构建 amp ${m}x ..."
    docker exec spark-master /opt/spark/bin/spark-submit --master local[2] --driver-memory 2g \
      /tmp/make_amplified.py --mult $m >> "$LOG" 2>&1
  fi
done

# ② 密度网格：1/4/8 × par{1,2}
for m in 1 4 8; do for p in 1 2; do run $p "" density $m; done; done

# ③ 事件网格：1/4/8 × par2
for m in 1 4 8; do
  run 2 "--py-files /opt/work/jobs/l2_common.py,/opt/work/jobs/m2_l2_events.py" events $m
done

# ④ Hawkes 加速比复测 ×2（取均值）
run 2 "--py-files /opt/work/jobs/hawkes_core.py" hawkes 1
run 2 "--py-files /opt/work/jobs/hawkes_core.py" hawkes 1

# ⑤ 倾斜对照：1× 与 8×
run 2 "" skew 1
run 2 "" skew 8

log "✅ 性能套件全部结束"
