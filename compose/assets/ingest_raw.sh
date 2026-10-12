#!/bin/bash
# raw-local → HDFS /raw/batch_date=YYYYMMDD/ 同步（幂等：-put -f 覆盖）
# 在 namenode 容器内执行：docker compose -f compose/docker-compose.b0.yml exec namenode bash /mnt/assets/ingest_raw.sh
# 批次映射（哪个 CSV 配哪个快照）的唯一事实源 = tools/stage_raw_local.py（宿主机先暂存到 data/raw-local/）
set -euo pipefail
SRC=/mnt/data/raw-local

for d in "$SRC"/batch_date=*/; do
  bd=$(basename "$d" | cut -d= -f2)
  hdfs dfs -mkdir -p "/raw/batch_date=$bd"
  for f in danmaku.csv meta.json hotlist.json; do
    if [ -f "$d$f" ]; then
      hdfs dfs -put -f "$d$f" "/raw/batch_date=$bd/$f"
    fi
  done
  echo "  [ok] batch=$bd: $(hdfs dfs -ls "/raw/batch_date=$bd" | tail -n +2 | awk '{print $NF}' | tr '\n' ' ')"
done

echo '── /raw 清单 ──'
hdfs dfs -ls -R /raw | head -40
