#!/bin/bash
# 灌批次三件套到 HDFS /raw/batch_date=YYYYMMDD/（幂等：-put -f 覆盖写）
# 在 namenode 容器内执行：docker compose -f compose/docker-compose.b0.yml exec namenode bash /mnt/assets/ingest_raw.sh
# 数据源：新仓库 data/cli_collected/（历史 4 批由 make seed-data 从旧仓库复制；新批次由 collect/bili_daily.py 采集）
#
# 榜单快照映射（2026-10-12 勘察结论）：
#   20260902 / 20260907：快照永久丢失（旧系统 hotlist_videos.json 被覆盖）→ 只灌 CSV+meta
#   20260914：快照幸存于旧仓库 legacy hotlist_videos.json（fetched_at=2026-09-14T13:12 与 meta 一致）
#   20260917：快照 hotlist_videos_20260917.json
#   之后日频批次：bili.py 自动落 hotlist_videos_<date>.json
set -euo pipefail
SRC=/mnt/data/cli_collected

ingest() {  # $1=batch_date  $2=csv文件名  $3=快照json文件名(可空)
  local bd=$1 csv=$2 snap=${3:-}
  hdfs dfs -mkdir -p "/raw/batch_date=$bd"
  hdfs dfs -put -f "$SRC/$csv" "/raw/batch_date=$bd/danmaku.csv"
  hdfs dfs -put -f "$SRC/$csv.meta.json" "/raw/batch_date=$bd/meta.json"
  if [ -n "$snap" ] && [ -f "$SRC/$snap" ]; then
    hdfs dfs -put -f "$SRC/$snap" "/raw/batch_date=$bd/hotlist.json"
  else
    echo "  [warn] batch=$bd 无榜单快照（ods_hotlist_video 该分区为空）"
  fi
  echo "  [ok] batch=$bd"
}

# ── 历史 4 批（周频 legacy，cadence=weekly_legacy）──
ingest 20260902 hotlist_top100_20260902.csv ""
ingest 20260907 hotlist_top200_20260907.csv ""
ingest 20260914 hotlist_top200_20260914.csv hotlist_videos.json
ingest 20260917 hotlist_top100_20260917.csv hotlist_videos_20260917.json

# ── 日频新批次（自动发现：有对应 hotlist_videos_<date>.json 的才算完整批次）──
for csv in "$SRC"/hotlist_top*.csv; do
  [ -f "$csv" ] || continue
  base=$(basename "$csv" .csv)
  bd=$(echo "$base" | grep -oE '[0-9]{8}')
  case $bd in
    20260902|20260907|20260914|20260917) continue ;;  # 历史批已处理
  esac
  snap="hotlist_videos_$bd.json"
  if [ -f "$SRC/$snap" ]; then
    ingest "$bd" "$(basename "$csv")" "$snap"
  else
    echo "  [skip] $bd：CSV 存在但快照缺失（采集可能中断），不灌入"
  fi
done

echo '── /raw 清单 ──'
hdfs dfs -ls -R /raw | head -40
