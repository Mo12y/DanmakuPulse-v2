#!/bin/bash
# backup.sh — B2.4 备份（raw 外备 + pg_dump）｜宿主执行｜幂等可重跑
# 用法：bash tools/backup.sh            （默认保留最近 3 份）
#       KEEP=5 bash tools/backup.sh     （自定义保留份数）
# 产物：data-out/backups/<时间戳>/{ads.sql.gz, metastore_db.sql.gz, raw/, MANIFEST.txt}
set -euo pipefail
cd "$(dirname "$0")/.."
KEEP=${KEEP:-3}
TS=$(date +%Y%m%d_%H%M%S)
DIR="data-out/backups/$TS"
mkdir -p "$DIR"

echo "[1/4] pg_dump：ads + metastore_db → $DIR"
docker exec postgres pg_dump -U danmaku -d ads | gzip > "$DIR/ads.sql.gz"
docker exec postgres pg_dump -U danmaku -d metastore_db | gzip > "$DIR/metastore_db.sql.gz"

echo "[2/4] HDFS /raw 外备（copyToLocal → docker cp）"
docker exec namenode bash -c 'rm -rf /tmp/raw_backup && hdfs dfs -copyToLocal /raw /tmp/raw_backup'
docker cp namenode:/tmp/raw_backup "$DIR/raw"
docker exec namenode rm -rf /tmp/raw_backup

echo "[3/4] 清单"
{
  echo "# backup $TS"
  echo "# pg: postgres/ads + postgres/metastore_db（pg_restore 用 gunzip | psql）"
  echo "# raw: HDFS /raw 全量（danmaku.csv/meta.json/hotlist.json per batch）"
  du -sh "$DIR" 2>/dev/null || true
  find "$DIR" -type f | wc -l
} > "$DIR/MANIFEST.txt"
cat "$DIR/MANIFEST.txt"

echo "[4/4] 保留最近 $KEEP 份"
ls -1dt data-out/backups/*/ 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -rf
ls -1dt data-out/backups/*/ 2>/dev/null | head -$KEEP

echo "✅ backup 完成 → $DIR"
du -sh "$DIR"
