#!/usr/bin/env bash
# load_v3_danmaku.sh — 把弹幕正文灌进 PG（v3 证据钻取供数）
#
# 为什么需要它：诊断卡的"点偏离 → 看这段观众在刷什么"要按时刻查真实弹幕。
# 源表 dwd_danmaku_clean 在 HDFS 上，fastapi 容器够不着；111.5 万条也不适合塞进 JSON 快照。
# 落 PG（fastapi 本来就连着）+ (bvid,batch_date,t) 索引，查某时刻是毫秒级。
#
# 表名带 v3_ 前缀：这是**服务层用表**，不属于 ads_* 生产链，别混。
#
# 用法：bash jobs/load_v3_danmaku.sh          （先跑 jobs/export_v3_danmaku.py 出 CSV）
set -euo pipefail
cd "$(dirname "$0")/.."

PG=postgres
DB=ads
DBUSER=danmaku
SRC=$(ls data-out/v3_danmaku/part-*.csv 2>/dev/null | head -1 || true)
if [ -z "$SRC" ]; then
  echo "找不到 data-out/v3_danmaku/part-*.csv —— 先跑：" >&2
  echo "  docker cp jobs/export_v3_danmaku.py spark-master:/tmp/ && \\" >&2
  echo "  docker exec spark-master /opt/spark/bin/spark-submit --master local[2] /tmp/export_v3_danmaku.py" >&2
  exit 1
fi
echo "源文件：$SRC（$(du -h "$SRC" | cut -f1)）"
docker cp "$SRC" $PG:/tmp/v3_danmaku.csv

docker exec -i $PG psql -U $DBUSER -d $DB -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE IF NOT EXISTS v3_video_danmaku (
  bvid       varchar(16)  NOT NULL,
  batch_date varchar(8)   NOT NULL,
  t          real         NOT NULL,
  txt        varchar(140) NOT NULL
);
TRUNCATE v3_video_danmaku;
COPY v3_video_danmaku (bvid, batch_date, t, txt)
     FROM '/tmp/v3_danmaku.csv' WITH (FORMAT csv, QUOTE '"', ESCAPE '"');
CREATE INDEX IF NOT EXISTS idx_v3_vd_bvid_t ON v3_video_danmaku (bvid, batch_date, t);
ANALYZE v3_video_danmaku;
SELECT count(*) AS rows, count(DISTINCT bvid) AS videos FROM v3_video_danmaku;
SQL
docker exec $PG rm -f /tmp/v3_danmaku.csv
echo "完成"

