# danmaku-v2 — B0 地基周 Makefile（交接物：up / etl / replay）
# 宿主机执行（molly，docker 组免 sudo）；所有集群作业经 spark-master 容器 spark-submit

COMPOSE   = docker compose -f compose/docker-compose.b0.yml
VENV      = $(HOME)/danmaku-venv/bin/python
SUBMIT    = $(COMPOSE) exec -T spark-master /opt/spark/bin/spark-submit \
            --master spark://spark-master:7077 \
            --conf spark.eventLog.enabled=true \
            --conf spark.eventLog.dir=file:/opt/spark/events \
            --driver-memory 512m --executor-memory 1g \
            --conf spark.executor.cores=1 --total-executor-cores 2 \
            --conf spark.sql.shuffle.partitions=8 \
            --jars /opt/spark/jars-extra/postgresql-42.7.4.jar

.PHONY: setup seed-data stage up down ps logs smoke ingest etl parity replay collect cron-install hms goldens dev-etl clean

setup: ## 一次性：下载 pg jdbc jar + disruptor + echarts（国内镜像）
	mkdir -p lib serve/static/vendor
	curl -fL -o lib/postgresql-42.7.4.jar https://maven.aliyun.com/repository/public/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar
	curl -fL -o lib/disruptor-3.3.6.jar https://maven.aliyun.com/repository/public/com/lmax/disruptor/3.3.6/disruptor-3.3.6.jar
	curl -fL -o serve/static/vendor/echarts.min.js https://registry.npmmirror.com/echarts/5.5.1/files/dist/echarts.min.js

seed-data: ## 一次性：历史 4 批三件套从旧仓库复制（旧仓库只读；meta 文件名无 .csv 段）
	mkdir -p data/cli_collected
	cp -n ~/projects/danmaku/data/cli_collected/hotlist_top100_20260902.csv \
	      ~/projects/danmaku/data/cli_collected/hotlist_top100_20260902.meta.json \
	      ~/projects/danmaku/data/cli_collected/hotlist_top200_20260907.csv \
	      ~/projects/danmaku/data/cli_collected/hotlist_top200_20260907.meta.json \
	      ~/projects/danmaku/data/cli_collected/hotlist_top200_20260914.csv \
	      ~/projects/danmaku/data/cli_collected/hotlist_top200_20260914.meta.json \
	      ~/projects/danmaku/data/cli_collected/hotlist_videos.json \
	      ~/projects/danmaku/data/cli_collected/hotlist_top100_20260917.csv \
	      ~/projects/danmaku/data/cli_collected/hotlist_top100_20260917.meta.json \
	      ~/projects/danmaku/data/cli_collected/hotlist_videos_20260917.json \
	      data/cli_collected/

stage: ## cli_collected → raw-local 暂存（映射唯一事实源 = tools/stage_raw_local.py）
	python3 tools/stage_raw_local.py

up: ## 起 B0 9 服务底座
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=50

smoke: ## B0.1 验收：compose ps + HDFS/PG 读写冒烟 + Spark master 探活
	$(COMPOSE) ps
	$(COMPOSE) exec -T namenode hdfs dfs -mkdir -p /smoke && \
	$(COMPOSE) exec -T namenode sh -c 'echo smoke | hdfs dfs -put -f - /smoke/t.txt' && \
	$(COMPOSE) exec -T namenode hdfs dfs -cat /smoke/t.txt && \
	$(COMPOSE) exec -T postgres psql -U danmaku -d ads -c "SELECT 'pg-smoke' AS ok, COUNT(*) FROM information_schema.tables WHERE table_schema='public'"
	$(COMPOSE) exec -T spark-master curl -sf http://localhost:8080/json/ | grep -oE '"aliveworkers":\s*"?[0-9]+'

ingest: ## raw-local → HDFS /raw（幂等覆盖）
	$(COMPOSE) exec -T namenode bash /mnt/assets/ingest_raw.sh

etl: ## B0.3 全链：M0-min → L1-min → ADS（幂等分区覆盖）
	$(SUBMIT) /opt/work/jobs/m0_min.py
	$(SUBMIT) /opt/work/jobs/m2_l1_min.py
	$(SUBMIT) /opt/work/jobs/ads_min.py

# HMS 表目录注册 —— B1 2026-10-12 起 metastore **已自建**（compose 的 hive-metastore 服务：
# 复用 danmaku-spark 镜像的 Hive 2.3.9 依赖 + datanucleus autoCreateSchema；配置见 compose/assets/hms/hive-site.xml）。
# 历史背景（B0 裁定）：原 bde2020/hive-metastore-postgresql:2.3.0 实为 HMS 背后的 PG（9083 拒连实测），
# 正确镜像两源不可得（daocloud 403 / 1ms.run not found）→ 曾移出 etl 主链；B1 自建后恢复可用。
# ⚠️ 恢复手册（2026-10-12 实测）：若 create_table 卡死（jstack 显示 datanucleus 等 PG 锁），执行
#   docker exec -i postgres psql -U danmaku -d metastore_db -c '...terminate idle in transaction...'
#   docker compose -f compose/docker-compose.b0.yml restart hive-metastore
#   完整根因与处置见 CHANGELOG 2026-10-12「HMS 自建」条。
hms: ## HMS 表目录注册（外部表 + MSCK；B1 起 metastore 自建已可用）
	$(SUBMIT) /opt/work/jobs/register_hms.py

# ── B1 生产链（与 min 对拍链隔离：min 链锚定 672,439，本链新口径 676,931）──
m0-full: ## M0 完整版：标准化+清洗打标+刷量检测（dwd_danmaku_clean / dim_video / dwd_video_metric_snapshot / dwd_spam_flag）
	$(SUBMIT) /opt/work/jobs/m0_full.py

m1: ## M1 对账：三值对账+归因+偏差+质量分（dws_batch_audit / dws_reconcile_detail / PG ads_quality_report）
	$(SUBMIT) /opt/work/jobs/m1_reconcile.py

m5: ## M5 词典 v0：9 情形打分+锚词+词典表+情感曲线（dwd_danmaku_sent / dict_* / dws_sentiment_curve_1pct）
	$(SUBMIT) /opt/work/jobs/m5_sentiment_v0.py

b1: m0-full m1 m5 ## B1 全链（三段串跑；每段各自验收后 commit）

parity: ## 对拍验收：曲线101点+8维 vs goldens + 池级锚点
	$(SUBMIT) /opt/work/jobs/parity_check.py

replay: ## 幂等验收：全链 ETL 两次 → 规范导出比对（整数精确/浮点≤1e-9）+ 对拍
	rm -rf data-out/canonical_run1 data-out/canonical_run2
	$(SUBMIT) /opt/work/jobs/m0_min.py && $(SUBMIT) /opt/work/jobs/m2_l1_min.py && $(SUBMIT) /opt/work/jobs/ads_min.py
	mv data-out/canonical data-out/canonical_run1
	$(SUBMIT) /opt/work/jobs/m0_min.py && $(SUBMIT) /opt/work/jobs/m2_l1_min.py && $(SUBMIT) /opt/work/jobs/ads_min.py
	mv data-out/canonical data-out/canonical_run2
	python3 tools/compare_canonical.py data-out/canonical_run1 data-out/canonical_run2
	$(SUBMIT) /opt/work/jobs/parity_check.py
	@echo "✅ 幂等验收通过：全链两跑一致（整数精确 / 浮点≤1e-9）+ 对拍 PASS"
	@echo "CHAIN_REPLAY_OK"

collect: ## 手动触发一次日频采集（Top200，约 2-4 小时）
	bash collect/bili_daily.sh

cron-install: ## 安装日频 crontab（每日 01:00 采集+灌 raw）
	(crontab -l 2>/dev/null | grep -v daily_cron; \
	 echo '0 1 * * * cd $(CURDIR) && bash collect/daily_cron.sh >> logs/cron_daily.log 2>&1') | crontab -
	crontab -l | tail -2

goldens: ## 生成对拍基准（宿主机旧 venv，只读旧仓库）+ 上传 HDFS /opt/goldens（集群 parity 实际读取源）
	$(VENV) tools/export_goldens.py --old-root ~/projects/danmaku --out tests/goldens
	$(COMPOSE) exec -T namenode hdfs dfs -mkdir -p /opt/goldens
	$(COMPOSE) exec -T namenode hdfs dfs -put -f - /opt/goldens/goldens_curve.csv < tests/goldens/goldens_curve.csv
	$(COMPOSE) exec -T namenode hdfs dfs -put -f - /opt/goldens/goldens_metrics.csv < tests/goldens/goldens_metrics.csv

dev-etl: ## 本地模式全链（不起 Docker，file:// 路径——开发迭代用）
	$(VENV) jobs/m0_min.py --local --raw-root file://$(CURDIR)/data/raw-local --wh-root file://$(CURDIR)/data-out/dev-wh
	$(VENV) jobs/m2_l1_min.py --local --wh-root file://$(CURDIR)/data-out/dev-wh
	$(VENV) jobs/parity_check.py --local --wh-root file://$(CURDIR)/data-out/dev-wh --goldens-dir $(CURDIR)/tests/goldens

clean:
	rm -rf data-out/dev-wh data-out/canonical*
