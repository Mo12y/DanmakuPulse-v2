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

setup: ## 一次性：下载 pg jdbc jar + echarts（国内镜像）
	mkdir -p lib serve/static/vendor
	curl -fL -o lib/postgresql-42.7.4.jar https://maven.aliyun.com/repository/public/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar
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

# HMS 表目录注册 —— B0 不算在 etl 链内（见 STATUS「B0 冻结决策」）：
# compose 的 ${HIVE_IMAGE}=bde2020/hive-metastore-postgresql:2.3.0 是 HMS 背后的 PG，
# 不是 metastore 服务本身（容器内 HiveMetaStore 进程数=0、9083 拒连）；
# 正确的 bde2020/hive:2.3.0-postgresql-metastore 在 daocloud 403、1ms.run not found。
# B0 的 DoD 不依赖表目录（作业按 Parquet 路径直读、ADS 落 PostgreSQL），故移出主链。
hms: ## 可选：HMS 表目录注册（B1 恢复；需先有 metastore 服务）
	$(SUBMIT) /opt/work/jobs/register_hms.py

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
