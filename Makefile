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

# L2 链：检测器/分段以 RDD + NumPy 数据并行执行（11 §0「算法保真优先」）——需把共享模块
# l2_common.py 分发到 workers（闭包反序列化时按模块名 import）
SUBMIT_L2 = $(SUBMIT) --py-files /opt/work/jobs/l2_common.py
# 对拍/复放链专属批次范围（legacy 4 批）——日频新批次不得并入 min 链（2026-10-13）
LEGACY_BATCHES := 20260902,20260907,20260914,20260917

# L3 链：Hawkes 逐视频 L-BFGS-B（scipy）——闭包依赖 hawkes_core（--py-files 分发）
SUBMIT_L3 = $(SUBMIT) --py-files /opt/work/jobs/hawkes_core.py
# M6 链：特征矩阵闭包依赖 l2_common + m2_l2_events 纯函数（--py-files 分发）
SUBMIT_M6 = $(SUBMIT) --py-files /opt/work/jobs/l2_common.py,/opt/work/jobs/m2_l2_events.py
# M4 链 v2（2026-10-14）：跨批轨迹/演化分型——闭包依赖 m4_track_common（+l2_common）
SUBMIT_M4 = $(SUBMIT) --py-files /opt/work/jobs/m4_track_common.py,/opt/work/jobs/l2_common.py

.PHONY: setup seed-data stage up down ps logs smoke ingest etl parity replay collect cron-install hms goldens dev-etl clean m2-l2 m2-l2-events m2-l2-sync m2-l2-lag m2-l2-novelty m2-l2-finalize l2-candidates l2-eval m2-l1-cluster cluster-kshape m2-l3 m2-l3-hawkes m4 m4-onlist m4-track m4-evolution m4-up ads-build serve-rebuild m0-daily daily m6 m6-features m6-train m6-panel perf-suite

setup: ## 一次性：下载 pg jdbc jar + disruptor + echarts（国内镜像）
	mkdir -p lib serve/static/vendor
	curl -fL -o lib/postgresql-42.7.4.jar https://maven.aliyun.com/repository/public/org/postgresql/postgresql/42.7.4/postgresql-42.7.4.jar
	curl -fL -o lib/disruptor-3.3.6.jar https://maven.aliyun.com/repository/public/com/lmax/disruptor/3.3.6/disruptor-3.3.6.jar
	curl -fL -o serve/static/vendor/echarts.min.js https://registry.npmmirror.com/echarts/5.5.1/files/dist/echarts.min.js
	curl -fL -o serve/static/vendor/html2canvas.min.js https://registry.npmmirror.com/html2canvas/1.4.1/files/dist/html2canvas.min.js

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


session-secret: ## 生成会话签名密钥写入 compose/.env（不进 git；缺失时自动补）
	@python3 -c "import io,secrets;p='compose/.env';t=io.open(p,encoding='utf-8').read();n=chr(10);t=t if 'ADS_SESSION_SECRET=change-me

VM_IP ?= 192.168.88.100

tls-cert: ## 生成自签证书（8088 的 TLS 用；证书不进 git，缺失时自动生成）
	@mkdir -p compose/assets/tls
	@test -f compose/assets/tls/server.crt || (openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
	  -keyout compose/assets/tls/server.key -out compose/assets/tls/server.crt \
	  -subj '/CN=danmaku-diagnosis' \
	  -addext 'subjectAltName=IP:$(VM_IP),IP:127.0.0.1,DNS:localhost' 2>/dev/null \
	  && echo '已生成自签证书（浏览器首次用 HTTPS 访问会提示一次，点继续前往即可）')
	@echo '证书就绪'

up: tls-cert ## 起 B0 9 服务底座
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

etl: ## B0.3 全链：M0-min → L1-min → ADS（幂等分区覆盖；限定 legacy 4 批）
	$(SUBMIT) /opt/work/jobs/m0_min.py --batches $(LEGACY_BATCHES)
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

m2-l1-density: ## L1 密度引擎（dws_density_5s / dws_density_1pct）
	$(SUBMIT) /opt/work/jobs/m2_l1_density.py

m2-l1-fingerprint: ## L1 指纹计算器（dws_video_rhythm_features）
	$(SUBMIT) /opt/work/jobs/m2_l1_fingerprint.py

m2-l1-baseline: ## L1 分区基准库（dws_partition_rhythm）
	$(SUBMIT) /opt/work/jobs/m2_l1_baseline.py

m2-l1: m2-l1-density m2-l1-fingerprint m2-l1-baseline ## L1 体积层全链

# ── B2 结构层（L2，08 §4；顺序 = 检测→同步→滞后→分段→收尾回填；每段可独立重跑）──
m2-l2-events: ## L2 四检测器+融合（dws_rhythm_events；未标定 → calibrated_on='uncalibrated'）
	$(SUBMIT_L2) /opt/work/jobs/m2_l2_events.py

m2-l2-sync: ## L2 同步性+羊群（dws_sync_features；回写事件表 sync 列）
	$(SUBMIT) /opt/work/jobs/m2_l2_sync.py

m2-l2-lag: ## L2 锚词滞后校正（dws_lag_calibration；回写密度 cal_cnt）
	$(SUBMIT) /opt/work/jobs/m2_l2_lag.py

m2-l2-novelty: ## L2 Novelty 分段（dws_segments）
	$(SUBMIT_L2) /opt/work/jobs/m2_l2_novelty.py

m2-l2-finalize: ## L2 收尾：回填 L1 指纹（peak_*/herd/lag_delta）+ feat_z_part
	$(SUBMIT) /opt/work/jobs/m2_l2_finalize.py

l2-candidates: ## 21 号 3：导出 theta 截断前的全候选（评估/标定用，非流水线环节）
	$(SUBMIT_L2) /opt/work/jobs/m2_l2_candidates_export.py
	@echo "-> data-out/l2_candidates.csv"

l2-eval: ## 21 号 4 表2：逐方法 P/R/AP 对比（TRUTH=<人工标注 csv>）
	$(PY) tools/eval_l2_peaks.py --cand data-out/l2_candidates.csv --truth $(TRUTH)

m2-l2: m2-l2-events m2-l2-sync m2-l2-lag m2-l2-novelty m2-l2-finalize ## L2 结构层全链（B2）

# ── 形态聚类（08 §3.4：K-means++ MLlib 全量 + K-shape 宿主 venv「driver 侧」）──
m2-l1-cluster: ## 形态聚类全链：K-means++ → 导出 → 宿主 K-shape + ARI → 合并入表（三步幂等）
	$(SUBMIT) /opt/work/jobs/m2_l1_cluster.py
	$(VENV) tools/kshape_validate.py
	$(SUBMIT) /opt/work/jobs/m2_l1_cluster.py

cluster-kshape: ## 形态聚类·仅宿主段（K-shape + 两法 ARI + sklearn 复算；需先跑过 m2-l1-cluster）
	$(VENV) tools/kshape_validate.py

# ── B2 机制层（L3，08 §5；BTM/时变弹性属周级，未含）──
m2-l3-hawkes: ## L3 Hawkes 拟合（dws_hawkes_params；镜像 scipy 已在 Dockerfile）
	$(SUBMIT_L3) /opt/work/jobs/m2_l3_hawkes.py

m2-l3: m2-l3-hawkes ## L3 机制层全链（当前仅 Hawkes）

# ── B2 M4 热榜纵向演化（08 §7）──
m4-onlist: ## M4 上榜追踪（dws_onlist_track；weekly_legacy 不做跨批差分——冻结决策）
	$(SUBMIT) /opt/work/jobs/m4_onlist.py

m4-track: ## M4 跨批轨迹（dws_metric_track；delta 按采集时刻记账 + 采集窗口倒挂排除）
	$(SUBMIT_M4) /opt/work/jobs/m4_track.py

m4-evolution: ## M4 演化 v2（dws_metric_track 轨迹分型 + dws_lifecycle 半衰期/饱和门控）
	$(SUBMIT_M4) /opt/work/jobs/m4_evolution.py

m4-up: ## M4 UP 档案（dws_up_profile）
	$(SUBMIT) /opt/work/jobs/m4_up.py

m4: m4-onlist m4-track m4-evolution m4-up ## M4 全链（跨批轨迹→演化分型 v2→生命周期）

# ── B2.3 服务层（08 §10：dws → PG ads 表；fastapi 重建见 docker compose build fastapi）──
# ── B4.2 M6：特征矩阵（日级）+ T1 消融训练（周级）（预注册 14 号）──
m6-features: ## M6 特征矩阵 ml_features（0..3 日窗截断口径；预注册 14 §3）
	$(SUBMIT_M6) /opt/work/jobs/m6_features.py

m6-train: ## M6 T1 四层消融（预注册 14 §5；容器 local[2]，~20-30min）
	docker exec spark-master /opt/spark/bin/spark-submit --master local[2] --driver-memory 2g /opt/work/jobs/m6_train.py

m6-panel: ## M6 截断面板（B4.3：构建 + 双切分实验；双 N/聚类稳健 SE）
	$(SUBMIT_M6) /opt/work/jobs/m6_panel.py
	docker exec spark-master /opt/spark/bin/spark-submit --master local[2] --driver-memory 2g /opt/work/jobs/m6_panel_train.py

m6: m6-features m6-train m6-panel ## M6 全链（特征+训练+面板）

# ── B5.1 性能实验（合成放大（重放+扰动）；套件自动等夜间链后运行）──
perf-suite: ## 启动性能套件（放大 1/4/8× + 密度/事件网格 + Hawkes 加速比 + 倾斜对照）
	setsid bash tools/perf_suite.sh < /dev/null > /dev/null 2>&1 &
	@echo "套件已后台启动；日志 logs/perf_suite_*.log；结果 data-out/perf_results.jsonl"

ads-build: ## 生产链 ADS 导出（11 张表 → PG；覆盖 3 页数据流 + M4）
	$(SUBMIT) /opt/work/jobs/ads_build.py

serve-rebuild: ## 重建并重启 fastapi 容器（改 serve/ 后执行；静态页 nginx 挂载无需重建）
	$(COMPOSE) build fastapi && $(COMPOSE) up -d --no-deps fastapi

# ── 日频增量链（B0.2 次日新批次自动到达；08 §0.1「滚动全量表每日全量重算」）──
m0-daily: ## 新批次增量装载（用法：make m0-daily BATCHES=20261012,20261013；只动指定批次分区）
	@test -n "$(BATCHES)" || (echo "用法：make m0-daily BATCHES=20261012,20261013" && exit 1)
	$(SUBMIT) /opt/work/jobs/m0_daily.py --batches $(BATCHES)

daily: ## 日频全链编排（DS 就位前手工版：守卫→装载→生产 dwd→对账→情感→L1→L2→L3→M4→ADS）
	@test -n "$(BATCHES)" || (echo "用法：make daily BATCHES=20261012,20261013" && exit 1)
	docker cp tools/probe_partition_guard.py spark-master:/tmp/guard.py
	@echo "── ① 装载前快照 ──"
	docker exec spark-master /opt/spark/bin/spark-submit --master local[2] /tmp/guard.py \
	  --tables ods_danmaku_full,ods_batch_meta,ods_hotlist_video \
	  --out /opt/work/data-out/guard_daily_before.json
	@echo "── ② 增量装载 ──"
	$(MAKE) m0-daily BATCHES=$(BATCHES)
	@echo "── ③ 守卫断言：旧分区只许原样、只许新增 ──"
	docker exec spark-master /opt/spark/bin/spark-submit --master local[2] /tmp/guard.py \
	  --tables ods_danmaku_full,ods_batch_meta,ods_hotlist_video \
	  --out /opt/work/data-out/guard_daily_after.json \
	  --compare-with /opt/work/data-out/guard_daily_before.json --assert-no-changes
	$(MAKE) m0-full m1 m5
	$(MAKE) m2-l1
	$(MAKE) m2-l2
	$(MAKE) m2-l3
	$(MAKE) m4
	-$(MAKE) m6-features
	$(MAKE) ads-build

m5: ## M5 词典 v0：9 情形打分+锚词+词典表+情感曲线（dwd_danmaku_sent / dict_* / dws_sentiment_curve_1pct）
	$(SUBMIT) /opt/work/jobs/m5_sentiment_v0.py

b1: m0-full m1 m5 ## B1 全链（三段串跑；每段各自验收后 commit）

parity: ## 对拍验收：曲线101点+8维 vs goldens + 池级锚点
	$(SUBMIT) /opt/work/jobs/parity_check.py

replay: ## 幂等验收：全链 ETL 两次 → 规范导出比对（整数精确/浮点≤1e-9）+ 对拍
	# ⚠️ 清理必须经容器（canonical 产物由容器 root 写出，宿主 rm 会"权限不够"——2026-10-13 实测）
	$(COMPOSE) exec -T spark-master rm -rf /opt/work/data-out/canonical_run1 /opt/work/data-out/canonical_run2
	$(SUBMIT) /opt/work/jobs/m0_min.py --batches $(LEGACY_BATCHES) && $(SUBMIT) /opt/work/jobs/m2_l1_min.py && $(SUBMIT) /opt/work/jobs/ads_min.py
	mv data-out/canonical data-out/canonical_run1
	$(SUBMIT) /opt/work/jobs/m0_min.py --batches $(LEGACY_BATCHES) && $(SUBMIT) /opt/work/jobs/m2_l1_min.py && $(SUBMIT) /opt/work/jobs/ads_min.py
	mv data-out/canonical data-out/canonical_run2
	python3 tools/compare_canonical.py data-out/canonical_run1 data-out/canonical_run2
	$(SUBMIT) /opt/work/jobs/parity_check.py
	@echo "✅ 幂等验收通过：全链两跑一致（整数精确 / 浮点≤1e-9）+ 对拍 PASS"
	@echo "CHAIN_REPLAY_OK"

collect: ## 手动触发一次日频采集（Top200，约 2-4 小时）
	bash collect/bili_daily.sh

cron-install: ## 安装日频 crontab（01:00 采集 + 每小时兜底守卫——覆盖 VM 挂起漏跑）
	(crontab -l 2>/dev/null | grep -v daily_cron; \
	 echo '0 1 * * * cd $(CURDIR) && bash collect/daily_cron.sh >> logs/cron_daily.log 2>&1'; \
	 echo '17 * * * * cd $(CURDIR) && bash collect/daily_cron_guard.sh >> logs/cron_daily.log 2>&1'; \
	 echo '@reboot sleep 600 && bash $(CURDIR)/collect/daily_cron_guard.sh >> $(CURDIR)/logs/cron_daily.log 2>&1') | crontab -
	crontab -l | tail -3

goldens: ## 生成对拍基准（宿主机旧 venv，只读旧仓库）+ 上传 HDFS /opt/goldens（集群 parity 实际读取源）
	$(VENV) tools/export_goldens.py --old-root ~/projects/danmaku --out tests/goldens
	$(COMPOSE) exec -T namenode hdfs dfs -mkdir -p /opt/goldens
	$(COMPOSE) exec -T namenode hdfs dfs -put -f - /opt/goldens/goldens_curve.csv < tests/goldens/goldens_curve.csv
	$(COMPOSE) exec -T namenode hdfs dfs -put -f - /opt/goldens/goldens_metrics.csv < tests/goldens/goldens_metrics.csv

dev-etl: ## 本地模式全链（不起 Docker，file:// 路径——开发迭代用）
	$(VENV) jobs/m0_min.py --local --batches $(LEGACY_BATCHES) --raw-root file://$(CURDIR)/data/raw-local --wh-root file://$(CURDIR)/data-out/dev-wh
	$(VENV) jobs/m2_l1_min.py --local --wh-root file://$(CURDIR)/data-out/dev-wh
	$(VENV) jobs/parity_check.py --local --wh-root file://$(CURDIR)/data-out/dev-wh --goldens-dir $(CURDIR)/tests/goldens

clean:
	rm -rf data-out/dev-wh data-out/canonical*
