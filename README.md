# DanmakuPulse-v2：基于 Spark 的 B 站热榜视频弹幕互动节奏分析系统

DanmakuPulse-v2 是 B 站热榜视频弹幕互动节奏分析系统的第二版实现。系统将采集数据组织为 ODS/DWD/DWS/ADS 分层数据仓库，以 Spark 与 Hive Metastore 承载计算，以预研系统的 NumPy 实现为数值基准完成对拍与重跑幂等验收，并通过受登录门禁保护的只读 API 与静态页面输出分析结果。

## 1 摘要

弹幕（danmaku）时间序列记录了观众在视频播放过程中的集体反应，其密度峰值、同步性与时间滞后结构可作为高能片段识别、分区比较与热榜视频演化分析的依据。预研系统以 NumPy 实现唯一算法口径（SSOT），指标计算依赖 PySpark 上的 Python UDF，对拍基准由宿主侧 NumPy 重算生成。本版把数据流拆分为采集、原始层（RAW）、贴源层（ODS）、明细层（DWD）、汇总层（DWS）与应用层（ADS），将指标实现迁移为 Spark SQL 原生表达式与按视频并行的数据并行算子，覆盖清洗、对账、密度与指纹、节奏事件、同步性、锚词滞后、Hawkes 拟合、热榜演化、情感打分与特征矩阵等作业；迁移正确性以「与预研实现数值一致」为判据，密度曲线 101 点与浮点指标容差为 1e-9，计数类指标零容差，池级锚点逐数对账。系统配套 9 个容器服务的运行底座、37 张 Hive 外部表注册、PostgreSQL 应用层以及 FastAPI 只读服务。当前节奏事件检测的权重与截断阈值尚未依据人工标注标定，Hawkes 拟合在现有数据上未通过残差检验，相应结论仅具探索性质。

在此之上，系统以同一份 DWD 明细增加了秒级的「精彩时刻检测」：以 5 秒桶统计同文本复读量并定界情绪共振段，以质疑词词典、文本长度与情感分三重条件筛出质疑段，按分区基准给出共振强度百分位；长视频（>1800 秒）输出切片建议，短视频输出高光定位。该模块只刻画观众反应的两个侧面（共振与质疑），不评价内容质量；不判定细粒度情绪（弹幕情绪词语义随内容语境漂移），只给「哪里起了反应、观众在刷什么、是否在质疑」三类可观测信号，并把原始弹幕直接展示给读者。相关实现见 [jobs/highlight_core.py](jobs/highlight_core.py)、[jobs/m2_l4_highlights.py](jobs/m2_l4_highlights.py) 与 [serve/highlights.py](serve/highlights.py)。

## 2 设计目标与范围

- **数值等价**：以预研仓库 `analysis/danmaku_metrics.py` 为唯一算法口径（SSOT），新实现须通过对拍验收；池级锚点与容差规则保留在 [config/params.yaml](config/params.yaml) 与 [jobs/parity_check.py](jobs/parity_check.py) 中。
- **分层可复算**：RAW → ODS → DWD → DWS/ML → ADS 逐层落盘，任何一层均可由上一层重建。
- **幂等**：所有作业采用分区覆盖写或全量重建，重跑结果满足整数精确、浮点容差 1e-9 的一致口径。
- **链隔离**：对拍链固定于历史 4 个批次（`LEGACY_BATCHES`），日频新批次只进入生产链，两者不混合，避免锚点失效。
- **服务化**：DWS/ADS 结果经 PostgreSQL 暴露为只读接口与页面，接口统一返回 `{code, data, meta}`。
- **秒级可定位**：在整段节奏指纹之外，把弹幕反应定位到秒级（5 秒桶），共振段与质疑段分别输出并接入同分区参照系；长/短视频走不同产品分支（切片建议 / 高光定位）。诚实边界写入接口与界面，不宣称内容质量。
- **可增量**：日频批次按 `batch_date` 增量装载，配合分区守卫脚本检查旧分区不被改写。

范围说明：代码头注中出现的「08 §x」「10 §x」「11 §x」等编号，指向预研仓库《新系统重构》系列设计文档；这些文档未随本仓库分发，以下内容只以本仓库的代码、配置与快照为依据。

## 3 数据与方法

### 3.1 数据来源与批次

弹幕与榜单数据由 `collect/bili.py` 调用 B 站公开热门榜与弹幕接口采集，按批次保存弹幕 CSV、批次元数据、榜单快照与视频级采集时刻。历史批次的血缘与已知问题集中在 [config/batch_lineage.json](config/batch_lineage.json)：

| 批次 | 采集节奏 | 采集覆盖 | 榜单快照 | 已知问题 |
| :--- | :--- | :--- | :--- | :--- |
| 20260902 | weekly_legacy | 91/100 | 永久丢失 | `list_fetched_at` 为推定值 |
| 20260907 | weekly_legacy | 118/200 | 永久丢失 | 覆盖率不足 |
| 20260914 | weekly_legacy | 200/200 | 有 | 排名尾部 87 个视频从未采集，存在系统性头部偏差；跨 2.5 天三窗口，非单点快照 |
| 20260917 | weekly_legacy | 100/100 | 有 | 榜单 200 条，采集深度 100 |

日频批次自 20261012 起由 `jobs/m0_daily.py` 增量装载，采集时刻由 `fetch_times.csv` 侧车文件记录；跨批差分按采集时刻而非批次日期记账。

### 3.2 基础口径与池级锚点

对拍链的池级锚点定义于 [config/params.yaml](config/params.yaml) 的 `anchors` 节，并在 [jobs/parity_check.py](jobs/parity_check.py) 中以常量固化：

| 锚点 | 数值 | 口径 |
| :--- | ---: | :--- |
| 视频批次数 | 509 | 91 + 118 + 200 + 100 |
| DWD 弹幕总行数 | 672,439 | 历史 4 批清洗后明细行数 |
| 池内视频数 | 508 | 每个 bvid 取最新批次画像 |
| 池内弹幕总量 | 669,995 | 池内 SUM(danmaku_count) |
| 清洗保留率 | 99.3364% | 672,439 / 676,931 |

两套清洗口径并存且互相隔离：对拍链的 `m0_min` 沿用预研系统的去重键 `(bvid, text, time_offset)`；生产链的 `m0_full` 使用 `(bvid, cid, row_id)`。实验脚本 [tools/probe_dedup_keys.py](tools/probe_dedup_keys.py) 与 params 配置的注释记录，旧键会误删 4,492 行「同文同刻但 row_id 唯一」的真实复读弹幕（3,021 组），新键在批内不产生误删。

### 3.3 计算流程

1. **M0 清洗**（`jobs/m0_min.py`、`jobs/m0_full.py`、`jobs/m0_daily.py`）：CSV 读取与类型转换生成 ODS；DWD 做去重、广告/无意义/超短打标、属性加权、发送时间与滞后分桶，并输出刷量证据表与视频维度表。
2. **M1 对账**（`jobs/m1_reconcile.py`）：以 `reported`、`got`、`dwd 行数` 三值对账，按证据优先级归因（多 P、截断、删除、字幕池、残差），输出批次质量分与偏差检验。
3. **L1 体积层**（`jobs/m2_l1_density.py`、`m2_l1_fingerprint.py`、`m2_l1_baseline.py`、`m2_l1_cluster.py`）：5 秒桶与 1% 桶密度；16 维节奏指纹与分区基准；K-means++ 与 K-shape 两法聚类。
4. **L2 结构层**（`jobs/m2_l2_events.py`、`m2_l2_sync.py`、`m2_l2_lag.py`、`m2_l2_novelty.py`、`m2_l2_finalize.py`）：四检测器融合的节奏事件、同步性与羊群权重、锚词滞后校正、Novelty 分段，并把结果回填 L1 指纹。
5. **L3 机制层**（`jobs/m2_l3_hawkes.py`）：逐视频拟合指数核 Hawkes 过程（L-BFGS-B），以时间变换残差 KS 检验作为模型通过条件，未通过时按设计降级为幂律拟合。
6. **M4 热榜演化**（`jobs/m4_onlist.py`、`m4_track.py`、`m4_evolution.py`、`m4_up.py`）：上榜追踪、按采集时刻记账的跨批指标轨迹、轨迹分型与生命周期、UP 主档案。
7. **M5 文本与情感**（`jobs/m5_sentiment_v0.py`）：9 情形规则统一为「否定取反、程度乘子、极性乘强度」的打分公式，输出情感分表、词典表与 1% 桶情感曲线。
8. **M6 特征与实验**（`jobs/m6_features.py`、`m6_train.py`、`m6_panel.py`、`m6_panel_train.py`）：0–3 日窗截断特征矩阵、四层消融训练、截断面板与多种稳健统计检验。
9. **ADS 与服务**（`jobs/ads_min.py`、`jobs/ads_build.py`、`serve/app.py`）：DWS 结果导出到 PostgreSQL，由 FastAPI 只读接口与 13 个静态页面消费。
10. **L4 精彩时刻**（`jobs/highlight_core.py`、`jobs/m2_l4_highlights.py`、`jobs/ads_highlights.py`）：5 秒桶复读聚集定界情绪共振段、质疑词+长度+情感分筛质疑段、视频级从众/中性/批判画像与分区对标；长视频出切片建议、短视频出高光定位。

### 3.4 关键算法与参数

参数集中在 `config/params.yaml`，各作业以文件头注记录实现口径。主要参数如下：

- **L1 密度曲线**：101 个等宽进度桶；高斯平滑 σ=1.0（9-tap 核，reflect 边界）；峰值判定高度为 μ+2σ，等同 scipy `find_peaks(height=μ+2σ, distance=n//50)` 的语义。
- **L2 事件检测**：5 秒网格（不做平滑）；D1 前缀 P70 自适应阈值加三点斜率门限；D2 密度门限加变化率门限；D3 熵阈值加突出度与峰距约束；D4 因果窗谱通量。四法按权重 `[0.3, 0.3, 0.2, 0.2]` 融合后以 θ=0.5 截断，跨方法候选按 IoU>0.3 聚合。`calibrated_on` 字段当前记为 `uncalibrated`。
- **L2 同步性**：事件峰窗内弹幕位置熵与跨天熵构成 `sync_score`，以 1000 次置换检验给出 `sync_p`，随机种子由 `crc32(bvid|batch_date)` 固定，保证重跑幂等。
- **L2 锚词滞后**：以回溯型锚弹幕相对事件中心的偏移中位数估计分区级 δ，仅当分区锚弹幕数不少于 30 时输出。
- **L2 分段**：窗口 10 桶的余弦自相似矩阵与 2k×2k 棋盘核（k=8）计算 novelty，取局部极大作为分段边界。
- **L3 Hawkes**：观测窗口为 `[pubdate, min(末事件, pubdate+21d)]`，事件数不少于 1000 才拟合；参数含 μ0/μ1、分支比 α/β 与半衰期；KS 检验 p>0.05 记为通过，否则 `method=powerlaw`。
- **M6 防泄漏**：特征只取 `pubdate ≤ send_time < pubdate+4d` 的弹幕，标签 `y_growth_v2 = log1p(c[0..7]) − log1p(c[0..3])`，样本池要求观测天数不小于 8 天且首日窗内弹幕非空。
- **L4 精彩时刻**：5 秒桶内同一归一化文本复读 ≥3 次计入复读；桶复读量 `n_repeat = Σ_text max(n−1,0)`，峰要求 `s(b) ≥ 3` 且同文本 `n ≥ 3` 的文本数 ≥1；以峰为中心按峰值 25% 回落成段，相邻峰谷值高于较小峰 50% 时合并为同一爆发区（沿用盲标指南 v3 的量化约定）。质疑弹幕需命中质疑词、长度 ≥8 且 M5 情感分 <0；候选桶 `n_crit ≥ 2`、段总量 ≥3 才保留（宁缺毋滥）。长/短视频分界 1800 秒，分区基准最小样本 20。

## 4 系统架构与模块

### 4.1 服务底座

[compose/docker-compose.b0.yml](compose/docker-compose.b0.yml) 定义 9 个服务；按 `mem_limit` 求和，容器内存上限合计 4,800 MiB。

| 服务 | 镜像/构建 | 作用 | 对外端口 |
| :--- | :--- | :--- | :--- |
| namenode | bde2020/hadoop-namenode 3.2.1 | HDFS 名称节点 | 9870 |
| datanode | bde2020/hadoop-datanode 3.2.1 | HDFS 数据节点（单副本） | — |
| postgres | postgres:15-alpine | Hive 元库与 ads 库 | 5433 |
| hive-metastore | 复用 Spark 镜像内的 Hive 2.3.9 依赖 | 元数据服务 | 内部 9083 |
| spark-master | danmaku-spark:3.5.4-py | Spark standalone 主节点与 driver | 8080 |
| spark-worker | danmaku-spark:3.5.4-py | 单个 executor（1g 内存、2 核） | 8081 |
| spark-history | danmaku-spark:3.5.4-py | 事件日志历史服务 | 18080 |
| fastapi | serve/Dockerfile（python:3.11-slim） | 只读 API | 8000（经 nginx 反代） |
| nginx | nginx:alpine | 唯一入口：静态页、登录门禁、/api 反代 | 8088（HTTP 与 TLS 同端口） |

Spark 镜像由 [compose/assets/spark-docker/Dockerfile](compose/assets/spark-docker/Dockerfile) 在 `apache/spark:3.5.4-java17` 基础上加装 python3、PyYAML、NumPy 与 SciPy；构建步骤未封装进 Makefile，首次运行前需自行构建并命名为 `SPARK_IMAGE` 指定的镜像。

### 4.2 数据分层与数据流

```
采集（collect/bili.py）
  → data/cli_collected/（CSV + meta.json + hotlist_videos_*.json + fetch_times_*.csv）
  → tools/stage_raw_local.py → data/raw-local/batch_date=*/（批次映射唯一事实源）
  → compose/assets/ingest_raw.sh → HDFS /raw/batch_date=*/
  → m0_min / m0_daily → ODS；m0_full → DWD
  → m1 / m2_l1_* / m2_l2_* / m2_l3_hawkes / m4_* / m5 / m6_* → DWS 与 ML 表
  → m2_l4_highlights → L4 DWS（共振峰/质疑段/反馈画像/质疑词表）
  → ads_min / ads_build / ads_highlights → PostgreSQL ads
  → FastAPI → nginx :8088 → 静态页面
```

| 层级 | 存储位置 | 主要表 | 写入作业 | 入口命令 |
| :--- | :--- | :--- | :--- | :--- |
| RAW | HDFS `/raw/batch_date=*/` | danmaku.csv、meta.json、hotlist.json、fetch_times.csv | ingest_raw.sh | `make ingest` |
| ODS | `/warehouse/ods_*` | ods_danmaku_full、ods_batch_meta、ods_hotlist_video、ods_video_fetch_times | m0_min、m0_daily | `make etl`、`make m0-daily` |
| DWD | `/warehouse/dwd_*` | dwd_danmaku_clean(_min)、dwd_video_metric_snapshot、dwd_spam_flag、dwd_danmaku_sent、dim_video | m0_full、m5 | `make m0-full`、`make m5` |
| DWS | `/warehouse/dws_*` | 密度、指纹、分区基准、事件、同步、滞后、分段、聚类、Hawkes、上榜轨迹、生命周期、UP 档案、批次对账；L4：共振峰/质疑段/反馈画像/质疑词表 | L1/L2/L3/M4/M1 作业与 L4 | `make m2-l1`、`make m2-l2`、`make m2-l3`、`make m4`、`make m1`、`make m2-l4-highlights` |
| ADS | PostgreSQL `ads` 库 | 19 张 ads_* 表：3 张对拍链产物（video_metrics、video_curve、batch_meta）+ 12 张既有生产链产物 + 4 张 L4 新表（resonance_peaks、critique_segments、video_feedback_profile、dict_critique） | ads_min、ads_build、ads_highlights | `make ads-build`、`make ads-highlights` |
| 词典/ML | `/warehouse` | dict_sentiment、dict_anchor、dict_meme、dict_emoticon、ml_features、ml_panel | m5、m6 | `make m5`、`make m6` |

Hive Metastore 由 `jobs/register_hms.py` 注册 41 张外部表：25 张 `batch_date` 分区表（MSCK REPAIR 同步分区）与 16 张平铺表（含 L4 的 4 张新表）；注册采用 `DROP + CREATE` 以刷新列变更，外部表删除不删除底层数据。

### 4.3 作业层模块

| 模块 | 作业文件 | Makefile 入口 | 主要输出 |
| :--- | :--- | :--- | :--- |
| M0 最小版 | m0_min.py | `make etl` 内第一步 | ods_danmaku_full、dwd_danmaku_clean_min、ods_batch_meta、ods_hotlist_video |
| M0 完整版 | m0_full.py | `make m0-full` | dwd_danmaku_clean、dim_video、dwd_video_metric_snapshot、dwd_spam_flag |
| M1 对账 | m1_reconcile.py | `make m1` | dws_batch_audit、dws_reconcile_detail、PG ads_quality_report |
| L1 体积层 | m2_l1_density.py、m2_l1_fingerprint.py、m2_l1_baseline.py | `make m2-l1` | dws_density_5s、dws_density_1pct、dws_video_rhythm_features、dws_partition_rhythm |
| L1 聚类 | m2_l1_cluster.py、tools/kshape_validate.py | `make m2-l1-cluster` | dws_curve_clusters |
| L2 结构层 | m2_l2_events.py、m2_l2_sync.py、m2_l2_lag.py、m2_l2_novelty.py、m2_l2_finalize.py | `make m2-l2` | dws_rhythm_events、dws_sync_features、dws_lag_calibration、dws_segments |
| L3 机制层 | m2_l3_hawkes.py | `make m2-l3` | dws_hawkes_params |
| L4 精彩时刻 | highlight_core.py、m2_l4_highlights.py、ads_highlights.py | `make m2-l4-highlights`、`make ads-highlights` | dws_resonance_peaks、dws_critique_segments、dws_video_feedback_profile、dict_critique；PG 4 张 ads_* 表 |
| M4 纵向演化 | m4_onlist.py、m4_track.py、m4_evolution.py、m4_up.py | `make m4` | dws_onlist_track、dws_metric_track、dws_video_daily、dws_evolution_feats、dws_lifecycle、dws_up_profile |
| M5 情感 | m5_sentiment_v0.py | `make m5` | dwd_danmaku_sent、dict_*、dws_sentiment_curve_1pct |
| M6 特征与实验 | m6_features.py、m6_train.py、m6_panel.py、m6_panel_train.py | `make m6` | ml_features、ml_panel、ml_metrics.csv、ml_panel_metrics.csv |
| ADS 与服务 | ads_min.py、ads_build.py、register_hms.py | `make ads-build`、`make hms` | PostgreSQL ads 表与 Hive 外部表 |

### 4.4 服务层

[serve/app.py](serve/app.py) 提供 FastAPI 只读服务，共 50 个路由，统一响应结构为 `{code, data, meta}`；接口覆盖批次、视频、分区、节奏事件、特征指纹、诊断卡、精彩时刻、在线诊断与后台管理等。页面门禁分两层：nginx 通过 `auth_request` 校验会话，FastAPI 对 `/api/*` 返回 401/403 JSON；会话为 HMAC-SHA256 签名 cookie，口令为 scrypt 哈希。静态页共 13 个 HTML 文件，`/api/health`、`/api/anchors` 等验收接口保留。

首次启动后需创建管理员账号，命令见 [serve/seed_admin.py](serve/seed_admin.py) 头注：

```bash
docker exec fastapi python seed_admin.py
# 或指定用户名与口令
docker exec -e ADS_ADMIN_PASSWORD=<password> fastapi python seed_admin.py
```

## 5 主要结果

### 5.1 对拍与重跑幂等

B0 验收由 `make smoke`、`make etl`、`make parity`、`make replay` 组成，判定条件写在 [jobs/parity_check.py](jobs/parity_check.py) 与 [tools/compare_canonical.py](tools/compare_canonical.py) 中：

| 检查组 | 内容 | 容差 |
| :--- | :--- | :--- |
| A 曲线 | 每个 (bvid, batch_date) 的 101 点密度曲线对比预研实现导出的 goldens | 逐点绝对差 ≤1e-9 |
| B 指标 | 8 维节奏指标与 goldens 重算值 | peak_count、peak_fwhm、danmaku_count 精确一致；其余浮点维 ≤1e-9 |
| C 池级锚点 | 509 个视频批次、672,439 行 DWD、508 个池内视频、669,995 条池内弹幕 | 逐数精确一致 |

三项全部通过才判定 parity 成功；重跑幂等采用相同容差：整数列精确一致，浮点列绝对差 ≤1e-9。比对不使用 md5，原因是 Spark 聚合的浮点结果位级受 shuffle 顺序影响，容差内一致才是可复现口径。

### 5.2 性能实验结果

[serve/v3data/perf.json](serve/v3data/perf.json) 是 `data-out/perf_results.jsonl` 的前端快照，记录 2026-10-13 的 12 条测量结果。合成放大数据由重放加扰动构造，放大倍率与输出行数近线性：

| 实验 | 放大倍率 | 输出规模 | 并行度 1 墙钟 | 并行度 2 墙钟 |
| :--- | ---: | ---: | ---: | ---: |
| 密度引擎 | 1× | 209,656 行 | 15.2 s | 9.2 s |
| 密度引擎 | 4× | 838,663 行 | 20.1 s | 12.7 s |
| 密度引擎 | 8× | 1,674,671 行 | 26.9 s | 15.1 s |
| 事件检测 | 1× | 557 个事件 / 888 条序列 | — | 4.4 s |
| 事件检测 | 8× | 5,374 个事件 / 7,104 条序列 | — | 14.5 s |

Hawkes 逐视频拟合的串行耗时为 25.5 s 与 25.0 s，2 路并行分别为 17.5 s 与 16.6 s，加速比 1.46 与 1.51。倾斜对照中普通分区与加盐分区的墙钟分别为 2.5 s 与 1.2 s。需要说明的限制：集群只有 1 个 Spark worker，`spark-submit` 固定 2 个 executor 核，因此并行度 4 无法测量；数据放大为合成数据，上述数字反映规模与并行行为，不代表真实流量承诺；数据量放大 7.99 倍时单并行墙钟放大 1.77 倍，墙钟呈亚线性。

### 5.3 数据快照

[serve/v3data/meta.json](serve/v3data/meta.json) 记录 2026-10-15 14:11:54 生成的前端快照：`n_videos`=888、`n_curve`=888、`n_partitions`=76、`n_events`=551（均有代表性弹幕），主视图 11 个维度、形状维度 6 个，最小分区样本数 20。该快照由 [jobs/export_v3_snapshot.py](jobs/export_v3_snapshot.py) 从 `/warehouse` 现有表只读导出，字段口径以文件内的 `source_tables` 与 `z_formula` 为准。

### 5.4 精彩时刻检测结果

`make m2-l4-highlights` 在全部分析口径弹幕（`is_spam=false`，1,115,185 行 / 888 个视频-批次）上构建 L4 表，`make ads-highlights` 导出 PostgreSQL：

| 指标 | 数值 |
| :--- | ---: |
| 覆盖视频-批次 / 覆盖率 | 888 / 100% |
| 共振段总数 / 检出视频-批次 | 2,807 段 / 757 个（85.25%） |
| 平均共振段数（每视频-批次） | 3.16 |
| 片头 60 秒内共振段占比 | 633 / 2,807（22.55%） |
| 质疑段总数 / 检出视频-批次（strict 阈值） | 336 段 / 230 个（25.90%，未触发 <5% 放宽规则） |
| 从众复读 / 中性 / 认真批判（弹幕条数占比） | 525,135（47.09%）/ 576,171（51.67%）/ 13,879（1.24%） |
| 长视频样例（BV1GjYe6KEGt，25,205 秒） | 21 个共振段 / 3 个质疑段 / 分区 23 支 / 共振强度 pctl 95.65 |
| 短视频样例（BV1DnYe6aEjn，154 秒） | 2 个共振段 / 未检出质疑段 / 高光定位 2 条（含 1 条片头回落） |
| 作业耗时（cluster 模式，两轮） | 73.51 s / 57.51 s；两轮 4 表内容级指纹一致 |
| 测试 | `pytest tests/ -q` 113 passed；`tests/accept_auth.sh` PASS=75 / FAIL=0 |

需要说明的限制：共振与质疑都是**观众反应**侧的代理指标，不代表内容质量；系统不判定细粒度情绪（语义随语境漂移）；质疑层宁缺毋滥，未检出即未检出；复读以文本完全相等为近似，近义变体可能漏检。完整边界见 `docs/` 同级设计文档（预研仓库 35 号）。

## 6 项目结构

```
collect/    采集 CLI 与日频包装脚本（bili.py 为预研仓库零改动副本）
compose/    9 服务 compose 文件、Hadoop/Spark/HMS 配置、nginx 配置、Spark 镜像 Dockerfile
config/     params.yaml（参数中心）、batch_lineage.json（批次血缘）、dicts/（情感、锚词与质疑词表 critique_words.tsv）
jobs/       49 个 Python 脚本：Spark 作业、共享纯函数、对拍与 HMS 注册
serve/      FastAPI 服务（50 路由，含精彩时刻聚合端点）、13 个静态页、v3data/ 前端快照
tests/      单元测试与接口契约测试；goldens/ 为对拍基准 CSV
tools/      52 个文件（含 45 个 Python 脚本）：对拍基准生成、诊断探针、评估与性能测试
Makefile    底座、ETL、对拍、分析链与日频链的统一入口
docs/       人工标注指南（annotation_guides.md）
```

## 7 环境与复现

### 7.1 环境前提

- 宿主为 Linux，安装 Docker 与 Docker Compose v2；所有集群作业经 `spark-master` 容器内的 `spark-submit` 执行。
- 代码中存在固定的路径假设：仓库位于 `~/projects/danmaku-v2`，预研仓库位于 `~/projects/danmaku`，宿主虚拟环境为 `~/danmaku-venv`。这些路径出现在 `Makefile`、`collect/daily_cron_guard.sh`、`tools/perf_suite.sh` 与测试命令中。
- 数据、基准与构建产物不进 git：`data/`、`data-out/`、`logs/`、`lib/*.jar`、`serve/static/vendor/`、`compose/assets/tls/` 均由 [.gitignore](.gitignore) 排除。历史批次数据与 goldens 的生成依赖预研仓库与本机虚拟环境，仓库本身不携带这些数据。
- 宿主 venv 需包含 PySpark、NumPy、SciPy 与 pytest；形态聚类的宿主段另需 tslearn 与 scikit-learn。

### 7.2 首次部署与 B0 验收

```bash
make setup                 # 下载 PostgreSQL JDBC、Disruptor、ECharts 与 html2canvas
docker build -t danmaku-spark:3.5.4-py compose/assets/spark-docker
make seed-data             # 从预研仓库复制历史 4 批采集数据
make stage                 # data/cli_collected → data/raw-local
make up                    # 生成自签证书并启动 9 个服务
make goldens               # 生成对拍基准并上传 HDFS /opt/goldens
make ingest                # data/raw-local → HDFS /raw
make smoke                 # HDFS/PG 读写冒烟与 Spark master 探活
make etl                   # 对拍链：m0_min → m2_l1_min → ads_min
make parity                # 曲线、8 维指标与池级锚点对拍
make replay                # 全链两跑一致性比对并再次对拍
```

### 7.3 生产链与分析链

```bash
make m0-full               # M0 完整版清洗与刷量检测
make m1                    # 三值对账与质量报告
make m5                    # 情感词典 v0
make m2-l1                 # L1 密度、指纹与分区基准
make m2-l2                 # L2 事件、同步、滞后、分段与回填
make m2-l3                 # L3 Hawkes 拟合
make m2-l4-highlights      # L4 精彩时刻：共振段 / 质疑段 / 视频反馈画像（DWS）
make m4                    # M4 上榜、轨迹、演化与 UP 档案
make m6                    # M6 特征矩阵、消融训练与截断面板
make ads-build             # DWS/ADS → PostgreSQL
make ads-highlights        # L4 DWS → PostgreSQL（4 张新 ads_* 表）
make hms                   # Hive Metastore 外部表注册
```

补充说明：`make m2-l1-cluster` 依赖宿主 venv 中的 tslearn 与 scikit-learn，并分三步完成 K-means++、K-shape 与结果合并；`make l2-candidates` 与 `make l2-eval TRUTH=<标注文件>` 用于事件候选导出与人工真值评估，需先完成标注；`make dev-etl` 为不起 Docker 的本地 `file://` 模式，仅用于开发迭代。

### 7.4 日频增量与运维

```bash
make collect                              # 手动触发一次 Top200 采集（约 2-4 小时）
make cron-install                         # 安装 01:00 采集与每小时兜底补跑
make m0-daily BATCHES=20261012,20261013   # 指定批次增量装载 ODS
make daily BATCHES=20261012,20261013      # 守卫、装载、生产链、ADS 全链
make serve-rebuild                        # 修改 serve/ 后重建 FastAPI 容器
make perf-suite                           # 后台启动性能实验套件
```

`make daily` 先用 `tools/probe_partition_guard.py` 记录装载前快照，再以 `--assert-no-changes` 校验旧分区未被改写，随后串行执行 `m0-full`、`m1`、`m5`、L1、L2、L3、M4、`m6-features` 与 `ads-build`。采集与补跑脚本中的路径按 `~/projects/danmaku-v2` 硬编码，迁移部署目录时需同步修改。

## 8 测试与验证

[tests/](tests/) 目录共 11 个测试文件、113 个测试用例，按类型分为三组：

| 分组 | 文件数 | 测试函数 | 覆盖内容 | 运行前提 |
| :--- | ---: | ---: | :--- | :--- |
| 算法纯函数 | 6 | 74 | L2 四检测器与融合、L3 Hawkes 合成数据拟合、M4 轨迹规则、L2 峰评估的 IoU/匹配/AP、fetch_times 侧车读写、精彩时刻核心（峰检测/定界/合并/质疑加权/三分类） | 宿主 venv 的 NumPy 与 SciPy |
| 接口契约 | 2 | 16 | 端点状态码、统一响应结构、关键字段与 404 负例；精彩时刻端点长/短视频分支、片头回落、旧路由不冲突 | 服务在线，且存在可用账号，无服务时整组 skip |
| 服务与前端守卫 | 3 | 23 | 设计令牌一致性、页面功能标题与措辞规范、调用限额与默认关闭约束 | 其中设计令牌测试需通过 `DESIGN_MD` 指向预研仓库的 DESIGN.md |

运行命令：

```bash
~/danmaku-venv/bin/python -m pytest tests/ -q
bash tests/accept_auth.sh http://127.0.0.1:8088     # 登录与门禁机械验收
```

`tests/test_api_contract.py` 从环境变量 `API_USER`/`API_PASS` 读取账号，默认账号口令与 `tests/accept_auth.sh` 中的默认值一致；若用 `serve/seed_admin.py` 随机生成管理员口令，需要通过环境变量覆盖或重新建号。`tools/e2e_pages.py` 是早期页面 E2E 脚本，其页面清单与绝对路径已与当前静态页不一致，使用前需按现页面集调整。

## 9 实现状态与未完成项

以下状态按仓库最新提交中的代码、配置与文档核对；更细的逐批次结果记录在 `data-out/` 报告里，这些报告不随仓库分发。

| 模块 | 状态 | 依据 |
| :--- | :--- | :--- |
| 服务底座与对拍链 | 已实现 | compose 9 服务、`make smoke`/`etl`/`parity`/`replay` |
| B1 生产链 | 已实现 | `m0_full`、`m1_reconcile`、`m5_sentiment_v0`、`register_hms` |
| L1 体积层与聚类 | 已实现 | 三个 L1 作业、K-means++ 与宿主 K-shape 合并流程 |
| L2 结构层 | 已实现，未标定 | `params.yaml` 的 detect 权重与 θ 为默认值，事件表 `calibrated_on='uncalibrated'` |
| L3 机制层 | 已实现，负结果 | `m2_l3_hawkes.py` 记录当前数据上 KS 未通过、按设计降级为幂律 |
| M4 纵向演化 | 已实现 v2 | v1 口径因弹幕接口滑窗而作废，v2 改用可观测指标 |
| M5 情感 | 已实现 v0 | 词典与规则打分完成；人工一致性验证未完成 |
| M6 特征与面板 | 已实现 | 特征、消融、面板三组作业与预注册协议 |
| L4 精彩时刻 | 已实现 | `m2_l4_highlights.py` + `ads_highlights.py` + `/api/video/{bvid}/highlights`；诚实边界写入接口与页面 |
| ADS 与服务层 | 已实现 | 15 张生产 ADS 表、50 个 API 路由、13 个静态页面 |
| 人工标注 | 未开始 | [docs/annotation_guides.md](docs/annotation_guides.md) 进度表：高能片段 0/30、情感 0/1000 |

尚未完成的关键项：30 个视频的高能片段人工标注与 1000 条双人情感标注均未启动，因此 L2 融合权重的网格搜索标定、融合 F1 不低于最佳单法的验收条件、M5 的 κ 与 macro-F1 均无法给出结论；L3 的拟合通过率与分支比只具诊断意义。

## 10 局限性与适用范围

- **样本与覆盖**：历史数据只有 4 个周频批次，其中 20260902 与 20260907 的榜单快照永久丢失，20260914 的排名尾部 87 个视频从未采集，存在系统性头部偏差；20260917 只采前 100。基于历史 4 批的池级锚点与对拍结论不能外推到全站热榜。
- **采集口径**：B 站弹幕接口对饱和视频返回固定容量的最近 N 条滑动窗，`danmaku_got` 在部分视频上呈量子化并小于报告值，跨批水平与形状不能直接比较。M4 v2 因此改用播放、点赞、投币、收藏、分享、评论、报告弹幕量与排名等可观测指标，并把采集时刻倒挂的批次对排除出差分。
- **L2 结论未经人工真值验证**：检测器权重、融合截断阈值与事件定义均为实现默认值，未用人工标注网格搜索；融合 F1 是否优于最佳单法尚未验证，事件数量与位置只能作为探索性结果使用。
- **L3 模型失配**：现有数据上 Hawkes 时间变换残差的 KS 检验未通过，代码按设计降级为幂律方法；降级行的 Hawkes 参数仅作诊断。代码头注记录了同秒并列比例 8%–22%、残差过散布等失配证据，该结论限于当前数据口径。
- **精彩时刻检测的边界**：共振与质疑分别用复读聚集与质疑词+情感分作代理，只反映观众反应，不评价内容质量；系统不判定细粒度情绪（弹幕情绪词语义随内容语境漂移，判断留给有画面上下文的人）；质疑层宁缺毋滥，未检出即未检出；切片/高光建议指向共振最强处，不等于内容最精彩处；复读近似为文本完全相等，近义变体可能漏检；弹幕不含画面与内容标签，不对内容含义作解释。
- **复现边界**：仓库不携带采集数据、goldens 与宿主 venv 等运行资产（词表源数据保留在 `config/dicts/source/`）；`make seed-data`、`make goldens`、`make cron-install`、`make perf-suite` 依赖 `~/projects/danmaku` 与 `~/projects/danmaku-v2` 的目录布局。Compose 注释中提到的 DolphinScheduler 与监控叠加文件 `docker-compose.obs.yml` 未包含在本仓库，日频编排目前以宿主 crontab 与 `make daily` 手工版为准。
- **性能结论的边界**：集群只有 1 个 worker 与 2 个 executor 核，并行度 4 无法测量；放大数据为合成重放数据，结果只说明随规模与并行的行为，不是真实流量下的吞吐承诺。
- **工具链残留**：`tools/` 下的部分探针与早期 E2E 脚本保留了当时的本机绝对路径或已下线的页面引用，运行前需按当前环境调整。

## 11 作者与许可

作者（GitHub：Mo12y）。

许可：本仓库不包含 LICENSE 文件，版权归作者所有；对外引用、复用或分发前，请与作者联系。