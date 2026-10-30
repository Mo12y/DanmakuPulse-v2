# DanmakuPulse-v2

基于 Spark 的 B 站热榜视频弹幕互动节奏分析系统 · 第二版实现。

第一版（`DanmakuPulse`）是单机 Flask 应用；本版把数据流重做为 **RAW → ODS → DWD → DWS/ML → ADS** 的分层数仓，用 Docker Compose 拉起 9 个服务（HDFS、Hive Metastore、Spark、PostgreSQL、FastAPI、Nginx），对拍基准沿用第一版的 NumPy 实现。

在节奏分析之外，本版新增了**秒级精彩时刻检测**：定位观众在哪一秒集体起反应（共振峰）、哪一段被质疑（质疑段），并做同分区对标。

> 定位：描述性测量工具。不预测热度、不判情绪、不评内容质量。

---

## 1. 它分析什么

| 能力 | 输出 |
| :-- | :-- |
| 节奏指纹 | 每视频 16 维特征 + 101 点密度曲线 + 分区基准 |
| 节奏事件 | 四检测器融合的秒级事件（5 秒桶）+ 同步性 + 锚词滞后 |
| 纵向演化 | 上榜轨迹、跨批指标轨迹、轨迹分型、UP 档案 |
| 文本与情感 | 词典打分（否定取反 / 程度乘子 / 极性×强度） |
| 精彩时刻 | 共振段、质疑段、从众/中性/批判画像、共振强度百分位 |
| 切片建议 | 长视频（>1800s）给切片位置，短视频给高光位置 |

**不做的**：不预测播放量、不做推荐、不判定细粒度情绪、不评价内容质量。共振与质疑都是**观众反应**的代理指标。

---

## 2. 系统架构

### 2.1 九个服务

| 服务 | 镜像 | 作用 | 端口 |
| :-- | :-- | :-- | :-- |
| namenode | bde2020/hadoop-namenode 3.2.1 | HDFS 名称节点 | 9870 |
| datanode | bde2020/hadoop-datanode 3.2.1 | HDFS 数据节点（单副本） | — |
| postgres | postgres:15-alpine | Hive 元库 + ads 应用库 | 5433 |
| hive-metastore | Spark 镜像内 Hive 2.3.9 | 元数据服务 | 9083（内部） |
| spark-master | danmaku-spark:3.5.4-py | Spark standalone 主节点 / driver | 8080 |
| spark-worker | danmaku-spark:3.5.4-py | 单 executor（1g / 2 core） | 8081 |
| spark-history | danmaku-spark:3.5.4-py | 事件日志历史 | 18080 |
| fastapi | serve/Dockerfile（python:3.11-slim） | 只读 API | 8000（经 nginx） |
| nginx | nginx:alpine | 唯一入口：静态页 + 登录门禁 + /api 反代 | **8088** |

### 2.2 数据流

```
collect/bili.py
  → data/cli_collected/（CSV + meta.json + 榜单快照 + fetch_times）
  → tools/stage_raw_local.py → data/raw-local/batch_date=*/
  → compose/assets/ingest_raw.sh → HDFS /raw/batch_date=*/
  → m0_min / m0_daily → ODS；m0_full → DWD
  → m1 / L1 / L2 / L3 / M4 / M5 / M6 → DWS 与 ML 表
  → m2_l4_highlights → 共振峰 / 质疑段 / 反馈画像
  → ads_min / ads_build / ads_highlights → PostgreSQL ads
  → FastAPI → nginx :8088 → 静态页面
```

### 2.3 分层与主要表

| 层 | 位置 | 主要表 |
| :-- | :-- | :-- |
| RAW | HDFS `/raw/batch_date=*/` | `danmaku.csv`、`meta.json`、`hotlist.json`、`fetch_times.csv` |
| ODS | `/warehouse/ods_*` | `ods_danmaku_full`、`ods_batch_meta`、`ods_hotlist_video` |
| DWD | `/warehouse/dwd_*` | `dwd_danmaku_clean(_min)`、`dwd_danmaku_sent`、`dwd_spam_flag`、`dim_video` |
| DWS | `/warehouse/dws_*` | 密度、指纹、分区基准、事件、同步、滞后、分段、聚类、Hawkes、轨迹、生命周期、共振峰、质疑段、反馈画像 |
| ADS | PostgreSQL `ads` | 19 张 `ads_*` 表（含 `ads_video_highlights`、`ads_partition_fingerprint`） |

Hive Metastore 由 `jobs/register_hms.py` 注册 41 张外部表（25 张分区表 + 16 张平铺表）。

---

## 3. 核心口径

- **密度曲线**：101 个等宽进度桶；高斯平滑 σ=1.0；峰值判定 μ+2σ。
- **节奏事件**：5 秒网格、不平滑；D1 前缀 P70 + 三点斜率、D2 变化率、D3 熵 + 峰距、D4 谱通量；权重 `[0.3, 0.3, 0.2, 0.2]`，θ=0.5 截断。当前为**未标定**默认值。
- **精彩时刻**：5 秒桶内同一文本复读 ≥3 次计入复读；峰值 25% 回落成段；相邻峰谷值高于较小峰 50% 时合并为同一爆发区。质疑弹幕需命中质疑词、长度 ≥8、情感分 <0，且段内 ≥3 条。
- **对拍容差**：计数类零容差，浮点类 ≤1e-9；曲线逐点 ≤1e-9。
- **池级锚点**（历史 4 批对拍链）：509 个视频批次、672,439 行 DWD、508 个池内视频、669,995 条池内弹幕、清洗保留率 99.3364%。
- **去重键双轨**：对拍链 `(bvid, text, time_offset)`；生产链 `(bvid, cid, row_id)`。旧键会误删「同文同刻但 row_id 唯一」的真实复读弹幕，两链隔离、互不影响。

---

## 4. 目录结构

```
collect/    采集 CLI 与日频包装脚本
compose/    9 服务 compose、Hadoop/Spark/HMS 配置、nginx、Spark 镜像 Dockerfile
config/     params.yaml（参数中心）、batch_lineage.json（批次血缘）、dicts/（情感/锚词/质疑词表）
jobs/       Spark 作业、共享纯函数、对拍与 HMS 注册
serve/      FastAPI 服务（50 个路由）、13 个静态页、v3data/ 前端快照
tests/      单元测试与接口契约测试；goldens/ 为对拍基准 CSV
tools/      对拍基准生成、诊断探针、评估与性能测试
Makefile    底座、ETL、对拍、分析链与日频链的统一入口
```

`data/`、`data-out/`、`logs/`、`lib/*.jar`、`serve/static/vendor/`、`compose/assets/tls/` 均被忽略。

---

## 5. 快速开始

### 5.1 环境前提

- 宿主 Linux，Docker + Docker Compose v2。
- 代码含固定路径假设：仓库在 `~/projects/danmaku-v2`，第一版仓库在 `~/projects/danmaku`，宿主虚拟环境 `~/danmaku-venv`。
- 宿主 venv 需含 PySpark、NumPy、SciPy、pytest；聚类步骤另需 tslearn、scikit-learn。

### 5.2 首次部署

```bash
make setup          # 下载 JDBC / Disruptor / ECharts / html2canvas
docker build -t danmaku-spark:3.5.4-py compose/assets/spark-docker
make seed-data      # 从第一版仓库复制历史 4 批采集数据
make stage          # data/cli_collected → data/raw-local
make up             # 生成自签证书并启动 9 个服务
make goldens        # 生成对拍基准并上传 HDFS
make ingest         # data/raw-local → HDFS /raw
make smoke          # HDFS/PG 读写冒烟
make etl            # 对拍链：m0_min → m2_l1_min → ads_min
make parity         # 曲线 / 8 维指标 / 池级锚点对拍
make replay         # 全链两跑一致性比对
```

### 5.3 生产链

```bash
make m0-full           # 完整清洗 + 刷量检测
make m1                # 三值对账
make m5                # 情感词典 v0
make m2-l1             # 密度、指纹、分区基准
make m2-l2             # 事件、同步、滞后、分段
make m2-l3             # Hawkes 拟合
make m2-l4-highlights  # 共振段 / 质疑段 / 反馈画像
make m4                # 上榜、轨迹、演化、UP 档案
make m6                # 特征矩阵、消融、面板
make ads-build         # DWS → PostgreSQL
make ads-highlights    # L4 → PostgreSQL
make hms               # 注册 Hive 外部表
```

### 5.4 日频与运维

```bash
make collect                              # 手动采一次 Top200
make cron-install                         # 每日 01:00 采集 + 每小时兜底
make daily BATCHES=20261012,20261013      # 守卫 → 装载 → 生产链 → ADS
make serve-rebuild                        # 改动 serve/ 后重建 FastAPI
```

### 5.5 创建管理员

```bash
docker exec fastapi python seed_admin.py
docker exec -e ADS_ADMIN_PASSWORD=<password> fastapi python seed_admin.py
```

服务入口：`http://<host>:8088`（nginx）。API 统一返回 `{code, data, meta}`；页面门禁为 nginx `auth_request` + FastAPI 对 `/api/*` 返回 401/403。

---

## 6. 测试

```bash
~/danmaku-venv/bin/python -m pytest tests/ -q
bash tests/accept_auth.sh http://127.0.0.1:8088
```

- **算法纯函数**：检测器与融合、Hawkes 合成数据拟合、轨迹规则、IoU/AP、精彩时刻核心。
- **接口契约**：端点状态码、统一响应、404 负例（需服务在线，否则跳过）。
- **服务与前端守卫**：设计令牌、页面标题与措辞、AI 调用限额。

对拍链三项全过才算 parity 成功：曲线逐点 ≤1e-9、8 维指标（计数类精确 / 浮点 ≤1e-9）、池级锚点逐数一致。

---

## 7. 结果与状态

| 模块 | 状态 |
| :-- | :-- |
| 服务底座与对拍链 | 已实现（`smoke` / `etl` / `parity` / `replay`） |
| 生产链（M0/M1/M5） | 已实现 |
| L1 密度 / 指纹 / 聚类 | 已实现 |
| L2 事件 / 同步 / 滞后 / 分段 | 已实现，**未标定**（`calibrated_on='uncalibrated'`） |
| L3 Hawkes | 已实现，**负结果**（KS 未通过，按设计降级为幂律） |
| M4 纵向演化 | 已实现 v2 |
| M6 特征与面板 | 已实现 |
| L4 精彩时刻 | 已实现（888 视频批次，共振段 2,807 段，质疑段 336 段） |
| 人工标注 | 未开始 |

精彩时刻实测（分析口径 `is_spam=false`，1,115,185 行 / 888 个视频-批次）：共振段 2,807 段覆盖 757 个视频-批次（85.25%），片头 60 秒内占 22.55%；质疑段 336 段覆盖 230 个（25.90%）；弹幕构成 从众 47.09% / 中性 51.67% / 认真质疑 1.24%。

---

## 8. 局限

- **样本**：历史只有 4 个周频批次，两批榜单快照永久丢失，9/14 批排名尾部 87 个视频未采集，存在头部偏差；结论不外推到全站。
- **采集**：弹幕接口对饱和视频返回固定容量滑窗，`danmaku_got` 可能量子化且小于上报值，跨批水平不可直接比较。
- **L2 未经人工真值验证**：权重与阈值为默认值，事件数量与位置只能作探索性结果。
- **L3 模型失配**：Hawkes 残差 KS 未通过，参数仅作诊断。
- **精彩时刻边界**：只反映观众反应，不评内容质量；不判细粒度情绪；质疑层宁缺毋滥；切片/高光指向共振最强处，不等于内容最精彩处；复读以文本完全相等为近似。
- **性能结论**：集群仅 1 worker / 2 executor 核，放大数据为合成重放，不代表真实流量承诺。
- **文档**：设计文档与标注指南由作者本地保留，不随仓库分发；本 README 为仓库内唯一说明。

---

## 9. 作者与许可

作者（GitHub：Mo12y）。

本仓库不含 LICENSE 文件，版权归作者所有；引用、复用或分发前请与作者联系。
