# FINAL_REPORT · exp46 系统救援（2026-10-27 21:35）

## 1. 系统现状（全部实测）

- **服务**：fastapi 容器运行中（2026-10-27 20:45 重建 v3）；nginx 静态挂载实时生效。
- **API**：`GET /api/ml/forecast[?bvid=]`、`GET /api/ml/forecast/{bvid}`——登录中间件保护；列表 code=0 n=2167；未知 bvid 404；缺快照 503。
- **页面**：`/ml_forecast.html`——**三区展示**：【观测】弹幕曲线+前中后密度+S1；【候选事件】算法候选区间（标注"未经人工确认"）；【预测】D1 余量（ok/pending/insufficient 三态）+ D2 结论稳定度（概率 vs 实测分列）。顶部含数据窗、关键指标与限制说明。
- **快照 v2**：2,167 支（D1：ok 1,313 / pending 706 / insufficient 148）；每支含 rhythm（seg3_dpm、S1、事件候选、≤120 点曲线）。
- **测试**：全库 pytest **264 passed / 1 skipped**（修复页面升级引入的 2 处裸 hex 违规后恢复全绿）；skips=宿主无 fastapi 的载荷单测+登录竞态，均有等效 live 验证。

## 2. ML 方法与真实结果（分层呈现，不混指标）

| 能力 | 任务 | 真实结果 | 状态 |
|---|---|---|---|
| D1 未来 24h 弹幕量 | 回归（时间切、bvid 隔离） | ΔR²=+0.095 [0.045,0.144]，线性模型足够 | ✅ 已集成 |
| D2 结论稳定度 | 分类（时间切+GroupKFold） | ΔAUC=+0.078 [0.060,0.096]；PR-AUC 0.725；Brier 0.120 | ✅ 已集成 |
| C1 节奏阶段识别 | 人工一致性 + 算法 | 人工一致性 0.90/0.92 ✅；算法 EqualK3/SingleCP/BinsegBIC 召回中位全 0 ❌；事件型 e45-v2 召回 0.667 ✓ 但误差 31s ✗ | ⚠️ 自动分段未过门槛（负结果如实）；候选事件以"未确认"身份进系统 |
| 节奏统计画像 | 描述统计（exp42/43 口径） | 片头聚集群体模式（前/中=177/113）；1,586 支全片可观测；合格池 669 支 | ✅ 已集成 |
| D4 爆发概率校准 | 训练内 OOF + 时间外推 cohort | skill +0.378/斜率 0.93（cohort）；**未接生产** | 🔒 实验产物 |

## 3. 端到端演示（3 支真实视频，证据在 `demo_response_*.json`）

| bvid | 分区/时长 | 观测（节奏画像） | 预测 | 说明 |
|---|---|---|---|---|
| BV1kBpi6aEDP | 音乐综合/37min | 前/中/后=251/108/98 条每分（片头聚集 2.3×）；S1=0.631；2 个候选事件 | D1=**insufficient**（前 6h 无弹幕，状态明确提示） | 降级路径演示 |
| BV1ZSp46AEF8 | 短片/15min | 110/92/90；S1=0.184；3 事件 | D1=86.4 条；D2 p=0.056（结论不稳，已实测漂移） | 低稳定度案例 |
| BV1wAp76yEu1 | 人文历史/27min | 209/93/88（片头 2.2×）；S1=0.458；4 事件 | D1=1197 条；D2 p=0.627 | 全功能案例 |

smoke：登录 → 列表 → 3 详情 → 页面 200 → 未知 bvid 404，全部通过（`bash smoke_test.sh` 可复跑）。

## 4. 已知限制

1. 快照数据窗固定 1012–1026（exp35 D-conf 边界外保守选择）；刷新需重跑 `jobs/ml_forecast_build.py`。
2. C1 候选事件为**算法候选**（泊松 2σ 局部基线），区间端点存在 25–60s 系统滞后（exp45 误差 31s>15s），页面已标注"未经人工确认"。
3. D1/D2 为单 seed 单次运行 + exp38 双次复算；跨月滚动复验未做。
4. D4 未接入生产（实验产物）；其 cohort 验证非 exp35 D-conf 口径。
5. 全库测试 1 skipped 的原因与等效验证见 38 号报告。

## 5. 产物清单

- `FINAL_REPORT.md`（本文件）、`eval_results.json`、`DEMO.md`、`smoke_test.sh`
- `figures/`：exp42 S1 ECDF 与观测起点图、exp45 事件对照图（5 支）、演示响应 JSON×4
- 代码：`jobs/ml_forecast_build.py`（+rhythm）、`serve/ml_forecast.py`（+rhythm 透传）、`serve/static/ml_forecast.html`（三区页面）——**均已部署到运行容器（v3）**
- 未提交事项：exp40–45 与本轮修改均未 commit（工作区另有 31 项用户修改未动），提交范围与时机由用户决定。

---

## 6. 2026-10-27 夜间二次迭代（本轮）

### 6.1 ML 侧：exp48 核心结论被推翻，页面口径同步改写

- 新增 `experiments/exp48_1_scoring_aligned.py` + `experiments/exp48_1_scoring_aligned/`（report/results/per_video_K4.csv/fig/run.log/verify）。
- 根因：exp48 的 B2（视频边际 LOO）`loo` 被广播成 `(n,n,K)`，`.mean()` 平均了 n×K 个伪概率（含该视频从未出现的状态），B2 被系统性压低 0.33–0.65 nats。
- 结果（K=4，同划分配对）：M1−B2 由 **+0.869 → −0.078 [−0.111, −0.043]**；新增因果基线 B2b（只用 t 之前）得 M1−B2b=+0.010 [−0.021, +0.042]；M2−M1=+0.064 [0.049, 0.080] 稳健；置换检验 p≈0.0005（11.6 SD）。
- `experiments/exp48_markov_increment/SUPERSEDED.md` 标注原目录结论作废，原 `results.json`/`report.md` 原样保留。
- **影响**：任何材料不得再引用 0.87 nats、98.6% 正向占比、"高弹幕组增量最大"。已 grep 确认 `docs/00-规范/` 与 `docs/` 全库未引用这些数字，无 SSOT 污染。

### 6.2 系统侧：节奏分析卡补全（`/ml_forecast.html`）

| 变更 | 位置 | 说明 |
|---|---|---|
| 状态转移矩阵 | `jobs/ml_forecast_build.py::state_fields` + 页面 `matrixHtml()` | 训练矩阵概率 + 本视频实际转移次数；当前状态行高亮；转移次数 <10 时提示"样本不足，不可当规律" |
| 数据覆盖提示 | 同上 `coverage` | 分钟桶数、零弹幕桶数与占比、尾部空白桶；零桶 >50% 或尾部空白 >max(3, 20%) 时显式告警 |
| 状态字段补全 | 同上 | 新增 `K`、`trans_counts`、`coverage`、`note`；`trans`/`next_prob`/`seq`/`edges` 保持原样（向后兼容） |
| 口径写死 | `STATE_NOTE` 常量 | 说明状态为操作性强度等级、转移矩阵来自训练集不重拟合、exp48.1 负结果 |
| 辅助预测降级标注 | 页面 | D1/D2 标为「预测·辅助」，D1 标题修正为 (6h,30h] 口径 |
| 候选事件说明 | 页面 | 明确"算法候选、未经人工确认、端点滞后 25–60s、人工标注仅 5–6 支" |
| 错误处理 | 页面 `boot()` | 非 2xx 显式报错；快照缺失给出重建命令 |

### 6.3 验证（全部实测）

| 项 | 命令 | 结果 |
|---|---|---|
| 快照重建 | `~/danmaku-venv/bin/python jobs/ml_forecast_build.py` | `SNAPSHOT-DONE items=2167 d1={'ok':1313,'pending':706,'insufficient':148}`；含状态 1807 支 |
| 全量测试 | `~/danmaku-venv/bin/python -m pytest tests/ -q` | **266 passed / 4 skipped / 0 failed**（跳过：2 项需 `EXP35_DATA_SMOKE=1`+真实 ODS；2 项宿主 venv 无 fastapi，已由容器内 live 契约覆盖） |
| 前端模板离屏校验 | `node check_card.js serve/static/ml_forecast.html data-out/.../ml_snapshot.json` | `CARD-CHECK-OK`：长(391桶)/短(2桶)/无状态 三类样本均渲染；`trans_counts` 合计=n−1；矩阵行和=1 |
| 端到端 | `e2e_check.py` | `E2E-OK ok=9`：未登录 401、列表 2167、4 类代表视频字段一致、D1 三态齐全、404 正确、页面 200 |

### 6.4 本轮发现的系统级缺口（未修，如实记录）

1. **Spark 管线停在 1015**：HDFS `/raw` 已到 `batch_date=20261027`，但 `/warehouse/ods_danmaku_full` 最新 **20261015**、`/warehouse/dwd_danmaku_clean` 最新 **20261013**；`docker logs spark-master --since 72h` 无 Application。⇒ 10-16 起 13 批数据只落地未加工。
2. **v3 分析页数据止于 1013**：PG `v3_video_danmaku` 仅含 0902/0907/0914/0917/1012/1013 六批（1,115,185 行 / 803 支）；`ads_video_curve` 更旧（仅 9 月 4 批，max 20260917），由 legacy `/api/video/{bvid}` 读取，v3 页面不依赖它。
3. **节奏分析卡不受上述影响**：`ml_forecast` 快照直读 `data/cli_collected/hotlist_top200_202610*.csv`（1012–1026），与 Spark/PG 解耦。
4. **补齐路径**（按代价从低到高）：`make ingest` → `make m0-full --batches 20261014..20261027` → `make m2-l1` → `jobs/export_v3_danmaku.py` + `jobs/load_v3_danmaku.sh`。需连续数小时且涉及分区覆盖，本轮未执行。
