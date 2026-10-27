# exp35 冻结前审计报告

- 审计日期：2026-10-27
- 审计对象：exp35 正式实现、测试、数据核验、D-train-only 分析、冻结候选协议
- 代码基线：`danmaku-v2` @ `b113aba`（实现+测试+清单核验）→ `2a0adb5`（本报告+方法章草稿+行尾统一）
- 审计性质：**只读核验 + 机械验收**。未读取 D-conf（项目中不存在 `dt >= 20261027` 的 ODS 分区）
- 结论：**正式实验尚未冻结** —— 8 项阻塞（B-01…B-08）

状态词：`PASS` 有本轮命令证据｜`FAIL` 不满足且不得绕过｜`BLOCK` 冻结前必须由研究者裁定｜`N/A` 本设计不适用

---

## 0. 一句话结论

> **数据可信、实现可跑、测试全绿；但判据（SESOI vs 可分辨性）、模型定义（M0/M4）、多 P 口径、D-conf 采集规模、负对照判别力这 5 件事没解决之前，任何"D-conf 可以开跑"的说法都不成立。**

---

## 1. 数据核验

| # | 检查项 | 命令 / 证据 | 结果 |
|---|---|---|---|
| A1 | 20 个输入文件存在 | `sha256sum -c docs/exp35_508/manifest.sha256` | **PASS** 20/20 成功 |
| A2 | 哈希与字节数逐文件一致 | `--stage verify-manifest` → `20/20 ok` | **PASS** |
| A3 | 清单完整（无重复/缺失条目） | `file_manifest.csv` 21 行、`manifest.sha256` 20 行一一对应 | **PASS** |
| A4 | ODS/DWD/pool 行数 | 676,931 / 672,439 / 669,995 | **PASS** 全部复现 |
| A5 | dmid 审计 | 唯一 674,498；重复 2,433 ID/4,866 行；跨 bvid 0；仅 BV1mwY96jEom | **PASS** |
| A6 | 11 行旧批独有 | 11 行、全 BV1mwY96jEom、全 0914 | **PASS**（是"保留最新快照"的确定性后果，非遗失） |
| A7 | D-train/D-dev 交集 = 0 | 两侧独立构造后求交 | **PASS** |
| A8 | 时间字段合法性 | 缺失 0 / 负值 0 / `> duration` 0 | **PASS** |
| A9 | 标签与门槛计数与已报告一致 | 见下 | **FAIL（可定位）** |

### A9 明细

| 项 | 已报告 | 本轮（批准规则 `>=`） | 判定 |
|---|---:|---:|---|
| 全池可评估视频 | 304 | **306** | 差 2 |
| D-train 可评估 | 133 | **134** | 差 1 |
| D-dev 可评估 | 171 | **172** | 差 1 |
| manifest count-only 点数/正例 | 13,049 / 4,667 | 13,107 / 4,689（pool+`>=`） | 差 58 / 22 |

- 已报告的 304/133/171 **只在严格 `>` 下逐项精确复现**；批准规则是 `>=`。⇒ **以 306/134/172 为准**（阻塞 B-06）。
- count-only 无法完全复现，正例率 0.3577 一致；降级为参考值（阻塞 B-07）。

---

## 2. 实现契约

| # | 检查项 | 证据 | 结果 |
|---|---|---|---|
| B1 | 标签：floor 分桶 / 左闭右开 / 阈值一次算 / `>=` | `tests/test_burst_label.py` 24 项 | **PASS** |
| B2 | 决策点闭区间上界、最后一点保留、非法点排除 | `test_last_legal_point_kept`（K=18/23/24/29/30/41） | **PASS** |
| B3 | 非法/越界时间显式计数 | `test_invalid_times_dropped_with_counts` | **PASS** |
| B4 | dmid 去重 + 真复读保留 | `test_dmid_dedup_keeps_first_copy` / `test_true_repeat_...` | **PASS** |
| B5 | 特征只用 `c[0:t]`（100% 决策点逐点验） | `test_all_features_causal_on_every_point` | **PASS** |
| B6 | `slope` 对数差分、无 epsilon、值域 < 50 | 单测 3 项 | **PASS** |
| B7 | `ent` 12 桶归一（0.2790 / 0 / 1） | 单测 5 项 | **PASS** |
| B8 | `since/censored` D2 滚动参照；t=12 必删失 | `test_first_decision_point_is_censored` 等 5 项 | **PASS** |
| B9 | 白名单 6 列、注入 `n_dm` 立即失败 | `test_inject_ndm_must_fail` | **PASS** |
| B10 | M0/M4 无定义 → 调用即 raise | `test_unapproved_models_keep_interface_only` | **PASS（接口层）** / 定义缺失见 B-02 |
| B11 | 切分：批次→集合、未登记批次拒绝、first_seen 用原始表 | 单测 9 项 | **PASS** |
| B12 | ODS/DWD 输出目录独立、禁止混层 | `test_output_dirs_are_independent` / `test_layer_mixing_rejected` | **PASS** |
| B13 | v2 探索数据不进正式路径 | `guard_path` + 单测 | **PASS** |
| B14 | 正式入口拒绝 D-conf（路径 + 批次 + manifest 行） | 单测 3 项 | **PASS** |
| B15 | 输入文件白名单（不通配目录） | `test_loader_reads_only_listed_files` + provenance 带 sha256 | **PASS** |
| B16 | 校准：按 bvid 5 折交叉拟合、折外视频交集 0、只用 D-train | `tests/test_burst_calibration.py` 9 项 | **PASS** |
| B17 | 主指标：视频宏平均视频内 AUC、单类剔除不填 0.5、手算一致 | 单测 5 项 | **PASS** |
| B18 | 配对 Bootstrap：单位 bvid、两模型同一抽样、同种子可复现 | 单测 4 项 | **PASS** |
| B19 | 多 P 排除 | `part_count` 全为空值 | **FAIL（不可执行）** → B-03 |

---

## 3. 测试执行

```
cd ~/projects/danmaku-v2
~/danmaku-venv/bin/python -m pytest \
  tests/test_burst_label.py tests/test_burst_features.py tests/test_burst_split.py \
  tests/test_burst_calibration.py tests/test_burst_eval.py tests/test_burst_isolation.py -q
```
结果：**91 passed in 1.83s，0 failed，0 skipped**（`【实测】`，行尾统一为 LF 后复跑一次）。

分文件：label 24 / features 26 / split 10 / calibration 9 / eval 12 / isolation 10。

---

## 4. D-train-only 分析

| # | 项 | 结果 |
|---|---|---|
| C1 | 只用 D-train，未读 D-dev 标签拟合 | **PASS**（`load_sequences(batches=D_TRAIN_BATCHES)`；训练入口 `assert_train_only`） |
| C2 | 效应量网格预先声明、不按结果删格 | **PASS**（`POWER_GRID` 常量 7 档） |
| C3 | 随机种子与重复次数写入产物 | **PASS**（`seed=20261026`, `reps=2000`） |
| C4 | 功效分析结果 | ΔAUC **+0.0040**，CI **[−0.0312, +0.0418]**，配对 SD **0.2220**，MDE(80%)@150 = **0.0508** |
| C5 | D-conf 门槛模拟结果 | n=150 → 合格 p50 **106**；n=250 → p05 **165**；建议正例 ≥2,288 / 负例 ≥3,691 |
| C6 | SESOI 0.02 与 MDE 0.05 不匹配 | **BLOCK B-01** |

**重要说明**：C4/C5 是 **设计分析**，不是模型效果证据。即使 ΔAUC 为负，也不构成"M2 无效"的结论（那要 D-conf）。

---

## 5. 负对照（管道自检）

| # | 对照 | 期望 | 实测 | 判定 |
|---|---|---|---|---|
| D1 | 标签置乱（评估侧） | ≈0.5 | **0.5105** | **PASS** |
| D2 | 标签置乱（拟合侧） | ≈0.5 | **0.3954** | **FAIL** |
| D3 | 仅时长 | 恰 0.5000 | **0.500000** | **PASS** |
| D4 | 特征窗前移 6 桶 | 下降 ≥0.02 | **0.6177 vs 基线 0.6158** | **FAIL（不降）** |

解读（**不是管道 bug**）：
- D2：视频内置换**不是零假设**——它保留了"每视频均值特征 ↔ 每视频均值标签"的跨视频结构，模型仍能学到真实的跨视频关系，于是在视频内评估时表现为反向（<0.5）。该对照应被"评估侧置乱"取代。
- D4：M2 的特征是 60 s 聚合量（`λ`、`slope`、`prog`），把窗口整体移动 30 s 几乎不改变输入向量；该对照对本特征集**没有判别力**。

⇒ 负对照清单需修订（**B-04**）：D2 删除或改为"跨视频置换"；D4 改为"桶序打乱"或"用相邻视频的特征"。

---

## 6. 冻结前阻塞清单

| 编号 | 阻塞 | 为什么阻止冻结 | 需要谁裁定 |
|---|---|---|---|
| **B-01** | SESOI 0.02 vs MDE 0.051 | 成功判据在本设计下不可执行；三条处置（承认 0.05 / 降级描述性 / 扩到 ~968 支）必须**揭盲前**选定 | 研究者 |
| **B-02** | M0 / M4 无协议定义 | 只保留接口（调用即 raise）；若论文要写阶梯，必须先补定义 | 研究者 |
| **B-03** | `part_count` 全空，多 P 规则不可执行 | 主分析口径不完整，且不得声称"已排除多 P" | 研究者（选 (a) 声明未知 / (b) 用 cid 另测） |
| **B-04** | 负对照 D2/D4 不具判别力 | 无法证明"这套评估能失败"；若直接冻结，等于把无效自检写进协议 | 研究者确认修订 |
| **B-05** | D-conf 采集规模 | "合格 ≥150"需抽 ~250 支；采集计划未确认 | 研究者 |
| **B-06** | 已报告 304/133/171 与 `>=` 冲突 | 引用错误会污染摘要/答辩 | 研究者签字改用 306/134/172 |
| **B-07** | manifest count-only 不可完全复现 | 基准表不可作为冻结基准 | 研究者确认以 `audit.json` 为准 |
| **B-08** | `M1 ≡ L1` 属文本推定 | 主比较基线身份未签字 | 研究者 |
| **B-09** | `git_status_clean = false` | 提交时工作区仍有 22 个与本实验无关的已跟踪文件被改动；commit 不代表运行代码 | 维护者清理后再冻结 |

---

## 7. 明确未做的事（禁区自检）

| 禁区 | 状态 |
|---|---|
| 读取/扫描/汇总 D-conf | **未执行**。项目中不存在 `dt >= 20261027` 的 ODS 分区；实现层有路径/批次/manifest 三重拒绝 + 单测 |
| D-conf 正式确认评估 | **未执行** |
| 用 D-dev 调模型/选特征/改阈值 | **未执行**。D-dev 只用于"复核已报告的合格视频数"（任务书要求的独立复核），**未训练、未评估、未调参** |
| 用 v2 探索数据/历史污染结果 | **未执行**（路径级拒绝） |
| 把历史 AUC 0.735 / 0.752 写成结果 | **未执行**，本报告不含任何模型有效性主张 |
| 修改已批准数据口径 | **未执行**（差异如实登记，未重写清单） |
| 采集任何新数据 | **未执行** |

---

## 8. 最终判定

```
正式实验尚未冻结。阻塞项：B-01 ~ B-09（共 9 项，其中 B-01/B-03/B-05 属"能否做"级别）
```

**已达成**：数据可核验、实现可运行、测试全绿、D-train-only 分析可复现、校准实现唯一且被测试、集合隔离有自动化保障、文件/命令/版本可追溯。

**未达成**：判据可执行性、模型定义完整性、多 P 口径、负对照判别力、D-conf 采集规模、工作区洁净度。

在 B-01…B-09 全部关闭并由研究者签字前，**不得**把 `35_formal_spec_rc4_candidate.md` 的 `status` 改为 `frozen`，不得采集 D-conf，不得执行正式确认评估。

---

## 9. 本轮变更

| 类型 | 文件 |
|---|---|
| 新增实现 | `jobs/m2_l6_burst_risk.py`（约 1,010 行） |
| 新增测试 | `tests/test_burst_{label,features,split,calibration,eval,isolation}.py`（91 项） |
| 新增文档 | `docs/exp35_508/{audit_manifest_report.md, 35_formal_spec_rc4_candidate.md, pre_freeze_audit_report.md, thesis_method_chapter_draft.md}` |
| 新增产物 | `docs/exp35_508/{audit_manifest_files.csv, power_dtrain.json, gate_sim_dtrain.json, negative_controls_dtrain.json}` |
| 数据清单 | 从旧仓 `~/projects/danmaku/docs/exp35_508/` **原样复制**到本仓（字节未改，哈希不变） |
| 未改动 | 任何数据文件、任何既有 jobs/serve 代码、任何历史协议版本 |

提交记录（`danmaku-v2`，显式路径提交，未 `git add -A`）：

```
2a0adb5 [WorkBuddy] exp35 冻结前审计+论文方法章草稿+统一 LF 行尾（仍未冻结，9 项阻塞）
b113aba [WorkBuddy] exp35 正式实现+91 项测试+D-train 功效/门槛模拟+清单核验（未冻结，rc4 候选）
```

工作区洁净度：提交后仍有 **22 个与本实验无关的已跟踪文件处于 modified 状态**（`Makefile`、`serve/*.py`、`serve/static/*.html` 等，均为本轮之前既存改动）⇒ `git_status_clean = false`，计入 B-09。
