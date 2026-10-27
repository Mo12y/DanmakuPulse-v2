# exp35 数据清单与 SHA-256 独立核验报告

- 核验日期：2026-10-27
- 核验对象：`docs/exp35_508/`（清单与 ID 文件）+ `~/projects/danmaku/data/warehouse/`（ODS/DWD 本体）
- 核验方式：**只读**。未改写任何清单、未重算哈希去掩盖差异、未触碰 D-conf
- 执行环境：Ubuntu VM `molly@192.168.88.100`，`~/danmaku-venv/bin/python`（pandas 3.0.4 / numpy 2.3.5 / pyarrow 25.0.1 / scikit-learn 1.9.0）
- 本报告状态：`NOT FROZEN`（本报告只描述核验结果，不构成冻结）

状态词：`【实测】` 本轮命令跑过，附可复现路径｜`【口径】` 来自项目文档｜`【差异】` 与已报告基准不一致｜`【阻塞】` 冻结前必须裁定

---

## 1. 结论

| 项 | 结果 |
|---|---|
| 文件数量 | **20 / 20 存在**（4 原始 CSV + 16 ODS Parquet）`【实测】` |
| SHA-256 | **20 / 20 通过**，无 FAILED、无缺文件 `【实测】` |
| 字节数 | **20 / 20 与 `file_manifest.csv` 一致** `【实测】` |
| 行数（ODS/DWD/pool） | 676,931 / 672,439 / 669,995 **全部复现** `【实测】` |
| dmid 审计 | 唯一 674,498；重复 2,433 ID / 4,866 行；跨 bvid 0；仅 BV1mwY96jEom **全部复现** `【实测】` |
| 11 行旧批独有 | 11 行、全是 BV1mwY96jEom、全在 20260914 **全部复现** `【实测】` |
| D-train/D-dev 交集 | **0**（175 / 234，两侧独立构造后求交）`【实测】` |
| 视频数 / 快照数 | 508 / 509 `【实测】` |
| 标签与门槛计数 | ⚠️ **2 处差异**，见 §6（`>=` vs `>`；count-only 无法完全复现） |
| 多 P 排除规则 | ⛔ **不可执行**：`part_count` 在本 4 批中全为空值 `【实测】` |

**净结论：数据本体可信（哈希、行数、去重审计全部与清单一致）；不一致的是"标签口径与门槛计数"这一层，且差异可定位到比较运算符。**

---

## 2. 逐文件核验

命令（原始口径，路径为绝对路径）：

```bash
cd ~/projects/danmaku && sha256sum -c docs/exp35_508/manifest.sha256
```

结果：20 行全部 "成功"，退出码 0。`【实测】`

模块口径（含字节数比对，且生成 CSV 明细）：

```bash
cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python jobs/m2_l6_burst_risk.py --stage verify-manifest
# → manifest: 20/20 ok, all_ok=True   （退出码 0）
```

逐文件明细见 `docs/exp35_508/audit_manifest_files.csv`（字段：layer / batch / file / expected_bytes / actual_bytes / expected_sha256 / actual_sha256 / status）。该 CSV 由上述命令生成，20 行 `status=ok`。`【实测】`

| 层 | 文件数 | 备注 |
|---|---:|---|
| `raw_csv` | 4 | `hotlist_top100_20260902` / `top200_20260907` / `top200_20260914` / `top100_20260917` |
| `ods_parquet` | 16 | 4 批 × 4 part，Spark 4.2.0 写出 |

**清单本身的完整性**：`file_manifest.csv` 21 行（表头 + 20 行数据），无重复路径、无缺失条目；`manifest.sha256` 20 行，与 CSV 一一对应。`【实测】`

> 注：`file_manifest.csv` 带 UTF-8 BOM（首字段显示为 `﻿layer`）。读取时必须用 `encoding="utf-8-sig"`，否则列名不匹配。已在实现中固定。

---

## 3. 数据口径复算（逐项对基准）

| 指标 | 已报告 | 本轮复算 | 结论 |
|---|---:|---:|---|
| ODS 原始行数 | 676,931 | 676,931 | ✅ |
| ODS 视频数 | 508 | 508 | ✅ |
| ODS 快照数（bvid×batch） | 509（文中未列） | 509 | ✅（1 支视频跨两批） |
| dmid 唯一值 | 674,498 | 674,498 | ✅ |
| 重复 dmid ID / 行 | 2,433 / 4,866 | 2,433 / 4,866 | ✅ |
| 同一 dmid 跨 bvid | 0 | 0 | ✅ |
| 重复所属视频 | BV1mwY96jEom | BV1mwY96jEom | ✅ |
| DWD `quality_flag=ok` | 672,439 | 672,439 | ✅ |
| market-pool `video_metrics` | 669,995 | 669,995（DWD 672,439 − 2,444） | ✅ |
| 旧批次独有行 | 11 | 11 | ✅ |
| D-train 视频 | 175 | 175（ID 清单逐条比对一致） | ✅ |
| D-dev 视频 | 234 | 234（同上） | ✅ |
| 交集 | 0 | 0 | ✅ |

分批复算（`【实测】`）：

| batch | ODS 行 | ODS 视频 | 集合 |
|---|---:|---:|---|
| 20260902 | 129,976 | 91 | D-train |
| 20260907 | 181,018 | 118 | D-train |
| 20260914 | 249,685 | 200 | D-dev |
| 20260917 | 116,252 | 100 | D-dev |

**时间字段体检** `【实测】`：`time_offset` 缺失 0、负值 0、`> duration` 0；`status` 全部为 `ok`（676,931 行）。分桶时 `offset == duration` 且 `duration` 为 5 的整数倍的行 **4 条**，按"归入最后一桶 K−1"处理并计数。

**切分规则的内生事实**：`D_train_end = 2026-09-07`（= max(D-train 批次)）。两侧视频集合由**两个独立调用**分别构造后求交，交集为空；不是 `tr = ~te` 的恒等式。`【实测】`

---

## 4. dmid 映射：ODS 无 `dmid` 字段

`【实测】` ODS Parquet schema 为：

```
bvid, cid, up_mid, up_name, title, duration, play, danmaku_reported, danmaku_got,
like, coin, favorite, share, reply, pubdate, time_offset, send_time, mode, font_size,
color, text, text_len, pool, row_id, status, source, tname, part_count, part_cids
```

**没有 `dmid` 列。** 与 dmid 语义对应的是 `row_id`：

1. 全部 676,931 行 `row_id` 均为 19 位数字串（`dmid` 的典型形态）；
2. 跨批次同一条弹幕的 `row_id` 相同 → 才可能产生 2,433 个跨批重复 ID 且"内容相同"；
3. 同一 `row_id` 不跨 bvid（0 组）。

⇒ 实现中以 `dmid := row_id` 作为正式去重键，并把该映射写入协议 §5.1。`【口径 + 实测】`

**去重范围的一处必要澄清**：2,433 个重复 dmid 全部是 `BV1mwY96jEom` 在 0914 与 0917 两批各出现一次。`【实测】` 在本实现中，去重**按 (bvid, batch) 快照内部**执行——同一快照内重复 dmid 数 = **0**（509 条序列的 `dropped_duplicate_dmid` 合计为 0）。跨快照的同一 dmid 保留，因为两个快照是两次独立观测（与"保留跨批重复观测、按 bvid 聚类推断"一致）。
> 若按**全局** dmid 去重，0914 或 0917 之一会被削掉 2,433 行，该快照的桶序列不再对应任何一次真实采集。**这一条需要在协议中写明**（见 rc4 候选 §5.2）。

---

## 5. 11 行旧批独有审计

`【实测】`：`old_batch_only_11_rows.csv` 共 11 行，全部 `bvid=BV1mwY96jEom`、全部 `dt=20260914`。

- 这 11 行在 0917 快照中不存在 ⇒ 若按"latest per bvid"建池，它们会随 0914 快照一起被丢弃。
- 2,444 = 2,433（跨批重复）+ 11（旧批独有）⇒ `672,439 − 2,444 = 669,995` 与已报告的 pool 行数一致。`【实测】`
- **判定：不是数据丢失，是"保留最新快照"规则的确定性后果。** 主分析按快照建模，因此这 11 行在 0914 序列中**仍然参与**，不受影响。

---

## 6. 差异登记（冻结前必须裁定）

### 差异 1：已报告的 304 / 133 / 171 只能用严格 `>` 复现，与批准的 `>=` 冲突

`【实测】` 在 `B=5, PRE=12, H=6, step=6, K=ceil(duration/5), duration>=120, 门槛(≥5点, ≥2正, ≥2负)` 下：

| 阈值比较 | 全池可评估 | D-train | D-dev |
|---|---:|---:|---:|
| `c[k] >= thr_t`（**批准规则**） | **306** | **134** | **172** |
| `c[k] > thr_t`（严格大于） | **304** | **133** | **171** |

已报告的 **304 / 133 / 171 与严格 `>` 逐项精确吻合**。⇒ 该组数字来自一个把边界判为"不命中"的实现。

批准规则写的是 `>=`（任务书 §5.3：`c[k] >= threshold_t`；协议 §4.3 的 T4 用例也要求"阈值恰为 3 时 `c=3` 判正"）。因此：

- **正式实现采用 `>=`**（已固化在 `label_points()`，并有单测 `test_t4_exactly_at_threshold`）。
- 正确计数应为 **306 / 134 / 172**，已报告的 304 / 133 / 171 **需要更正**，不得继续引用。

### 差异 2：manifest 的 count-only 表（13,049 / 4,667 / rate 0.3577）无法用单一变体精确复现

`【实测】` 在 4 种候选定义下：

| 定义 | 点数 | 正例 | 正例率 |
|---|---:|---:|---:|
| 全快照 509 条 + `>=`（无时长过滤） | 13,268 | 4,745 | 0.3576 |
| `duration>=120` 的 410 条快照 + `>=` | 13,224 | 4,734 | 0.3580 |
| pool（每视频取最新快照，409 条）+ `>=` | **13,107** | **4,689** | **0.3577** |
| `duration>=120` + `>` | 13,224 | 4,521 | 0.3419 |
| pool + `>` | 13,107 | 4,479 | 0.3417 |
| **已报告** | **13,049** | **4,667** | **0.3577** |

- 正例率与 `pool + >=` **完全一致（0.3577）**，点数差 58、正例差 22；
- H=12 时点数为 6,516、正例率 0.5068 ⇒ **manifest 用的是 H=6，不是 H=12**。

⇒ 差异幅度小但无法归零，**未定位**（最可能来自 pool 构建所用的 `video_metrics` 时长/点数口径与本轮 ODS `duration` 的细微差异）。处置：**不重写 manifest**，把该表标记为"参考值、非冻结口径"，以本报告 §6 与本轮 `audit.json` 为准。

### 差异 3：`part_count` 全空 ⇒ 多 P 排除规则不可执行

`【实测】` 4 个批次的 `part_count` 全部为空（509 条序列中 `part` 解析为 0 的有 509 条；`part_count == 1` 的视频数 = 0）。

⇒ 协议 §5.4「主分析排除 `part_count > 1`」**在本数据上无法执行**（不是"已排除 0 支"，而是"字段不可用"）。可选处置：(a) 声明本 4 批无多 P 元数据、按未标注处理并写入局限；(b) 改用 `cid` 去重检测多 P（本 4 批每视频 `cid` 数是否为 1 需另测）。**需研究者裁定，计入阻塞项 B-03。**

---

## 7. 复现命令

```bash
# 1) 原始哈希核验（20/20）
cd ~/projects/danmaku && sha256sum -c docs/exp35_508/manifest.sha256

# 2) 模块化核验（含字节数 + 生成 CSV）
cd ~/projects/danmaku-v2
~/danmaku-venv/bin/python jobs/m2_l6_burst_risk.py --stage verify-manifest

# 3) 全量口径 + 切分 + 资格复核
~/danmaku-venv/bin/python jobs/m2_l6_burst_risk.py --stage audit
```

本轮产物：

- `data-out/m2_l6/ods/manifest_verify.json`
- `data-out/m2_l6/ods/audit.json`（含 509 条序列的丢弃计数、逐文件 provenance 与 sha256）
- `docs/exp35_508/audit_manifest_files.csv`

---

## 8. 本报告不覆盖的内容

1. 未核验 `data-out/` 之外的历史产物（旧探针脚本、`_audit_scratch/`）；它们**不得**作为正式证据。
2. 未核验 DWD 之外的其它层（`dws` 未使用）。
3. 未做任何 D-conf 相关工作：当前项目中不存在 `dt >= 20261027` 的 ODS 分区，实现层对此有路径拒绝规则与单测（`tests/test_burst_isolation.py`）。
4. 本轮所有数字均为**数据事实**，不含任何模型性能。
