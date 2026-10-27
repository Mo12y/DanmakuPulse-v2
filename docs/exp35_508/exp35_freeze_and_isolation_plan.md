# exp35 文件隔离与冻结验收方案

- 日期：2026-10-27
- 适用：`danmaku-v2` 仓，工作区存在 **22 个与本实验无关的既有修改**（`git_status_clean = false` 恒成立）
- 性质：**方案**。本文件不执行冻结；`frozen_at` 仍为空
- 依据：`35_formal_spec_rc5_candidate.md` §11

---

## 1. 问题

标准冻结要求 `git_status_clean == true`（rc2.1 §3.3-5），否则"commit 不代表运行代码"。但本仓有 22 个无关的已跟踪修改（`Makefile`、`jobs/ads_highlights.py`、`serve/*`、`tests/accept_auth.sh` 等），本轮**不允许清理、恢复或重置**。

⇒ 需要一种**不依赖工作区洁净**的凭据机制，且能证明"跑的就是冻结的那份代码"。

## 2. 方案：路径级冻结凭据（path-scoped freeze）

对**显式文件清单**逐个算 sha256，再对排序后的 `path:sha256` 行求总摘要：

```
digest = sha256( concat( sorted("path:sha256\n") ) )
```

清单（13 项，与 `jobs/m2_l6_burst_risk.py` 的 `EXP35_FREEZE_PATHS` 一致）：

```
jobs/m2_l6_burst_risk.py
tests/test_burst_label.py
tests/test_burst_features.py
tests/test_burst_split.py
tests/test_burst_calibration.py
tests/test_burst_eval.py
tests/test_burst_isolation.py
docs/exp35_508/file_manifest.csv
docs/exp35_508/manifest.sha256
docs/exp35_508/D-train-508_bvids.txt
docs/exp35_508/D-dev-508_bvids.txt
docs/exp35_508/35_formal_spec_rc5_candidate.md
docs/exp35_508/exp35_freeze_and_isolation_plan.md
```

命令：`~/danmaku-venv/bin/python jobs/m2_l6_burst_risk.py --stage freeze-digest`
产物：`data-out/m2_l6/ods/freeze_digest.json`（含逐文件 bytes/sha256 与总摘要）

**为什么这样够用**：22 个无关文件不在清单内，它们无论怎么改都不影响 `digest`；清单内任一文件改一个字节，`digest` 必变。缺文件直接 `raise`（不允许"冻结时少一两个文件"）。

**局限（必须写进冻结记录）**：该机制只保证清单内 13 个文件一致；不保证 `serve/` 等运行时代码未被改动。因此冻结记录里必须写"本次实验不依赖 `serve/`"。

## 3. 冻结步骤（研究者批准后执行）

| 步 | 动作 | 判据 |
|---|---|---|
| 1 | 最终审计（`--stage audit` + `verify-manifest`） | 20/20 sha 通过；`eligible_videos` = 306；`eligible_videos_excl_known_multip` = 293 |
| 2 | 全量测试 | `pytest tests/test_burst_*.py -q` → **106 passed** |
| 3 | 重算 `freeze_digest` | 与第 1 步同一次运行内，两次调用摘要一致 |
| 4 | 填冻结记录（模板见 §5） | `frozen_at` 晚于所有 D-train 批次 `windows[-1].end` |
| 5 | 提交（**显式路径**）并打 tag `exp35-freeze-YYYYMMDD` | `git show --stat` 只含 exp35 路径 |
| 6 | 冻结后禁止改清单内文件 | 任何改动 ⇒ 新协议版本 + 新批次 |

## 4. D-conf 批次定义（冻结后）

一个批次成为 D-conf，必须**同时**满足：

1. **采集开始时间 > `frozen_at`**（用 `*.meta.json` 的 `windows[0].start`，不用文件名日期）；
2. `meta.status == "complete"`，且采集时无并发进程（`pgrep -f "bili.py hotlist"` 无匹配）；
3. `windows[-1].end` 非空且 `end > start`，日志含正常结束标记；
4. 通过完整性检查（rc2.1 §3.5 的九项，含台账 / 重复 / 窗口异常 / 缺失日 / 并发分类）；
5. 封版时复制到只读目录、在副本上重算哈希一致。

不满足任一条 ⇒ 该批次**不得**作 D-conf；可用于 D-train（若时间在 `D_train_end` 前）或直接弃用，结论写进报告。

**隔离的三道闸（已实现，有测试）**：

| 闸 | 机制 | 测试 |
|---|---|---|
| 批次 | `split_of()` 对 `>= 20261027` 与非登记批次 `raise` | `test_dconf_batches_rejected` / `test_unknown_batch_rejected_not_defaulted_to_train` |
| 路径 | `guard_path()` 拒绝 `dconf`、`dt=2026102[7-9]+`、`danmaku-v2/data` | `test_dconf_paths_rejected` / `test_v2_exploration_data_rejected` |
| 清单 | 只按 `file_manifest.csv` 白名单逐文件读，不通配目录 | `test_loader_reads_only_listed_files` |

## 5. 冻结记录模板（现在为空）

```
protocol_version : exp35-formal-rc5-candidate
frozen_at        : <ISO8601>
frozen_by        : <研究者>
git_commit       : <冻结提交 sha>
git_tag          : exp35-freeze-<YYYYMMDD>
freeze_digest    : <13 文件总摘要>
manifest_sha256  : <file_manifest.csv 的 sha256>
git_status_clean : false
unrelated_dirty  : <22 个文件的路径清单 + "与 exp35 清单无交集"的声明>
d_train          : 20260902, 20260907   (D_train_end = 2026-09-07)
d_dev            : 20260914, 20260917
dconf_batches    : <冻结后按 §4 的判据确定的批次列表>
test_evidence    : 106 passed / 0 failed  (命令 + 日期)
audit_evidence   : eligible 306 (134/172); excl_known_multip 293 (128/165)
```

## 6. 验收清单（逐条可判）

| # | 检查 | 命令 | 预期 |
|---|---|---|---|
| F1 | 13 个冻结路径全部存在 | `--stage freeze-digest` | `n_files=13`，退出 0 |
| F2 | 摘要可复现 | 连续两次运行 | `digest` 相同 |
| F3 | 清单内文件未被改动 | 冻结前再跑一次 | `digest` 与冻结记录一致 |
| F4 | manifest 20/20 | `--stage verify-manifest` | `all_ok=true` |
| F5 | 测试全绿 | `pytest tests/test_burst_*.py -q` | `106 passed` |
| F6 | 资格计数与协议一致 | `--stage audit` | 306 / 293 |
| F7 | D-conf 未被触碰 | `find data -newer <frozen_at>` 无 `dt>=20261027` | 空 |
| F8 | 22 个无关文件未被本实验改动 | `git status --short` 与冻结前快照比对 | 集合不变 |

## 7. 现在能做的 / 现在不能做的

- **能**：跑 F1/F2（除 rc5 与本文件尚未入仓外）、F4/F5/F6；这三项目前已通过。
- **不能**：F3/F7/F8 需要冻结时刻的参照，现在无参照；因此本方案**只是可执行的设计**，不构成冻结。
- 触发冻结的唯一条件：研究者对 `35_formal_spec_rc5_candidate.md` §12 的 B-02 / B-05 / B-07 三项签字。
