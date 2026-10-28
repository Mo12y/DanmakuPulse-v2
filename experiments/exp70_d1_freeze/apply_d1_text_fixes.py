# -*- coding: utf-8 -*-
"""exp70 apply_d1_text_fixes.py — correct D1 wording/metadata only; no model behavior change.

Edits:
  - serve/ml_forecast.py docstring (6h -> retrospective head-6h)
  - serve/static/ml_forecast.html (6h / OLS-5f / stale ΔR²)
  - experiments/exp38_integrate/models/d1_ols.json model string (OLS-5f -> OLS-6)
  - experiments/exp38_integrate/audit_scan.py model string (generator)
  - tests/test_ml_forecast.py docstring
  - docs/exp38_integrate/thesis_evidence_pack.md (6h, split <=1019 -> <=1020, ΔR² label)
  - exp55/exp60 reports and r6.md (814.2 / 3.4x / 6h / cross-batch wording)
  - exp67 D1_D2_AUDIT.md CI typo (0.1200 -> 0.1484)

No coefficients, features, thresholds, splits, or predictions are changed.
"""
import hashlib
import json
import os
import shutil

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp70_d1_freeze"
LOG = []
os.makedirs(OUT, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def edit(rel, old, new, must=1):
    p = f"{V2}/{rel}"
    s = open(p, encoding="utf-8").read()
    n = s.count(old)
    assert n == must, f"{rel}: expected {must} occurrence(s), found {n}: {old[:60]!r}"
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    LOG.append(dict(file=rel, old=old[:160], new=new[:160], count=n, sha256=sha(p)))
    print("edited", rel)


# 0) snapshot d1_ols before edit
shutil.copyfile(f"{V2}/experiments/exp38_integrate/models/d1_ols.json", f"{OUT}/d1_ols_before.json")
before = json.load(open(f"{OUT}/d1_ols_before.json"))

# 1) serve docstring
edit("serve/ml_forecast.py",
     "  d1 = 「发布后 6h 时点、对 (6h,30h] 弹幕量的预测」，不是播放量/热度/质量预测；",
     "  d1 = 「头 6h 计数（来自 age≥31h 累计快照的回溯）对 (6h,30h] 弹幕量的预测」，不是 6h 在线预测/播放量/热度/质量预测；")

# 2) static page
edit("serve/static/ml_forecast.html",
     "    ④ D1 = 发布后 6h 时点对 (6h,30h] 弹幕量的预测（OLS-5f，时间切 ΔR²=+0.095 [0.045,0.144]）；",
     "    ④ D1 = 头 6h 计数（age≥31h 快照回溯）对 (6h,30h] 弹幕量的预测（OLS-6，时间切相对 B1 ΔR²=+0.108 [0.071,0.148]）；不是 6h 在线预测；")

# 3) production metadata string
edit("experiments/exp38_integrate/models/d1_ols.json",
     '  "model": "OLS-5f(renamed from OLS-6; 5 feature columns)",',
     '  "model": "OLS-6 (6 features: c0,n36,n13,n01,slope,dur; production full-window refit)",')

# 4) generator string
edit("experiments/exp38_integrate/audit_scan.py",
     'd1_art = dict(model="OLS-5f(renamed from OLS-6; 5 feature columns)", features=STEPS[-1],',
     'd1_art = dict(model="OLS-6 (6 features: c0,n36,n13,n01,slope,dur; production full-window refit)", features=STEPS[-1],')

# 5) test docstring
edit("tests/test_ml_forecast.py",
     '    """生产特征列序必须与 exp38 训练口径（OLS-5f）一致。"""',
     '    """生产特征列序必须与 exp38 训练口径（OLS-6，6 特征）一致。"""')

# 6) thesis evidence pack
edit("docs/exp38_integrate/thesis_evidence_pack.md",
     '- **任务定义**：在视频发布后 6h 的决策时点，用此前已可观测的弹幕到达计数，预测 (6h, 30h] 窗口（"未来 24 小时"）的弹幕总量。样本单元 = (视频, 批次快照)，要求快照时龄 ≥31h（标签窗完整可观测）。',
     '- **任务定义**：用头 6h（0≤rel<6h）的弹幕到达计数，预测 (6h, 30h] 窗口（"未来 24 小时"）的弹幕总量。**这是 age≥31h 累计快照的回溯重建，不是 6h 当时快照，也不是 6h 在线预测。** 样本单元 = (视频, 批次快照)，要求快照时龄 ≥31h（标签窗完整可观测）。')
edit("docs/exp38_integrate/thesis_evidence_pack.md",
     "训练窗（first_seen ≤1019 批）918 单元",
     "训练窗（first_seen ≤1020 批）918 单元")
edit("docs/exp38_integrate/thesis_evidence_pack.md",
     "exp38（D1 ΔR²=+0.095 / D2 ΔAUC=+0.078，时间切）",
     "exp38（D1 OLS-6 vs B1 ΔR²=+0.108；HGB 主判据 +0.095；D2 ΔAUC=+0.078，时间切）")
edit("docs/exp38_integrate/thesis_evidence_pack.md",
     "> 总纪律：训练/评估只用 ≤20261026 批次与 0902/0907 旧批；未触碰 exp35 的 D-dev(0914/0917)/D-conf(1027+) 与任何冻结产物。",
     "> 总纪律：训练/评估只用 ≤20261026 批次与 0902/0907 旧批；未触碰 exp35 的 D-dev(0914/0917)/D-conf(1027+) 与任何冻结产物。\n>\n> **2026-10-28 修订**：D1 的“6h 决策时点”改为“头 6h 计数回溯（快照 age≥31h）”，不是 6h 在线；训练切分更正为 `first_seen ≤20261020`；D1 主模型写 OLS-6 vs B1 ΔR²=+0.108（HGB 主判据 +0.095 保留）。")

# 7) exp55 report
edit("experiments/exp55_route_abc_verify/REPORT.md",
     "发布后 6h 时点预测", "头 6h 计数（age≥31h 快照回溯）预测")
edit("experiments/exp55_route_abc_verify/REPORT.md",
     "| 基线 | 持久性 4×c0、历史均值 |", "| 基线 | 4×c0 未校准外推、历史均值、B1 |")
edit("experiments/exp55_route_abc_verify/REPORT.md",
     "| 相对基线增益 | vs 持久性 **−72% MAE_orig**；vs 历史均值 −35% |",
     "| 相对基线增益 | vs 4×c0 未校准 **−71% MAE_orig**；vs B1 −14%；vs 历史均值 −35% |")
edit("experiments/exp55_route_abc_verify/REPORT.md",
     "  MAE_orig：OLS-6 238 < B1 278 < 历史均值 368 < 持久性 814——**OLS-6 比持久性基线好 3.4 倍**。",
     "  MAE_orig：OLS-6 238.4 < B1 277.7 < 历史均值 368.0 < 4×c0 814.2——**OLS-6 比 4×c0 未校准外推低 71% MAE；相对同口径 B1 低 14%。3.4× 不再作为公平基线表述（2026-10-28 修订）。**")
edit("experiments/exp55_route_abc_verify/REPORT.md",
     "限定语＝日批热榜视频、6h 决策时点、(6h,30h] 窗口。",
     "限定语＝日批热榜视频、头 6h 计数回溯（快照 age≥31h）、(6h,30h] 窗口；不是 6h 在线预测。")
edit("experiments/exp55_route_abc_verify/REPORT.md",
     "明确 D1 的适用边界（日批热榜、6h 时点、量预测≠阶段识别）",
     "明确 D1 的适用边界（日批热榜、头 6h 计数回溯、量预测≠阶段识别；不是 6h 在线）")
edit("experiments/exp55_route_abc_verify/REPORT.md",
     "> → `experiments/exp55_route_abc_verify/results.json`。运行：`~/danmaku-venv/bin/python experiments/exp55_route_abc_verify.py`（约 4 分钟）。",
     "> → `experiments/exp55_route_abc_verify/results.json`。运行：`~/danmaku-venv/bin/python experiments/exp55_route_abc_verify.py`（约 4 分钟）。\n>\n> **2026-10-28 修订**：撤回“3.4× 持久性”表述；814.2 是未校准的 `4×c0` 外推，公平基线是 B1=277.7（OLS-6 低 14%）；“6h 决策时点”改为“头 6h 计数回溯（快照 age≥31h）”。")

# 8) exp60 report
edit("experiments/exp60_finalize/REPORT.md",
     "原始 MAE **238.4** vs 持久性 **814.2**；跨批 1.17×",
     "原始 MAE **238.4** vs 4×c0 **814.2**（未校准，不作公平基线；vs B1 277.7 低 14%）；跨首见日 1.17×")
edit("experiments/exp60_finalize/REPORT.md",
     "6. D1 量预测有效且跨批稳定（R² 0.561、原始 MAE 238、优于持久性 3.4 倍）。",
     "6. D1 量预测有效且跨首见日队列稳定（R² 0.561、原始 MAE 238、相对 B1 低 14%；4×c0 814.2 仅算术参照）。")
edit("experiments/exp60_finalize/REPORT.md",
     "> 复跑：`~/danmaku-venv/bin/python experiments/exp60_finalize.py`（seed=20261028，约 3 分钟）。",
     "> 复跑：`~/danmaku-venv/bin/python experiments/exp60_finalize.py`（seed=20261028，约 3 分钟）。\n>\n> **2026-10-28 修订**：R6 的“814.2 持久性”“3.4×”“跨批 1.17×”已更正为“4×c0 未校准参照”“相对 B1 低 14%”“跨首见日队列”。")

# 9) r6.md
edit("experiments/exp60_finalize/r6.md",
     "| 基线（原始 MAE） | 持久性 4×c0 | **814.2**（OLS-6 好 3.4 倍） |",
     "| 基线（原始 MAE） | 4×c0 未校准外推 | **814.2**（仅算术参照；公平基线 B1 277.7，OLS-6 低 14%） |")
edit("experiments/exp60_finalize/r6.md",
     "| | B1 单特征 | 277.7 |", "| | B1 单特征（公平基线） | 277.7 |")
edit("experiments/exp60_finalize/r6.md",
     "**跨批稳定性**：测试批 MAE 69–279，**spread 1.17×**（1022 批 n=11 仅参考）。",
     "**跨首见日队列稳定性**：首见日 MAE 69–279，**spread 1.17×**（1022 组 n=11 仅参考；不是按快照批次）。")
edit("experiments/exp60_finalize/r6.md",
     "**切分核验**：训练 918 / 测试 395，`first_seen ≤20261020` vs `≥20261021`，**bvid 交集 = 0**（代码 assert）。",
     "**切分核验**：训练 918 / 测试 395，`first_seen ≤20261020` vs `≥20261021`，**bvid 交集 = 0**（代码 assert）。\n> **2026-10-28 修订**：4×c0 只作未校准算术参照；公平基线 B1=277.7；跨批改为跨首见日。")

# 10) exp67 CI typo
edit("experiments/exp67_evidence_audit/D1_D2_AUDIT.md",
     "ΔR²=+0.1076（CI [0.0711,0.1200]）", "ΔR²=+0.1076（CI [0.0711,0.1484]）")

# verify d1_ols coefficients unchanged
after = json.load(open(f"{V2}/experiments/exp38_integrate/models/d1_ols.json"))
assert after["coef"] == before["coef"] and after["intercept"] == before["intercept"]
assert after["features"] == before["features"]
assert after["model"] != before["model"]

log = dict(applied_at="2026-10-28", repo_head="500c7c7", edits=LOG,
           d1_ols=dict(before_sha256=sha(f"{OUT}/d1_ols_before.json"),
                       after_sha256=sha(f"{V2}/experiments/exp38_integrate/models/d1_ols.json"),
                       coef_unchanged=True, intercept_unchanged=True, features_unchanged=True,
                       before_model=before["model"], after_model=after["model"]))
json.dump(log, open(f"{OUT}/text_fix_log.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(dict(n_edits=len(LOG), d1_ols=log["d1_ols"]), ensure_ascii=False, indent=1))
print("EXP70-TEXT-FIX-DONE")
