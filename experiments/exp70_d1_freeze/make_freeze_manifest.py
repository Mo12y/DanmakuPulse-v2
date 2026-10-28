# -*- coding: utf-8 -*-
"""exp70 make_freeze_manifest.py — hash-pin the D1 limited freeze (no model/data change)."""
import hashlib
import json
import os

import pandas as pd

V2 = "/home/molly/projects/danmaku-v2"
OUT = f"{V2}/experiments/exp70_d1_freeze"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


files = {
    "experiments/exp70_d1_freeze/b1_coef.json": "B1 coefficients (existing exp55 fit, persisted)",
    "experiments/exp70_d1_freeze/ols6_eval_coef.json": "OLS-6 evaluation-model coefficients (train-only)",
    "experiments/exp70_d1_freeze/d1_test_predictions_395.csv": "395-row B1/OLS6/persistence predictions",
    "experiments/exp70_d1_freeze/exp55_test_rowkeys_395.csv": "395 (bvid,batch) keys from exp55 loop",
    "experiments/exp70_d1_freeze/artifact_manifest.json": "artifact metrics/keys/manifest",
    "experiments/exp70_d1_freeze/text_fix_log.json": "text/metadata edit log",
    "experiments/exp70_d1_freeze/D1_FREEZE.md": "freeze record",
    "experiments/exp70_d1_freeze/TEXT_ERRATA.md": "text/metadata errata",
    "experiments/exp38_integrate/models/d1_ols.json": "production D1 artifact (model string fixed; coef unchanged)",
    "experiments/exp38_integrate/audit_scan.py": "production D1 artifact generator (string fixed)",
    "serve/ml_forecast.py": "D1 API semantic boundary text",
    "serve/static/ml_forecast.html": "D1 prediction card text",
    "tests/test_ml_forecast.py": "D1 feature-contract test docstring",
    "docs/exp38_integrate/thesis_evidence_pack.md": "thesis evidence pack D1 text",
    "experiments/exp55_route_abc_verify/REPORT.md": "exp55 D1 report (3.4x/6h retracted)",
    "experiments/exp60_finalize/REPORT.md": "exp60 R6 summary (814.2/cross-batch corrected)",
    "experiments/exp60_finalize/r6.md": "exp60 R6 detail (814.2/cross-batch corrected)",
    "experiments/exp67_evidence_audit/D1_D2_AUDIT.md": "prior audit CI typo fixed",
    "experiments/exp68_d1_residual_bridge/d1_repro.json": "exp68 D1 reproduction",
    "experiments/exp68_d1_residual_bridge/results.json": "exp68 negative result",
    "experiments/exp69_d1_metric_audit/D1_METRIC_LEDGER.md": "D1 metric ledger",
    "experiments/exp69_d1_metric_audit/D1_AUDIT_REPORT.md": "D1 audit report",
    "experiments/exp69_d1_metric_audit/d1_metric_audit.json": "D1 audit machine-readable",
    "experiments/exp69_d1_metric_audit/rowset_check.json": "exp55/exp67 rowset check",
}
entry = {}
for rel, role in files.items():
    p = f"{V2}/{rel}"
    assert os.path.exists(p), rel
    entry[rel] = dict(role=role, sha256=sha(p), bytes=os.path.getsize(p))

# verification
pred = pd.read_csv(f"{OUT}/d1_test_predictions_395.csv", encoding="utf-8-sig")
keys = pd.read_csv(f"{OUT}/exp55_test_rowkeys_395.csv", encoding="utf-8-sig")
assert len(pred) == len(keys) == 395
assert set(zip(pred.bvid, pred.batch)) == set(zip(keys.bvid, keys.batch))
assert pred.bvid.nunique() == 354 and pred.bvid.duplicated().sum() == 41
before = json.load(open(f"{OUT}/d1_ols_before.json"))
after = json.load(open(f"{V2}/experiments/exp38_integrate/models/d1_ols.json"))
assert before["coef"] == after["coef"] and before["intercept"] == after["intercept"]
assert before["features"] == after["features"]
am = json.load(open(f"{OUT}/artifact_manifest.json"))
assert am["metrics"]["persistence_4xc0"]["MAE_orig"] == 814.24
assert am["metrics"]["B1"]["MAE_orig"] == 277.69
assert am["metrics"]["OLS6"]["MAE_orig"] == 238.39

manifest = dict(
    freeze_id="D1-FREEZE-20261028",
    status="FROZEN-LIMITED (documentation + hash manifest; NOT committed)",
    repo_head="500c7c7",
    created_at="2026-10-28",
    frozen_claim="头 6h 计数（age≥31h 快照回溯）→ (6h,30h] 弹幕量延续；R² 0.5607 / MAE 238.39；vs B1 ΔR² 0.1076 [0.0711,0.1484]、MAE −14.1%。不是 6h 在线预测。",
    not_frozen=["6h 在线预测", "3.4× 公平基线", "生产模型已验证", "R7 独立人工盲评", "节奏结构解释 D1 残差"],
    verification=dict(b1_rows=int(len(pred)), b1_bvid=int(pred.bvid.nunique()),
                      duplicate_bvid_rows=int(pred.bvid.duplicated().sum()),
                      keys_equal_exp55=True, d1_ols_coef_unchanged=True, d1_predict_unchanged=True,
                      tests="pytest tests/test_ml_forecast.py -q -> 11 passed, 2 skipped",
                      metrics=am["metrics"]),
    files=entry,
    external_working_tree_changes=[
        "M serve/static/diagnosis.html", "M tests/test_ml_critique_contract.py",
        "M tests/test_v3_ai_highlights.py", "?? docs/design/", "?? screenshots/"],
    note="No commit. External changes pre-existed/arrived during audit; not touched.",
)
json.dump(manifest, open(f"{OUT}/FREEZE_MANIFEST.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(dict(freeze_id=manifest["freeze_id"], n_files=len(entry),
                      verification=manifest["verification"]), ensure_ascii=False, indent=1))
print("EXP70-FREEZE-MANIFEST-DONE")
