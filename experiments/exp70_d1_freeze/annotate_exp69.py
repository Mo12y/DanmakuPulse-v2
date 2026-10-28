# -*- coding: utf-8 -*-
V2='/home/molly/projects/danmaku-v2'
def edit(rel,old,new):
    p=f'{V2}/{rel}'; s=open(p,encoding='utf-8').read(); assert s.count(old)==1,(rel,s.count(old))
    open(p,'w',encoding='utf-8').write(s.replace(old,new)); print('edited',rel)
note='> **后续状态（2026-10-28）**：本文件记录的“6h / OLS-5f / 3.4×”与 `d1_ols.json` 元数据冲突，已按 `experiments/exp70_d1_freeze/TEXT_ERRATA.md` 修正；本文件保留修正前的审计证据。\n\n'
edit('experiments/exp69_d1_metric_audit/D1_AUDIT_REPORT.md','## 0. 四类结论摘要',note+'## 0. 四类结论摘要')
edit('experiments/exp69_d1_metric_audit/D1_METRIC_LEDGER.md','## A. 样本、目标与切分',note+'## A. 样本、目标与切分')
edit('experiments/exp69_d1_metric_audit/EVIDENCE_AND_OPEN_ISSUES.md','## 0. 工作区漂移（审计期间，非本轮产生）','> **后续状态（2026-10-28）**：6h/OLS-5f/3.4× 与 `d1_ols.json` 元数据已按 `experiments/exp70_d1_freeze/TEXT_ERRATA.md` 修正；本文件保留修正前的审计证据。\n\n## 0. 工作区漂移（审计期间，非本轮产生）')
