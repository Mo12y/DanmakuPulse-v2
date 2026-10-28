# -*- coding: utf-8 -*-
import hashlib
V2='/home/molly/projects/danmaku-v2'
def edit(rel,old,new):
    p=f'{V2}/{rel}'; s=open(p,encoding='utf-8').read(); assert s.count(old)==1,(rel,s.count(old),old)
    open(p,'w',encoding='utf-8').write(s.replace(old,new)); print('edited',rel)
edit('experiments/exp55_route_abc_verify/REPORT.md',
     '| 新增实验 | MAE/RMSE（log1p+原始尺度）+ 持久性/历史均值基线 |',
     '| 新增实验 | MAE/RMSE（log1p+原始尺度）+ 4×c0/历史均值基线 |')
edit('docs/exp38_integrate/thesis_evidence_pack.md',
     '训练窗（first_seen ≤1020 批）918 单元 / 测试窗（first_seen ≥1021 批）395 单元',
     '训练窗（first_seen ≤20261020）918 单元 / 测试窗（first_seen ≥20261021）395 单元')
