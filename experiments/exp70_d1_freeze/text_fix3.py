# -*- coding: utf-8 -*-
V2='/home/molly/projects/danmaku-v2'
p=f'{V2}/experiments/exp55_route_abc_verify/REPORT.md'
s=open(p,encoding='utf-8').read()
old='| 应用含义 | 热度余量预估（运营/推荐参考） |'
new='| 应用含义 | 互动量余量参考（不作热度/推荐决策依据） |'
assert s.count(old)==1
open(p,'w',encoding='utf-8').write(s.replace(old,new)); print('edited')
