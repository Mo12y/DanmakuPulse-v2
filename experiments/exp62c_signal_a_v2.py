#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp62c_signal_a_v2.py — 信号 A 重跑（P0-2 修复：仅用 row/身份交集验证过的纯增量对）

纯增量 = overlap_frac ≥ 0.999（早期弹幕全部保留在最新池中，delta = 真实新增）。
同时报告 overlap 分布（量化"计数近似"到底漏掉了多少滚动对）。
输出：experiments/exp62_two_signals/signal_a_v2.json
"""
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

REPO = '/home/molly/projects/danmaku-v2'
EXP62 = f'{REPO}/data-out/exp62'
OUT = f'{REPO}/experiments/exp62_two_signals'
SEED = 20261028
rng = np.random.default_rng(SEED)


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{d}/*.csv'))],
                     ignore_index=True)


rc = read_dir(f'{EXP62}/rollcheck_csv')
for k in ('nE', 'nL', 'overlap', 'nE_keys', 'overlap_frac'):
    rc[k] = pd.to_numeric(rc[k], errors='coerce')
cur = read_dir(f'{EXP62}/curves_multi_csv')
cur['bucket1pct'] = pd.to_numeric(cur['bucket1pct'], errors='coerce').astype(int)
cur['cnt'] = pd.to_numeric(cur['cnt'], errors='coerce')
mi = cur.set_index(['bvid', 'batch_date', 'bucket1pct'])['cnt']
_c = {}


def vec(bv, b):
    if (bv, b) not in _c:
        try:
            s = mi.xs((bv, b))
        except KeyError:
            s = pd.Series(dtype=float)
        _c[(bv, b)] = np.array([float(s.get(i, 0.0)) for i in range(100)], float)
    return _c[(bv, b)]


def norm(v):
    return v / v.sum() if v.sum() > 0 else v


def sp(a, b):
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1])


rc = rc.sort_values('overlap_frac')
dist_hist = {'pure(>=0.999)': int((rc.overlap_frac >= 0.999).sum()),
             '0.9-1.0': int(((rc.overlap_frac >= 0.9) & (rc.overlap_frac < 0.999)).sum()),
             '0.7-0.9': int(((rc.overlap_frac >= 0.7) & (rc.overlap_frac < 0.9)).sum()),
             '<0.7': int((rc.overlap_frac < 0.7).sum())}

pure = rc[rc.overlap_frac >= 0.999]
rhos, recs = [], []
for r in pure.itertuples():
    v1 = vec(r.bvid, r.b_e)
    v2 = vec(r.bvid, r.b_l)
    delta = np.clip(v2 - v1, 0, None)
    if delta.sum() < 10 or v1.sum() < 50:
        continue
    n1, nd = norm(v1), norm(delta)
    if n1.std() == 0 or nd.std() == 0:
        continue
    rho = sp(n1, nd)
    if rho is None:
        continue
    rhos.append(rho)
    recs.append(dict(bvid=r.bvid, rho=round(rho, 4)))
rhos = np.array(rhos)
bs = [float(np.median(rng.choice(rhos, len(rhos), True))) for _ in range(2000)]
res = dict(
    n_pairs_total=int(len(rc)), overlap_distribution=dist_hist,
    n_pure_increment=int(len(rhos)),
    spearman_median=round(float(np.median(rhos)), 4),
    ci95_median=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
    share_pos=round(float((rhos > 0).mean()), 3),
    share_gt_03=round(float((rhos > 0.3).mean()), 3),
    verdict=('纯增量口径下信号 A 成立（老位置持续吸收新弹幕）'
             if len(rhos) >= 20 and float((rhos > 0).mean()) > 0.7 else '纯增量口径下信号 A 不成立'),
    wording='正确表述：早期形状与后续新增分布的关联（纯增量对）；机制（回访/新观众/推荐）不可分',
    caveat='用户删除弹幕会被误判为滚动，无法区分；纯增量判定基于 (cid,send_time,offset,text) 身份交集',
)
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(f'{OUT}/signal_a_v2.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
pd.DataFrame(recs).to_csv(f'{OUT}/signal_a_pure_pairs.csv', index=False, encoding='utf-8-sig')
print(json.dumps({k: res[k] for k in ('n_pairs_total', 'overlap_distribution', 'n_pure_increment',
                                      'spearman_median', 'ci95_median', 'share_pos', 'share_gt_03',
                                      'verdict')}, ensure_ascii=False, indent=1))
print('EXP62C-DONE')
