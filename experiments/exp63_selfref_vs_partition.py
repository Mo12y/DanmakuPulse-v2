#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp63_selfref_vs_partition.py — 自参照对标 vs 分区参照（配对比较）

问题：预测"新增弹幕落在哪些位置"时，用视频自己的早期快照，还是用同分区其他视频的中位形状，哪个更准？
样本：exp62 的 556 个无池滚动视频对（delta = 真实新增弹幕的桶分布）。
参照：self = 早期快照归一形状；partition = 该 tname 其他视频 100 点曲线的中位形状（exp56 全库计算）。
指标：Spearman(ref, delta) + Precision@10（top-10 桶重合度），视频级配对 bootstrap。
附带：形状补全任务里"分区形状 vs 全局形状"基线对照（补 exp62 的缺口）。
输出：experiments/exp63_selfref/results.json
"""
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

REPO = '/home/molly/projects/danmaku-v2'
EXP56, EXP62 = f'{REPO}/data-out/exp56', f'{REPO}/data-out/exp62'
OUT = f'{REPO}/experiments/exp63_selfref'
os.makedirs(OUT, exist_ok=True)
SEED = 20261028
rng = np.random.default_rng(SEED)


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{d}/*.csv'))],
                     ignore_index=True)


units = read_dir(f'{EXP56}/units_csv')
units['tname'] = units['tname'].astype(str)
tname_of = dict(zip(units.bvid, units.tname))
cur1 = read_dir(f'{EXP56}/curves1pct_csv')
cur1['bucket1pct'] = pd.to_numeric(cur1['bucket1pct'], errors='coerce').astype(int)
cur1['cnt'] = pd.to_numeric(cur1['cnt'], errors='coerce')

# 分区中位形状（含全部 2,218 支；对查询视频的自包含效应可忽略——分区 ≥20 支才计入）
dense = cur1.pivot_table(index='bvid', columns='bucket1pct', values='cnt', aggfunc='sum').fillna(0.0)
# P1-1 修复：留一法（排除自身）+ 最小分区样本 20（与生产 MIN_PARTITION 一致）
part_curves = {}
for t, g in units.groupby('tname'):
    bvs = [b for b in g.bvid if b in dense.index]
    if len(bvs) < 20:
        continue
    part_curves[t] = (bvs, dense.loc[bvs].values)
part_n = {t: len(b) - 1 for t, (b, _) in part_curves.items()}


def part_shape_loo(t, bv):
    bvs, M = part_curves[t]
    i = bvs.index(bv)
    return np.median(np.delete(M, i, axis=0), axis=0)

# ---------------- exp62 的无滚动对 + delta ----------------
bm = read_dir(f'{EXP62}/batch_meta_csv')
cur = read_dir(f'{EXP62}/curves_multi_csv')
bm['age_max_h'] = pd.to_numeric(bm['age_max_h'], errors='coerce')
bm = bm.sort_values(['bvid', 'age_max_h'])
pairs = []
for bv, d in bm.groupby('bvid'):
    if len(d) < 2:
        continue
    e, l = d.iloc[0], d.iloc[-1]
    if l.n_pool >= e.n_pool * 0.995 and e.n_pool >= 100 and l.n_pool >= 100:
        pairs.append((bv, e.batch_date, l.batch_date))
print('no-roll pairs:', len(pairs), flush=True)

mi = cur.set_index(['bvid', 'batch_date', 'bucket1pct'])['cnt']
_cache = {}


def vec(bv, b):
    if (bv, b) not in _cache:
        try:
            s = mi.xs((bv, b))
        except KeyError:
            s = pd.Series(dtype=float)
        _cache[(bv, b)] = np.array([float(s.get(i, 0.0)) for i in range(100)], float)
    return _cache[(bv, b)]


def norm(v):
    return v / v.sum() if v.sum() > 0 else v


def sp(a, b):
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1])


def p10(a, b):
    ta = set(np.argsort(a)[-10:]); tb = set(np.argsort(b)[-10:])
    return len(ta & tb) / 10.0


rows = []
for bv, be, bl in pairs:
    t = tname_of.get(bv)
    if t not in part_curves:
        continue
    E, L = vec(bv, be), vec(bv, bl)
    delta = norm(np.clip(L - E, 0, None))
    sref = norm(E)
    pref = norm(part_shape_loo(t, bv))
    if delta.sum() <= 0 or sref.std() == 0:
        continue
    rows.append(dict(bvid=bv, tname=t, rho_self=sp(sref, delta) or 0.0, rho_part=sp(pref, delta) or 0.0,
                     p10_self=p10(sref, delta), p10_part=p10(pref, delta)))
D = pd.DataFrame(rows)
print('evaluable pairs:', len(D), flush=True)

d_rho = (D.rho_self - D.rho_part).values
d_p10 = (D.p10_self - D.p10_part).values
bs_rho = [float(np.mean((D.rho_self - D.rho_part).values[rng.integers(0, len(D), len(D))])) for _ in range(1000)]
bs_p10 = [float(np.mean(d_p10[rng.integers(0, len(D), len(D))])) for _ in range(1000)]
out = dict(n_pairs=int(len(D)),
           self_ref=dict(rho_median=round(float(D.rho_self.median()), 4),
                         p10_median=round(float(D.p10_self.median()), 4)),
           partition_ref=dict(rho_median=round(float(D.rho_part.median()), 4),
                              p10_median=round(float(D.p10_part.median()), 4)),
           paired=dict(d_rho_median=round(float(np.median(d_rho)), 4),
                       d_rho_mean=round(float(np.mean(d_rho)), 4),
                       ci95=[round(float(np.percentile(bs_rho, 2.5)), 4), round(float(np.percentile(bs_rho, 97.5)), 4)],
                       share_self_better=round(float((d_rho > 0).mean()), 3),
                       d_p10_mean=round(float(np.mean(d_p10)), 4),
                       ci95_p10=[round(float(np.percentile(bs_p10, 2.5)), 4), round(float(np.percentile(bs_p10, 97.5)), 4)]),
           partition_sizes=dict(median=round(float(pd.Series(part_n).median()), 1),
                                n_partitions=len(part_n)),
           per_tname={t: dict(n_others=part_n[t],
                              d_rho=round(float(D[D.tname == t].rho_self.sub(D[D.tname == t].rho_part).median()), 4))
                      for t in sorted(set(D.tname)) if (D.tname == t).sum() >= 15},
           fix_note='P1-1：分区基线改为留一法（排除查询自身）且仅保留 n>=20 的分区（与生产 MIN_PARTITION 一致）')
out['verdict'] = ('自参照显著优于分区参照' if out['paired']['ci95'][0] > 0 and out['paired']['share_self_better'] > 0.6
                  else '自参照不优于分区参照' if out['paired']['ci95'][1] < 0 else '两者相当（CI 含 0）')
json.dump(out, open(f'{OUT}/results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('self_ref', 'partition_ref', 'paired', 'verdict')}, ensure_ascii=False, indent=1))

# ---------------- 附：形状补全任务的分区形状基线（补 exp62 缺口） ----------------
# exp62 B 的测试集重用：prefix→suffix；基线从"全局均值形状"换成"分区中位形状"
samples = []
for bv, be, bl in pairs:
    E = vec(bv, be)
    if E.sum() < 150:
        continue
    full = norm(E)
    samples.append(dict(bvid=bv, t=tname_of.get(bv), full=full))
if len(samples) >= 30:
    from sklearn.linear_model import Ridge
    vids = [s['bvid'] for s in samples]
    X = np.array([s['full'][:25] / max(s['full'][:25].sum(), 1e-9) for s in samples])
    Y = np.array([s['full'][25:] / max(s['full'][25:].sum(), 1e-9) for s in samples])
    Ts = np.array([s['t'] for s in samples])
    order = rng.permutation(len(vids))
    ntr = int(len(vids) * 0.8)
    tr, te = order[:ntr], order[ntr:]
    glob_shape = Y[tr].mean(0)
    ridge = Ridge(alpha=1.0).fit(X[tr], Y[tr])
    # 分区中位 suffix（训练集内）
    part_suffix = {}
    for t in set(Ts[tr]):
        m = Ts[tr] == t
        if m.sum() >= 3:
            part_suffix[t] = Y[tr][m].mean(0)
    mae_g, mae_p, mae_r = [], [], []
    for j in te:
        y = Y[j]
        t = Ts[j]
        cand = {'g': glob_shape,
                'p': part_suffix.get(t, glob_shape),
                'r': np.clip(ridge.predict(X[j:j + 1])[0], 0, None)}
        mae_g.append(float(np.abs(cand['g'] - y).mean()))
        mae_p.append(float(np.abs(cand['p'] - y).mean()))
        mae_r.append(float(np.abs(cand['r'] - y).mean()))
    out['shape_completion_partition_baseline'] = dict(
        n_test=int(len(te)),
        MAE_x1000=dict(global_shape=round(float(np.mean(mae_g)) * 1000, 2),
                       partition=round(float(np.mean(mae_p)) * 1000, 2),
                       ridge=round(float(np.mean(mae_r)) * 1000, 2)),
        skill_vs_global=dict(partition=round(1 - float(np.mean(mae_p)) / float(np.mean(mae_g)), 4),
                             ridge=round(1 - float(np.mean(mae_r)) / float(np.mean(mae_g)), 4)),
        note='补 exp62 缺口：分区形状是否优于全局形状（跨视频信息的下限）')
    json.dump(out, open(f'{OUT}/results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('[shape-completion+partition]', json.dumps(out['shape_completion_partition_baseline'], ensure_ascii=False), flush=True)
print('EXP63-DONE')
