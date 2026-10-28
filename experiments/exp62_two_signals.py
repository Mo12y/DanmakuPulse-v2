#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp62_two_signals.py — 两个从未被测过的信号

A 持续吸引点：同一视频的同一段进度，是否在后续批次**再次**收到弹幕（二刷/回访）？
   口径：取 n_pool2 >= n_pool1 的视频对（无池滚动污染），delta = 最新曲线 − 最早曲线（clip≥0），
        Spearman(首波形状, delta 形状)；>0 ⇒ 老位置继续吸新弹幕。
B 形状补全：前 25% 桶的**归一化形状**能否预测后 75% 的形状？
   任务与 exp55 路线 C 不同（那是绝对计数，被量主导；这里去量、只看形状）。
   训练/测试按视频 80/20（seed=20261028）；基线 = 全局均值形状、1NN；模型 = ridge(prefix→suffix)。
输出：experiments/exp62_two_signals/results.json
"""
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

REPO = '/home/molly/projects/danmaku-v2'
EXP = f'{REPO}/data-out/exp62'
OUT = f'{REPO}/experiments/exp62_two_signals'
os.makedirs(OUT, exist_ok=True)
SEED = 20261028
rng = np.random.default_rng(SEED)
PRE = 25
SUF = 75


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{d}/*.csv'))],
                     ignore_index=True)


bm = read_dir(f'{EXP}/batch_meta_csv')
cur = read_dir(f'{EXP}/curves_multi_csv')
bm['age_max_h'] = pd.to_numeric(bm['age_max_h'], errors='coerce')
bm['n_pool'] = pd.to_numeric(bm['n_pool'], errors='coerce')
cur['bucket1pct'] = pd.to_numeric(cur['bucket1pct'], errors='coerce')
cur['cnt'] = pd.to_numeric(cur['cnt'], errors='coerce')

# 每支视频的批次按采集时间（age_max）排序
bm = bm.sort_values(['bvid', 'age_max_h'])
pairs, early_meta = [], []
for bv, d in bm.groupby('bvid'):
    if len(d) < 2:
        continue
    e, l = d.iloc[0], d.iloc[-1]
    pairs.append((bv, e.batch_date, l.batch_date, e.n_pool, l.n_pool, e.age_max_h, l.age_max_h))
P = pd.DataFrame(pairs, columns=['bvid', 'b_e', 'b_l', 'n_e', 'n_l', 'age_e', 'age_l'])
P['no_roll'] = P.n_l >= P.n_e * 0.995
print('multi-batch videos:', len(P), ' no-roll pairs:', int(P.no_roll.sum()), flush=True)

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
    s = v.sum()
    return v / s if s > 0 else v


# ============================== A 持续吸引点 ==============================
rhos, rhos_vs_base = [], []
n_used, nz = 0, 0
for r in P.itertuples():
    if not r.no_roll:
        continue
    v1 = vec(r.bvid, r.b_e)
    v2 = vec(r.bvid, r.b_l)
    delta = np.clip(v2 - v1, 0, None)
    if delta.sum() < 10 or v1.sum() < 50:
        continue
    n1, nd = norm(v1), norm(delta)
    nz += int(v1.sum() > 0)
    if nd.std() == 0 or n1.std() == 0:
        continue
    rx = pd.Series(n1).rank().values
    ry = pd.Series(nd).rank().values
    rhos.append(float(np.corrcoef(rx, ry)[0, 1]))
    n_used += 1
rhos = np.array(rhos)
A = dict(n_videos=n_used,
         spearman_median=round(float(np.median(rhos)), 4) if len(rhos) else None,
         share_pos=round(float((rhos > 0).mean()), 3) if len(rhos) else None,
         share_gt_03=round(float((rhos > 0.3).mean()), 3) if len(rhos) else None,
         interpretation='rho>0 ⇒ 老的高密度位置继续吸收新弹幕（回访/二刷）；rho≈0 ⇒ 新弹幕与旧峰无关',
    debug_vec_nonzero=nz)
A['status'] = ('有初步支持证据' if len(rhos) >= 20 and A['share_pos'] and A['share_pos'] > 0.7
               and A['spearman_median'] and A['spearman_median'] > 0.2 else '结果不稳定')

# ============================== B 形状补全 ==============================
samples = []
for r in P.itertuples():
    if r.n_e < 100 or r.n_l < 300:
        continue                      # 早期太稀（<100）或晚期不足 300 ⇒ 形状噪声过大
    v = vec(r.bvid, r.b_e)
    nshape = norm(v)
    samples.append(dict(bvid=r.bvid, full=nshape))
print('shape-completion videos:', len(samples), flush=True)
B = None
if len(samples) >= 30:
    vids = [s['bvid'] for s in samples]
    Xall = np.array([s['full'][:PRE] / max(s['full'][:PRE].sum(), 1e-9) for s in samples])
    Yall = np.array([s['full'][PRE:] / max(s['full'][PRE:].sum(), 1e-9) for s in samples])
    order = rng.permutation(len(vids))
    ntr = int(len(vids) * 0.8)
    tr, te = order[:ntr], order[ntr:]
    glob_shape = Yall[tr].mean(0)
    pred_g = np.tile(glob_shape, (len(te), 1))
    maes = {'global_mean_shape': [], 'oneNN': [], 'ridge': []}
    sps = {'global_mean_shape': [], 'oneNN': [], 'ridge': []}
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0).fit(Xall[tr], Yall[tr])
    for j in te:
        y = Yall[j]
        preds = {'global_mean_shape': glob_shape,
                 'oneNN': Yall[tr[int(np.argmin(np.linalg.norm(Xall[tr] - Xall[j], axis=1)))]],
                 'ridge': np.clip(ridge.predict(Xall[j:j + 1])[0], 0, None)}
        for name, pv in preds.items():
            maes[name].append(float(np.abs(pv - y).mean()))
            sps[name].append(float(np.corrcoef(pv, y)[0, 1]) if y.std() > 0 else 0.0)
    B = dict(n_train=int(len(tr)), n_test=int(len(te)),
             prefix=PRE, suffix=SUF,
             MAE_x1000={k: round(float(np.mean(v)) * 1000, 2) for k, v in maes.items()},
             shape_corr_median={k: round(float(np.median(v)), 4) for k, v in sps.items()},
             skill_vs_global=round(1 - float(np.mean(maes['ridge'])) / float(np.mean(maes['global_mean_shape'])), 4),
             skill_1NN_vs_global=round(1 - float(np.mean(maes['oneNN'])) / float(np.mean(maes['global_mean_shape'])), 4))
    B['status'] = ('有初步支持证据' if B['skill_vs_global'] > 0.10 else '无增益（全局均值形状已是最优）')
    B['note'] = 'prefix/suffix 各自归一 ⇒ 与 exp55 路线 C（绝对计数）是不同任务'

R = dict(meta=dict(seed=SEED, multi_batch_bvids=int(len(P)), no_roll_pairs=int(P.no_roll.sum())),
         signal_A_persistent_attraction=A, signal_B_shape_completion=B)
json.dump(R, open(f'{OUT}/results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1)[:2200])
print('EXP62-DONE')
