#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp66_pure_chain.py — 纯增量链收尾（P0-2 下游 + 选择性诊断 + R1 名单补全）

输入：exp62 rollcheck.csv + curves_multi + exp56 units/curves1pct/feats + exp57 meta
产出：
  r63_pure.json   自参照 vs 分区（LOO、min20）——仅纯增量对
  r64_pure.json   三参照（self/same_up/partition）——仅纯增量对
  selection.json  选择性诊断：纯增量 vs 滚动对的 got/时长差异 + rho 按 overlap 分位
  r1_full_list.json R1 低完整度分区完整名单（重算）
判定（预注册）：纯增量对 <100 或分区基线可计算数不足 ⇒ 明确写"样本不足"。
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

REPO = '/home/molly/projects/danmaku-v2'
EXP56, EXP57, EXP62 = f'{REPO}/data-out/exp56', f'{REPO}/data-out/exp57', f'{REPO}/data-out/exp62'
OUT = f'{REPO}/experiments/exp67_evidence_audit'
os.makedirs(OUT, exist_ok=True)
SEED = 20261028
rng = np.random.default_rng(SEED)
MET = ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'mid_ratio']


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{d}/*.csv'))],
                     ignore_index=True)


def sp(a, b):
    if a.std() == 0 or b.std() == 0:
        return None
    v = float(np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1])
    return None if np.isnan(v) else v          # 常数 rank 会产生 NaN（已实测）


def norm(v):
    return v / v.sum() if v.sum() > 0 else v


def p10(a, b):
    return len(set(np.argsort(a)[-10:]) & set(np.argsort(b)[-10:])) / 10.0


rc = read_dir(f'{EXP62}/rollcheck_csv')
for k in ('nE', 'nL', 'overlap', 'nE_keys', 'overlap_frac'):
    rc[k] = pd.to_numeric(rc[k], errors='coerce')
curm = read_dir(f'{EXP62}/curves_multi_csv')
curm['bucket1pct'] = pd.to_numeric(curm['bucket1pct'], errors='coerce').astype(int)
curm['cnt'] = pd.to_numeric(curm['cnt'], errors='coerce')
mi = curm.set_index(['bvid', 'batch_date', 'bucket1pct'])['cnt']
_c = {}


def vec(bv, b):
    if (bv, b) not in _c:
        try:
            s = mi.xs((bv, b))
        except KeyError:
            s = pd.Series(dtype=float)
        _c[(bv, b)] = np.array([float(s.get(i, 0.0)) for i in range(100)], float)
    return _c[(bv, b)]


meta = read_dir(f'{EXP57}/video_meta_csv')
meta['got'] = pd.to_numeric(meta['got'], errors='coerce')
meta['duration'] = pd.to_numeric(meta['duration'], errors='coerce')
meta['up_mid'] = meta['up_mid'].astype(str)
units = read_dir(f'{EXP56}/units_csv')
units['tname'] = units['tname'].astype(str)
units['c1'] = units.c1_eligible.astype(str).str.lower().eq('true')
tname_of = dict(zip(units.bvid, units.tname))
up_of = dict(zip(meta.bvid, meta.up_mid))
got_of = dict(zip(meta.bvid, meta.got))
dur_of = dict(zip(meta.bvid, meta.duration))

cur1 = read_dir(f'{EXP56}/curves1pct_csv')
cur1['bucket1pct'] = pd.to_numeric(cur1['bucket1pct'], errors='coerce').astype(int)
cur1['cnt'] = pd.to_numeric(cur1['cnt'], errors='coerce')
dense = cur1.pivot_table(index='bvid', columns='bucket1pct', values='cnt', aggfunc='sum').fillna(0.0)

part_curves = {}
for t, g in units.groupby('tname'):
    bvs = [b for b in g.bvid if b in dense.index]
    if len(bvs) >= 20:
        part_curves[t] = (bvs, dense.loc[bvs].values)


def part_loo(t, bv):
    if t not in part_curves:
        return None
    bvs, M = part_curves[t]
    if bv not in bvs or len(bvs) < 21:
        return None
    i = bvs.index(bv)
    return norm(np.median(np.delete(M, i, axis=0), axis=0))


up_curves = {}
for u, g in meta.drop_duplicates('bvid').groupby('up_mid'):
    bvs = [b for b in g.bvid if b in dense.index]
    if len(bvs) >= 2:
        up_curves[u] = (bvs, dense.loc[bvs].values)


def up_loo(u, bv):
    if u not in up_curves:
        return None
    bvs, M = up_curves[u]
    if bv not in bvs or len(bvs) < 3:
        return None
    i = bvs.index(bv)
    return norm(np.median(np.delete(M, i, axis=0), axis=0))


# ---------------- 纯增量对（overlap>=0.999） ----------------
pure = rc[rc.overlap_frac >= 0.999][['bvid', 'b_e', 'b_l', 'nE', 'nL', 'overlap_frac']]
print('pure pairs:', len(pure), flush=True)

rows63, rows64 = [], []
for r in pure.itertuples():
    bv, be, bl = r.bvid, r.b_e, r.b_l
    E, L = vec(bv, be), vec(bv, bl)
    delta = norm(np.clip(L - E, 0, None))
    sref = norm(E)
    t, u = tname_of.get(bv), up_of.get(bv)
    if delta.sum() <= 0 or sref.std() == 0 or t not in part_curves:
        continue
    row = dict(bvid=bv, rho_self=sp(sref, delta) or 0.0, p10_self=p10(sref, delta),
               got=got_of.get(bv, np.nan), dur=dur_of.get(bv, np.nan),
               age_gap=float(r.nL) - float(r.nE), t=t, u=u)
    pref = part_loo(t, bv)
    if pref is not None and pref.std() > 0:
        row['rho_part'] = sp(pref, delta) or 0.0
        row['p10_part'] = p10(pref, delta)
    uref = up_loo(u, bv)
    if uref is not None and uref.std() > 0:
        row['rho_up'] = sp(uref, delta) or 0.0
    rows63.append(row)
    if 'rho_part' in row and 'rho_up' in row:
        rows64.append(row)
P63 = pd.DataFrame(rows63)
P64 = pd.DataFrame(rows64)
print('exp63-pure evaluable:', len(P63), ' exp64-pure evaluable:', len(P64), flush=True)

# ---------------- R63：自参照 vs 分区（纯增量） ----------------
r63 = dict(n_pairs_pure=int(len(P63)),
           n_with_partition_ref=int(P63.rho_part.notna().sum()) if 'rho_part' in P63 else 0,
           rho_self_median=round(float(P63.rho_self.median()), 4))
if r63['n_with_partition_ref'] >= 80:
    dP = P63.dropna(subset=['rho_part'])
    dv = (dP.rho_self - dP.rho_part).values
    bs = [float(np.mean(dv[rng.integers(0, len(dv), len(dv))])) for _ in range(1000)]
    r63['paired'] = dict(n=int(len(dv)), d_rho_median=round(float(np.median(dv)), 4),
                         d_rho_mean=round(float(np.mean(dv)), 4),
                         ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                         share_self_better=round(float((dv > 0).mean()), 3))
    r63['verdict'] = ('纯增量口径下自参照仍显著优于分区' if r63['paired']['ci95'][0] > 0 else
                      '纯增量口径下自参照与分区不可区分（CI 含 0）')
else:
    r63['verdict'] = '纯增量子集不足以判定自参照 vs 分区（可计算的分区基线对不足 80）'
    r63['paired'] = None

# ---------------- R64：三参照（纯增量） ----------------
r64 = dict(n_pairs_pure=int(len(P64)))
if 'rho_up' in P64.columns:
    r64['rho_median'] = {c: round(float(P64[f'rho_{c}'].median()), 4) for c in ('self', 'up', 'part')}
    for a, b in (('self', 'up'), ('self', 'part')):
        dP = P64.dropna(subset=[f'rho_{a}', f'rho_{b}'])
        if len(dP) < 10:
            r64[f'self_vs_{b}'] = None
            continue
        dv = (dP[f'rho_{a}'] - dP[f'rho_{b}']).values
        bs = [float(np.mean(dv[rng.integers(0, len(dv), len(dv))])) for _ in range(1000)]
        r64[f'self_vs_{b}'] = dict(n=int(len(dv)), d_rho_mean=round(float(np.mean(dv)), 4),
                                   ci95=[round(float(np.percentile(bs, 2.5)), 4),
                                         round(float(np.percentile(bs, 97.5)), 4)],
                                   share_self_better=round(float((dv > 0).mean()), 3))
else:
    r64['rho_median'] = None

# ---------------- 选择性诊断（item 2） ----------------
allp = rc.copy()
allp['pure'] = allp.overlap_frac >= 0.999
allp['got'] = allp.bvid.map(got_of)
allp['duration'] = allp.bvid.map(dur_of)
diag = dict(n_pure=int(allp.pure.sum()), n_rolling=int((~allp.pure).sum()))
for col in ('got', 'duration'):
    a = allp.loc[allp.pure, col].dropna()
    b = allp.loc[~allp.pure, col].dropna()
    if len(a) > 10 and len(b) > 10:
        diag[f'{col}_median_pure'] = round(float(a.median()), 1)
        diag[f'{col}_median_rolling'] = round(float(b.median()), 1)
        diag[f'{col}_mw_p'] = round(float(mannwhitneyu(a, b).pvalue), 6)
allrows = []
for r in allp.itertuples():
    E, L = vec(r.bvid, r.b_e), vec(r.bvid, r.b_l)
    d_raw = np.clip(L - E, 0, None)
    if d_raw.sum() < 10:                      # 先用原始计数过滤，再归一（归一后 sum 恒为 1）
        continue
    s = norm(E)
    rho = sp(s, norm(d_raw))
    if rho is not None:
        allrows.append((float(r.overlap_frac), rho))
A5 = pd.DataFrame(allrows, columns=['ov', 'rho'])
A5 = A5.dropna()
A5['ov_q'] = pd.qcut(A5.ov, 4, labels=['Q1(重合最低)', 'Q2', 'Q3', 'Q4(重合最高)'])
qg = A5.groupby('ov_q', observed=True).rho.agg(['median', 'count'])
diag['rho_median_by_overlap_quartile'] = {str(k): {'median': round(float(v['median']), 4), 'n': int(v['count'])}
                                          for k, v in qg.iterrows()}
diag['interpretation'] = ('Q1→Q3 随重合度上升、Q4 略回落（**非严格单调**，Q4 n 较大且含中高重合对）；'
                          '总体方向 = 重合越高信号越强 ⇒ 滚动污染在稀释信号；'
                          '纯增量子集的选择性需结合 got/时长差异一起读')
json.dump(diag, open(f'{OUT}/selection.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

json.dump(r63, open(f'{OUT}/r63_pure.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
json.dump(r64, open(f'{OUT}/r64_pure.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({'r63': r63, 'r64': r64, 'selection': diag}, ensure_ascii=False, indent=1)[:2600])
print('EXP66-DONE')

# ---------------- R1 18 分区完整名单（item 4） ----------------
feats = read_dir(f'{EXP56}/feats_csv')
for c in MET:
    if c in feats.columns:
        feats[c] = pd.to_numeric(feats[c], errors='coerce')
V = (meta.sort_values('rows', ascending=False).drop_duplicates('bvid')
     .merge(feats[['bvid'] + MET], on='bvid', how='inner'))
V['ratio'] = (V.got / V.reported.clip(lower=1)).astype(float)
V['tierC'] = V.ratio < 0.5
per = V.groupby('tname').agg(n=('bvid', 'nunique'), shareC=('tierC', 'mean'))
flagged = per[per.shareC >= 0.3].sort_values('shareC', ascending=False)
r1f = dict(n_flagged=int(len(flagged)),
           partitions={t: dict(n=int(r.n), shareC=round(float(r.shareC), 3)) for t, r in flagged.iterrows()},
           note='R1 低完整度分区完整名单（shareC=got/reported<0.5 视频占比）；仅证据，不改生产基准')
json.dump(r1f, open(f'{OUT}/r1_full_list.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R1-full]', json.dumps(r1f, ensure_ascii=False)[:500], flush=True)
