#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp59_retained_directions.py — 保留方向逐项验证（R1–R4、R6；R5/R7/R8 见 REPORT）

每节写独立 JSON：experiments/exp59_retained/r{1,2,3,4,6}_results.json
R7 另生成盲评材料包（r7_blind_pairs.csv 等）。
纪律：只用既有导出（exp56/exp57/exp55 口径），不重复训练生产模型，不改生产基准；
      预注册判断标准写在各节 PRE 注释里，先于结果。
"""
import csv
import glob
import json
import os

import numpy as np
import pandas as pd

REPO = '/home/molly/projects/danmaku-v2'
EXP56, EXP57 = f'{REPO}/data-out/exp56', f'{REPO}/data-out/exp57'
OUT = f'{REPO}/experiments/exp59_retained'
os.makedirs(OUT, exist_ok=True)
SEED = 20261028
rng = np.random.default_rng(SEED)


def read_dir(d, **kw):
    return pd.concat([pd.read_csv(p, **kw) for p in sorted(glob.glob(f'{d}/*.csv'))], ignore_index=True)


def spearman(x, y):
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def resid_on(v, ctrls):
    A = np.column_stack([np.ones(len(v))] + ctrls)
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def boot_ci(vals, B=1000):
    vals = np.asarray(vals, float)
    v = vals[~np.isnan(vals)]
    if len(v) < 10:
        return None
    bs = [float(np.mean(rng.choice(v, len(v), True))) for _ in range(B)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


meta = read_dir(f'{EXP57}/video_meta_csv', encoding='utf-8-sig')
feats = read_dir(f'{EXP56}/feats_csv', encoding='utf-8-sig')
for c in ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'mid_ratio', 'dm_per_min', 'avg_text_len']:
    if c in feats.columns:
        feats[c] = pd.to_numeric(feats[c], errors='coerce')
meta['got'] = pd.to_numeric(meta['got'], errors='coerce')
meta['reported'] = pd.to_numeric(meta['reported'], errors='coerce')
meta['duration'] = pd.to_numeric(meta['duration'], errors='coerce')
V = meta.sort_values('rows', ascending=False).drop_duplicates('bvid').merge(feats, on='bvid', how='inner')
MET = ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'mid_ratio']

# ================================================================ R1 数据完整性
# PRE：若 got/reported 低分组与高分组的核心指标中位差 < 0.25×分区 MAD，且分区基准相对变化 <5% → 判"稳健"
ratio = (V.got / V.reported.clip(lower=1)).astype(float)
V['ratio'] = ratio
r1 = dict(n_videos=int(len(V)),
          ratio_p10=round(float(ratio.quantile(.1)), 4), ratio_median=round(float(ratio.median()), 4),
          n_ratio_lt_0p5=int((ratio < 0.5).sum()), n_ratio_ge_0p9=int((ratio >= 0.9).sum()))
r1['spearman_ratio_vs_metric'] = {m: round(spearman(ratio, V[m]), 4) for m in MET}
lo, hi = V[ratio < 0.5], V[ratio >= 0.9]
r1['low_vs_high'] = {}
for m in MET:
    a, b = lo[m].dropna(), hi[m].dropna()
    if len(a) < 10 or len(b) < 10:
        r1['low_vs_high'][m] = None
        continue
    mad = float((V[m] - V[m].median()).abs().median() * 1.4826)
    diff = float(a.median() - b.median())
    r1['low_vs_high'][m] = dict(med_low=round(float(a.median()), 4), med_high=round(float(b.median()), 4),
                                diff=round(diff, 4), diff_in_mad=round(diff / mad, 3) if mad > 1e-9 else None,
                                n_low=len(a), n_high=len(b))
# 批次对比值的影响（η²）
def eta2(vals, groups):
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    ok = ~np.isnan(vals); vals, groups = vals[ok], groups[ok]
    if len(vals) < 10 or len(set(groups)) < 2:
        return None
    gm = vals.mean(); sst = ((vals - gm) ** 2).sum()
    ssb = sum(len(vals[groups == g]) * (vals[groups == g].mean() - gm) ** 2 for g in set(groups))
    return float(ssb / sst) if sst > 0 else None
r1['eta2_ratio_by_batch'] = round(eta2(V.ratio, V.batch_date), 4)
# 敏感性：分区基准在"全量"vs"仅高完整度(ratio≥0.9)"下的相对变化
def partition_med(df, m):
    return df.groupby('tname')[m].median()
sens = {}
for m in MET:
    a = partition_med(V, m)
    b = partition_med(V[V.ratio >= 0.9], m)
    common = a.index.intersection(b.index)
    rel = ((b[common] - a[common]).abs() / a[common].abs().clip(lower=1e-9))
    sens[m] = dict(n_partitions=int(len(common)), median_rel_change=round(float(rel.median()), 4),
                   p90_rel_change=round(float(rel.quantile(.9)), 4),
                   max_rel_change=round(float(rel.max()), 4))
r1['partition_baseline_sensitivity'] = sens
r1['quality_flag_note'] = 'quality_flag 仅存在于 dwd_danmaku_clean_min（4 legacy 批，全为 ok）→ 日批不可用，本轮以 got/reported 代替'
r1['status'] = '待判（见 diff_in_mad 与 median_rel_change）'
json.dump(r1, open(f'{OUT}/r1_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R1]', json.dumps({k: r1[k] for k in ('ratio_p10', 'n_ratio_lt_0p5', 'spearman_ratio_vs_metric',
                                            'eta2_ratio_by_batch')}, ensure_ascii=False), flush=True)

# ================================================================ R2/R3/R4（用 exp57 文本样本）
tx = read_dir(f'{EXP57}/text_sample_csv', encoding='utf-8-sig',
              dtype={'bvid': str, 'user_hash': str, 'tnorm': str, 'txt': str})
tx['bucket60'] = tx['bucket60'].astype(int)
tx['text_len'] = pd.to_numeric(tx['text_len'], errors='coerce')
tx['color'] = pd.to_numeric(tx['color'], errors='coerce')
tx['lag_s'] = pd.to_numeric(tx['lag_s'], errors='coerce')
g = tx.groupby(['bvid', 'bucket60'])
bk = g.agg(n=('tnorm', 'size'), n_user=('user_hash', 'nunique'), n_txt=('tnorm', 'nunique'),
           mean_len=('text_len', 'mean')).reset_index()
rep = (tx.groupby(['bvid', 'bucket60', 'tnorm']).agg(k=('tnorm', 'size'), u=('user_hash', 'nunique'))
       .reset_index())
multi = rep[rep.u >= 2].groupby(['bvid', 'bucket60']).agg(multi_rows=('k', 'sum')).reset_index()
bk = bk.merge(multi, on=['bvid', 'bucket60'], how='left').fillna({'multi_rows': 0})
bk['multi_share'] = bk.multi_rows / bk.n
bk['repeat_share'] = 1 - bk.n_txt / bk.n
hk = tx.groupby(['bvid', 'bucket60', 'user_hash']).size().reset_index(name='k')
hk['p'] = hk.k / hk.groupby(['bvid', 'bucket60']).k.transform('sum')
hhi = hk.groupby(['bvid', 'bucket60']).apply(lambda d: float((d.p ** 2).sum()), include_groups=False).rename('hhi').reset_index()
bk = bk.merge(hhi, on=['bvid', 'bucket60'])
tx['is_colored'] = (tx.color != 16777215).astype(int)
tx['is_long'] = (tx.text_len >= 5).astype(int)
ex = tx.groupby(['bvid', 'bucket60']).agg(colored=('is_colored', 'mean'), long=('is_long', 'mean')).reset_index()
bk = bk.merge(ex, on=['bvid', 'bucket60'])

vid = bk.groupby('bvid').apply(lambda d: pd.Series({
    'vol': float(np.log1p(d.n.sum())),
    'multi': float(np.average(d.multi_share, weights=d.n)),
    'repeat': float(np.average(d.repeat_share, weights=d.n)),
    'hhi': float(np.average(d.hhi, weights=d.n)),
    'colored': float(np.average(d.colored, weights=d.n)),
    'long': float(np.average(d.long, weights=d.n)),
}), include_groups=False).reset_index()
vid = vid.merge(meta.drop_duplicates('bvid')[['bvid', 'tname', 'duration', 'got']], on='bvid', how='left')
vid['lg_dur'] = np.log1p(pd.to_numeric(vid.duration, errors='coerce'))
vid['lg_vol'] = vid.vol

# R2：同步 vs 集中度（控制量/时长/分区）
# PRE：偏相关 |r|<0.3 视为"分别携带信息"；同一指标与量的 |r|>0.7 视为量替身
d2 = vid.dropna(subset=['multi', 'hhi', 'lg_vol', 'lg_dur']).copy()
r2 = dict(n_videos=int(len(d2)), raw_spearman_multi_hhi=round(spearman(d2.multi, d2.hhi), 4))
c1 = resid_on(d2.multi.values, [d2.lg_vol.values, d2.lg_dur.values])
c2 = resid_on(d2.hhi.values, [d2.lg_vol.values, d2.lg_dur.values])
r2['partial_spearman_ctrl_vol_dur'] = round(spearman(c1, c2), 4)
# 再控分区（组内中心化）
d2['multi_c'] = d2.multi - d2.groupby('tname').multi.transform('mean')
d2['hhi_c'] = d2.hhi - d2.groupby('tname').hhi.transform('mean')
r2['partial_spearman_ctrl_vol_dur_partition'] = round(spearman(d2.multi_c, d2.hhi_c), 4)
r2['multi_vs_volume'] = round(spearman(d2.multi, d2.lg_vol), 4)
r2['hhi_vs_volume'] = round(spearman(d2.hhi, d2.lg_vol), 4)
r2['multi_boot_ci'] = boot_ci(d2.multi.values)
r2['status'] = ('分别携带信息' if abs(r2['partial_spearman_ctrl_vol_dur_partition']) < 0.3
                else '高度重叠（同一现象）')
r2['caveat'] = 'user_hash 覆盖 81.1%（0902/0907 全空）；用户级结论仅限 1012 后批次；相关≠因果'
json.dump(r2, open(f'{OUT}/r2_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R2]', json.dumps(r2, ensure_ascii=False), flush=True)

# R3：双时钟
lag = tx.dropna(subset=['lag_s']); lag = lag[lag.lag_s >= 0]
lag['h'] = (lag.lag_s / 3600.0).astype(int).clip(0, 720)


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


prog = bk[['bvid', 'bucket60', 'n']]
lagc = lag.groupby(['bvid', 'h']).size().reset_index(name='n')
rows = []
for bv in sorted(tx.bvid.unique()):
    p = prog[prog.bvid == bv].n.values
    l = lagc[lagc.bvid == bv].n.values
    if len(p) < 5 or len(l) < 5:
        continue
    rows.append(dict(bvid=bv, gini_prog=gini(p), gini_lag=gini(l),
                     peak_prog=float(np.argmax(p) / max(len(p) - 1, 1)),
                     peak_lag=float(np.argmax(l) / max(len(l) - 1, 1))))
r3 = pd.DataFrame(rows).merge(meta.drop_duplicates('bvid')[['bvid', 'tname', 'duration', 'got', 'batch_date']],
                              on='bvid', how='left')
r3['lg_dur'] = np.log1p(pd.to_numeric(r3.duration, errors='coerce'))
r3['lg_got'] = np.log1p(pd.to_numeric(r3.got, errors='coerce'))
d3 = r3.dropna(subset=['gini_prog', 'gini_lag', 'lg_dur', 'lg_got'])
res = dict(n_videos=int(len(d3)),
           gini_prog_median=round(float(d3.gini_prog.median()), 4),
           gini_lag_median=round(float(d3.gini_lag.median()), 4),
           spearman_gini=round(spearman(d3.gini_prog, d3.gini_lag), 4),
           spearman_peak_pos=round(spearman(d3.peak_prog, d3.peak_lag), 4))
e1 = resid_on(d3.gini_prog.values, [d3.lg_dur.values, d3.lg_got.values])
e2 = resid_on(d3.gini_lag.values, [d3.lg_dur.values, d3.lg_got.values])
res['partial_spearman_ctrl_dur_volume'] = round(spearman(e1, e2), 4)
per_batch = {}
for b, d in d3.groupby('batch_date'):
    if len(d) >= 10:
        per_batch[str(b)] = round(spearman(d.gini_prog, d.gini_lag), 4)
res['per_batch_spearman'] = per_batch
res['status'] = '两钟信息不同' if abs(res['partial_spearman_ctrl_dur_volume']) < 0.3 else '两钟高度重合'
res['caveat'] = 'send_time 受采集截断；lag≠观看时间；发布后延迟≠传播速度'
json.dump(res, open(f'{OUT}/r3_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R3]', json.dumps(res, ensure_ascii=False)[:300], flush=True)

# R4：表达形式（视频内配对：高峰桶 vs 其余桶）
# PRE：视频内配对差（去掉视频级混淆）的 95%CI 不含 0 且 |dz|≥0.2 才算"稳定差异"
bk['hi'] = bk.groupby('bvid').n.transform(lambda s: s >= s.quantile(0.8))
pairs = []
for bv, d in bk.groupby('bvid'):
    h, r = d[d.hi], d[~d.hi]
    if len(h) < 2 or len(r) < 2:
        continue
    pairs.append(dict(bvid=bv, d_long=float(h.long.mean() - r.long.mean()),
                      d_len=float(h.mean_len.mean() - r.mean_len.mean()),
                      d_colored=float(h.colored.mean() - r.colored.mean())))
P = pd.DataFrame(pairs)
r4 = dict(n_videos=int(len(P)))
for k, name in (('d_long', 'long_text_share'), ('d_len', 'mean_len'), ('d_colored', 'colored_share')):
    v = P[k].values
    r4[name] = dict(mean=round(float(np.mean(v)), 4), ci=boot_ci(v),
                    dz=round(float(np.mean(v) / np.std(v, ddof=1)), 3) if np.std(v) > 1e-9 else None)
r4['status'] = '待判（见各 dz 与 CI）'
r4['caveat'] = '颜色/字号/模式≠真实情绪；attr_weight 与计数相关 0.999（exp57）故未加权'
json.dump(r4, open(f'{OUT}/r4_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R4]', json.dumps(r4, ensure_ascii=False)[:400], flush=True)

# ================================================================ R6 D1 跨时间稳定性
# PRE：按测试批与体量五分位报告误差；若各批 MAE 极差 >2× 中位 → 稳定性不足
import sys
sys.path.insert(0, f'{REPO}/experiments/exp_ml_scan_20261027')
import run_scan as rs
from sklearn.linear_model import LinearRegression
d = rs.build_daily()
rows = []
for (bv, bt), gg in d.groupby(['bvid', 'batch']):
    pub = float(gg['pubdate'].iloc[0]); dur = float(gg['duration'].iloc[0])
    rel = (gg['send_time'].values - pub) / 3600.0
    if rel.max() < 31.0:
        continue
    n01 = ((rel >= 0) & (rel < 1)).sum(); n13 = ((rel >= 1) & (rel < 3)).sum()
    n36 = ((rel >= 3) & (rel < 6)).sum(); c0 = n01 + n13 + n36
    if c0 < 1:
        continue
    rows.append((bv, gg['first_seen'].iloc[0], n01, n13, n36, c0, ((rel > 6) & (rel <= 30)).sum(), dur))
u = pd.DataFrame(rows, columns=['bvid', 'fs', 'n01', 'n13', 'n36', 'c0', 'y', 'dur'])
tr, te = u[u.fs <= 20261020], u[u.fs >= 20261021]
assert len(set(tr.bvid) & set(te.bvid)) == 0
F6 = ['c0', 'n36', 'n13', 'n01', 'slope', 'dur']


def mats(df):
    return np.column_stack([np.log1p(df.c0), np.log1p(df.n36), np.log1p(df.n13),
                            np.log1p(df.n01), np.log1p(df.n36) - np.log1p(df.n13), np.log1p(df.dur)])


m6 = LinearRegression().fit(mats(tr), np.log1p(tr.y.values))
te = te.copy()
te['pred'] = np.clip(np.expm1(m6.predict(mats(te))), 0, None)
te['ae'] = (te.pred - te.y).abs()
te['re'] = te.ae / te.y.clip(lower=1)
r6 = dict(n_test=int(len(te)), r2_log1p=round(float(m6.score(mats(te), np.log1p(te.y))), 4),
          mae_orig=round(float(te.ae.mean()), 1), median_ae=round(float(te.ae.median()), 1),
          p90_ae=round(float(te.ae.quantile(.9)), 1))
r6['by_test_batch'] = {str(b): dict(n=int(len(g)), mae=round(float(g.ae.mean()), 1),
                                    re_median=round(float(g.re.median()), 3))
                       for b, g in te.groupby('fs') if len(g) >= 10}
r6['by_c0_quintile'] = {}
te['q'] = pd.qcut(te.c0, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
for q, g in te.groupby('q', observed=True):
    r6['by_c0_quintile'][str(q)] = dict(n=int(len(g)), c0_med=float(g.c0.median()),
                                        mae=round(float(g.ae.mean()), 1),
                                        re_median=round(float(g.re.median()), 3))
batches = [v['mae'] for v in r6['by_test_batch'].values()]
r6['batch_mae_spread_ratio'] = round(max(batches) / max(np.median(batches), 1e-9), 2) if batches else None
r6['status'] = ('跨批稳定' if r6['batch_mae_spread_ratio'] and r6['batch_mae_spread_ratio'] <= 2 else '跨批波动大')
r6['boundary'] = 'D1 为"互动量预测"（6h→(6h,30h]）；不是节奏阶段识别；仅日批热榜类视频'
json.dump(r6, open(f'{OUT}/r6_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R6]', json.dumps(r6, ensure_ascii=False)[:500], flush=True)

# ================================================================ R7 盲评材料
pairs_path = f'{OUT}/r7_blind_pairs.csv'
pool = sorted(tx.bvid.unique())
sampled = list(rng.choice(pool, size=min(30, len(pool)), replace=False))
with open(pairs_path, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['pair_id', 'bvid_a', 'bvid_b', 'url_a', 'url_b', 'similar_1to5', 'note'])
    pid = 1
    for i in range(0, len(sampled) - 1, 2):
        for pair in ((sampled[i], sampled[i + 1]),):
            w.writerow([f'P{pid:03d}', pair[0], pair[1],
                        f'https://www.bilibili.com/video/{pair[0]}',
                        f'https://www.bilibili.com/video/{pair[1]}', '', ''])
            pid += 1
print(f'[R7] 盲评材料 → {pairs_path}（{pid - 1} 对）', flush=True)
print('EXP59-DONE')
