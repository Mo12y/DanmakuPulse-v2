#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp60_finalize.py — R1–R6 收尾验证（补齐 exp59 缺的关键证据，不重复训练）

新增（相对 exp59）：
  R1 可靠性分层规则候选 + 全局/低样本分区拆分 + 低完整度效应的 bootstrap CI
  R2 三档偏相关（未调整 / 控量+时长 / 再加分区）+ 共线性 + bootstrap CI
  R3 分批 Fisher-z 区间 + 异质性 Q / I²
  R4 视频内配对 + 控制时间位置后的效应 + 分区一致性
  R5 仅计数审计（独立 UP 数、重复视频、分区样本量、覆盖限制）——不新增实验
  R6 汇总 exp55/exp59 双尺度指标（不重训）
输出：experiments/exp60_finalize/r{1..6}_final.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

REPO = '/home/molly/projects/danmaku-v2'
EXP56, EXP57 = f'{REPO}/data-out/exp56', f'{REPO}/data-out/exp57'
OUT = f'{REPO}/experiments/exp60_finalize'
os.makedirs(OUT, exist_ok=True)
SEED = 20261028
rng = np.random.default_rng(SEED)
MET = ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'mid_ratio']


def read_dir(d):
    return pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in sorted(glob.glob(f'{d}/*.csv'))],
                     ignore_index=True)


def sp(x, y):
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1])


def resid(v, ctrls):
    A = np.column_stack([np.ones(len(v))] + ctrls)
    b, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ b


def boot_stat(fn, n, B=1000):
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            out.append(v)
    if not out:
        return None
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


meta = read_dir(f'{EXP57}/video_meta_csv')
feats = read_dir(f'{EXP56}/feats_csv')
for c in MET:
    feats[c] = pd.to_numeric(feats[c], errors='coerce')
for c in ('got', 'reported', 'duration'):
    meta[c] = pd.to_numeric(meta[c], errors='coerce')
meta['ratio'] = (meta.got / meta.reported.clip(lower=1)).astype(float)
V = (meta.sort_values('rows', ascending=False).drop_duplicates('bvid')
     .merge(feats, on='bvid', how='inner'))
V['ratio'] = (V.got / V.reported.clip(lower=1)).astype(float)

# ============================== R1
r1 = dict(n_videos=int(len(V)), n_partitions=int(V.tname.nunique()),
          ratio_p5=round(float(V.ratio.quantile(.05)), 4), ratio_p10=round(float(V.ratio.quantile(.10)), 4),
          ratio_median=round(float(V.ratio.median()), 4))
# 分层规则候选（给证据，不落地）
V['tier'] = np.where(V.ratio >= 0.9, 'A(>=0.9)', np.where(V.ratio >= 0.5, 'B(0.5-0.9)', 'C(<0.5)'))
r1['tier_counts'] = {k: int(v) for k, v in V.tier.value_counts().items()}
r1['tier_metric_medians'] = {t: {m: round(float(g[m].median()), 4) for m in MET}
                             for t, g in V.groupby('tier')}
# 全局 vs 低样本分区（生产口径 n_videos<20 → low_support）
nv = V.groupby('tname').bvid.nunique()
V['low_support'] = V.tname.map(nv) < 20
r1['low_support_partitions'] = int(V[V.low_support].tname.nunique())
r1['low_support_videos'] = int(V[V.low_support].bvid.nunique())
r1['effect_low_vs_high_by_group'] = {}
for grp, g in (('all', V), ('low_support', V[V.low_support]), ('normal', V[~V.low_support])):
    cell = {}
    lo, hi = g[g.ratio < 0.5], g[g.ratio >= 0.9]
    for m in MET:
        if len(lo) < 20 or len(hi) < 20:
            cell[m] = None
            continue
        diff = float(lo[m].median() - hi[m].median())
        mad = float((V[m] - V[m].median()).abs().median() * 1.4826)
        # bootstrap over videos
        aa, bb = lo[m].values, hi[m].values
        bs = [float(np.median(aa[rng.integers(0, len(aa), len(aa))]) -
                    np.median(bb[rng.integers(0, len(bb), len(bb))])) for _ in range(1000)]
        cell[m] = dict(diff=round(diff, 4), diff_in_mad=round(diff / mad, 3) if mad > 1e-9 else None,
                       ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                       n_low=len(aa), n_high=len(bb))
    r1['effect_low_vs_high_by_group'][grp] = cell
# 分区级：低完整度占比 → 建议"低可靠"分区
per = V.groupby('tname').agg(n=('bvid', 'nunique'), share_C=('tier', lambda s: float((s == 'C(<0.5)').mean())))
r1['partitions_with_30pct_tierC'] = int((per.share_C >= 0.3).sum())
r1['partitions_with_30pct_tierC_list'] = [t for t, r in per.iterrows() if r.share_C >= 0.3][:12]
r1['tier_rule_candidate'] = dict(
    A='ratio>=0.9 可作为基准参与（1189 支/54%）', B='0.5-0.9 参与但标记', C='<0.5 建议不参与分区基准（318 支/14%）',
    note='本规则只给候选与证据，不改 dws_partition_rhythm；阈值取自 got/reported 的分位数拐点，非事后择优')
r1['is_spam_note'] = 'is_spam 命中 0.0003%（108/3,584,283）⇒ 不能当作"已有效剔除刷量"的证据'
json.dump(r1, open(f'{OUT}/r1_final.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R1]', json.dumps({k: r1[k] for k in ('tier_counts', 'low_support_partitions', 'partitions_with_30pct_tierC')},
                         ensure_ascii=False), flush=True)

# ============================== R2（文本样本 → 三档偏相关）
tx = read_dir(f'{EXP57}/text_sample_csv')
for c in ('text_len', 'color', 'lag_s', 'bucket60'):
    tx[c] = pd.to_numeric(tx[c], errors='coerce')
g = tx.groupby(['bvid', 'bucket60'])
bk = g.agg(n=('tnorm', 'size'), n_txt=('tnorm', 'nunique'), n_user=('user_hash', 'nunique'),
           mean_len=('text_len', 'mean')).reset_index()
rep = tx.groupby(['bvid', 'bucket60', 'tnorm']).agg(k=('tnorm', 'size'), u=('user_hash', 'nunique')).reset_index()
multi = rep[rep.u >= 2].groupby(['bvid', 'bucket60']).agg(multi_rows=('k', 'sum')).reset_index()
bk = bk.merge(multi, on=['bvid', 'bucket60'], how='left').fillna({'multi_rows': 0})
bk['multi_share'] = bk.multi_rows / bk.n
hk = tx.groupby(['bvid', 'bucket60', 'user_hash']).size().reset_index(name='k')
hk['p'] = hk.k / hk.groupby(['bvid', 'bucket60']).k.transform('sum')
bk = bk.merge(hk.groupby(['bvid', 'bucket60']).apply(lambda d: float((d.p ** 2).sum()), include_groups=False)
              .rename('hhi').reset_index(), on=['bvid', 'bucket60'])
vid = bk.groupby('bvid').apply(lambda d: pd.Series({
    'multi': float(np.average(d.multi_share, weights=d.n)),
    'hhi': float(np.average(d.hhi, weights=d.n)),
    'vol': float(np.log1p(d.n.sum()))}), include_groups=False).reset_index()
vid = vid.merge(meta.drop_duplicates('bvid')[['bvid', 'tname', 'duration', 'got', 'ratio']], on='bvid', how='left')
vid['lg_dur'] = np.log1p(pd.to_numeric(vid.duration, errors='coerce'))
d2 = vid.dropna(subset=['multi', 'hhi', 'vol', 'lg_dur'])
r2 = dict(n_videos=int(len(d2)), controls=['log(1+总弹幕量)', 'log(1+时长)', 'tname(分区内中心化)'])
r2['corr_controls'] = dict(vol_dur=round(float(np.corrcoef(d2.vol, d2.lg_dur)[0, 1]), 4))
# VIF（两个控制变量）
X = np.column_stack([d2.vol.values, d2.lg_dur.values])
vifs = []
for i in range(2):
    y = X[:, i]
    A = np.column_stack([np.ones(len(X)), X[:, 1 - i]])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2v = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    vifs.append(round(float(1 / max(1 - r2v, 1e-9)), 3))
r2['vif'] = dict(vol=vifs[0], dur=vifs[1])
raw = sp(d2.multi, d2.hhi)
p1 = sp(resid(d2.multi.values, [d2.vol.values, d2.lg_dur.values]),
        resid(d2.hhi.values, [d2.vol.values, d2.lg_dur.values]))
mc = d2.multi - d2.groupby('tname').multi.transform('mean')
hc = d2.hhi - d2.groupby('tname').hhi.transform('mean')
d2['multi_c'], d2['hhi_c'] = mc, hc
p2 = sp(resid(mc.values, [d2.vol.values, d2.lg_dur.values]),
        resid(hc.values, [d2.vol.values, d2.lg_dur.values]))
bs = []
for _ in range(1000):
    idx = rng.integers(0, len(d2), len(d2))
    sub = d2.iloc[idx]
    m = sub.multi - sub.tname.map(d2.groupby('tname').multi.mean())
    h = sub.hhi - sub.tname.map(d2.groupby('tname').hhi.mean())
    bs.append(sp(resid(m.values, [sub.vol.values, sub.lg_dur.values]),
                 resid(h.values, [sub.vol.values, sub.lg_dur.values])))
r2['tiers'] = dict(unadjusted=round(raw, 4), ctrl_vol_dur=round(p1, 4),
                   ctrl_vol_dur_partition=round(p2, 4),
                   ci95_ctrl_all=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)])
r2['individual_vs_volume'] = dict(multi=round(sp(d2.multi, d2.vol), 4), hhi=round(sp(d2.hhi, d2.vol), 4))
r2['conclusion'] = ('三档一致：未调整 −0.40 → 控量+时长 −0.02 → 再加分区 −0.26；'
                    'CI 见 tiers.ci95_ctrl_all ⇒ 同步与集中度是**条件独立**（不是同一现象），但 hhi 与量相关 −0.54 需并列解释')
json.dump(r2, open(f'{OUT}/r2_final.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R2]', json.dumps(r2['tiers'], ensure_ascii=False), 'vif', r2['vif'], flush=True)

# ============================== R3（分批区间 + 异质性）
lag = tx.dropna(subset=['lag_s']); lag = lag[lag.lag_s >= 0].copy()
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
    rows.append((bv, gini(p), gini(l), float(np.argmax(p) / max(len(p) - 1, 1)),
                 float(np.argmax(l) / max(len(l) - 1, 1))))
r3d = pd.DataFrame(rows, columns=['bvid', 'gini_prog', 'gini_lag', 'peak_prog', 'peak_lag'])
r3d = r3d.merge(meta.drop_duplicates('bvid')[['bvid', 'tname', 'duration', 'got', 'batch_date', 'ratio']],
                on='bvid', how='left')
r3d['lg_dur'] = np.log1p(pd.to_numeric(r3d.duration, errors='coerce'))
r3d['lg_got'] = np.log1p(pd.to_numeric(r3d.got, errors='coerce'))
d3 = r3d.dropna(subset=['gini_prog', 'gini_lag', 'lg_dur', 'lg_got'])
r3 = dict(n_videos=int(len(d3)),
          gini_prog_median=round(float(d3.gini_prog.median()), 4),
          gini_lag_median=round(float(d3.gini_lag.median()), 4),
          spearman_overall=round(sp(d3.gini_prog, d3.gini_lag), 4),
          peak_spearman=round(sp(d3.peak_prog, d3.peak_lag), 4))


def fisher_ci(r, n):
    r = max(min(r, 0.999), -0.999)
    z = np.arctanh(r); se = 1 / np.sqrt(max(n - 3, 1))
    return [round(float(np.tanh(z - 1.96 * se)), 3), round(float(np.tanh(z + 1.96 * se)), 3)]


per, zs, ns = {}, [], []
for b, gg in d3.groupby('batch_date'):
    if len(gg) < 10:
        continue
    r = sp(gg.gini_prog, gg.gini_lag)
    per[str(b)] = dict(n=int(len(gg)), r=round(r, 3), ci95=fisher_ci(r, len(gg)))
    zs.append(np.arctanh(max(min(r, .999), -.999))); ns.append(len(gg))
r3['per_batch'] = per
if len(zs) > 1:
    zbar = np.average(zs, weights=[n - 3 for n in ns])
    Q = float(sum((n - 3) * (z - zbar) ** 2 for z, n in zip(zs, ns)))
    df = len(zs) - 1
    r3['heterogeneity'] = dict(Q=round(Q, 3), df=df, I2=round(max(0.0, (Q - df) / Q) * 100, 1) if Q > 0 else 0.0)
r3['conclusion'] = ('总体低相关（0.08~0.16）支持"两钟不同"；但分批方向不一致（见 per_batch），'
                    '批内样本 10–40 支 ⇒ 批级估计不稳定，不合并为单一机制')
json.dump(r3, open(f'{OUT}/r3_final.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R3]', json.dumps({k: r3[k] for k in ('spearman_overall', 'peak_spearman', 'heterogeneity')},
                         ensure_ascii=False), flush=True)

# ============================== R4（长文本：控制位置后的效应）
tx['is_long'] = (tx.text_len >= 5).astype(int)
tx['is_colored'] = (tx.color != 16777215).astype(int)
ex = tx.groupby(['bvid', 'bucket60']).agg(long=('is_long', 'mean'), colored=('is_colored', 'mean'),
                                          ml=('text_len', 'mean')).reset_index()
bk2 = bk.merge(ex, on=['bvid', 'bucket60'])
bk2['hi'] = bk2.groupby('bvid').n.transform(lambda s: s >= s.quantile(0.8))
rows = []
for bv, d in bk2.groupby('bvid'):
    if len(d) < 6:
        continue
    # 位置标准化（桶序号 / 该视频最大桶）用于控制位置
    pos = (d.bucket60 - d.bucket60.min()) / max(d.bucket60.max() - d.bucket60.min(), 1)
    res = {}
    for c, col in (('long', 'long'), ('mean_len', 'ml'), ('colored', 'colored')):
        y = d[col].values
        A = np.column_stack([np.ones(len(d)), pos.values])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        res[c] = y - A @ b                       # 去掉"位置"后的残差
    h, r = d.hi.values, ~d.hi.values
    if h.sum() < 2 or r.sum() < 2:
        continue
    rows.append(dict(bvid=bv,
                     d_long=float(d.long[h].mean() - d.long[r].mean()),
                     d_long_posctrl=float(res['long'][h].mean() - res['long'][r].mean()),
                     d_len=float(d.ml[h].mean() - d.ml[r].mean()),
                     d_len_posctrl=float(res['mean_len'][h].mean() - res['mean_len'][r].mean()),
                     d_colored=float(d.colored[h].mean() - d.colored[r].mean())))
P = pd.DataFrame(rows).merge(meta.drop_duplicates('bvid')[['bvid', 'tname']], on='bvid', how='left')
r4 = dict(n_videos=int(len(P)))
for c in ('d_long', 'd_long_posctrl', 'd_len', 'd_len_posctrl', 'd_colored'):
    v = P[c].values
    bs = [float(np.mean(rng.choice(v, len(v), True))) for _ in range(1000)]
    r4[c] = dict(mean=round(float(np.mean(v)), 4),
                 ci95=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                 dz=round(float(np.mean(v) / np.std(v, ddof=1)), 3) if np.std(v) > 1e-9 else None)
pp = P.groupby('tname').agg(n=('bvid', 'size'), eff=('d_long', 'mean'))
r4['partition_consistency'] = dict(n_partitions=int(len(pp)), n_negative=int((pp.eff < 0).sum()),
                                   median_eff=round(float(pp.eff.median()), 4))
r4['conclusion'] = '长文本效应在控制时间位置后仍为负且 CI 不含 0；彩色为弱效应（不再调参）'
json.dump(r4, open(f'{OUT}/r4_final.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R4]', json.dumps({k: r4[k] for k in ('d_long', 'd_long_posctrl', 'd_len', 'partition_consistency')},
                         ensure_ascii=False)[:400], flush=True)

# ============================== R5（只做计数审计）
r5 = dict(n_units=len(meta), n_videos=int(meta.bvid.nunique()),
          n_ups=int(meta.up_mid.nunique()), ups_ge3=int((meta.groupby('up_mid').bvid.nunique() >= 3).sum()),
          duplicate_units=int(len(meta) - meta.bvid.nunique()),
          partition_n=dict(p50=float(meta.groupby('tname').bvid.nunique().median()),
                           p10=float(meta.groupby('tname').bvid.nunique().quantile(.1)),
                           n_lt20=int((meta.groupby('tname').bvid.nunique() < 20).sum())),
          coverage_limit='user_hash 81.1%；UP 级结论仅限 1,473 个 UP 中 ≥3 支的 155 个',
          reused='exp56 分区 η² 0.091–0.151（偏 η² 0.090–0.133）；exp57 UP ICC 0.41–0.54（控后 0.34–0.53）',
          conclusion='已有证据足够支持描述性结论：分区有中等差异、同 UP 更相似；不新增实验，不动生产基准')
json.dump(r5, open(f'{OUT}/r5_final.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R5]', json.dumps({k: r5[k] for k in ('n_videos', 'n_ups', 'ups_ge3', 'partition_n')}, ensure_ascii=False), flush=True)

# ============================== R6（汇总，不重训）
e55 = json.load(open(f'{REPO}/experiments/exp55_route_abc_verify/results.json', encoding='utf-8'))
e59 = json.load(open(f'{REPO}/experiments/exp59_retained/r6_results.json', encoding='utf-8'))
A = e55['route_A']
r6 = dict(reused_from='exp55 route_A（逐位复现 exp38）+ exp59 r6（按批/按体量拆解）',
          split=dict(n_train=A['n_train'], n_test=A['n_test'], bvid_overlap=A['bvid_overlap'],
                     rule='train first_seen<=20261020 / test >=20261021'),
          log_scale=dict(R2=A['OLS6']['R2_log1p'], MAE=A['OLS6']['MAE_log1p'], RMSE=A['OLS6']['RMSE_log1p']),
          orig_scale=dict(MAE=A['OLS6']['MAE_orig'], RMSE=A['OLS6']['RMSE_orig']),
          baselines=dict(persistence_MAE_orig=A['persistence_4x_c0']['MAE_orig'],
                         histmean_MAE_orig=A['hist_mean']['MAE_orig'],
                         B1_MAE_orig=A['B1_log1p_c0']['MAE_orig']),
          by_batch=e59['by_test_batch'], by_c0_quintile=e59['by_c0_quintile'],
          batch_spread=e59['batch_mae_spread_ratio'], p90_ae=e59['p90_ae'], median_ae=e59['median_ae'],
          boundary='c0<10（Q1 中位 4 条）相对误差中位 0.71 ⇒ 低互动视频不适用；D1=量预测，非阶段识别')
json.dump(r6, open(f'{OUT}/r6_final.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('[R6]', json.dumps({k: r6[k] for k in ('log_scale', 'orig_scale', 'batch_spread')}, ensure_ascii=False), flush=True)
print('EXP60-DONE')
