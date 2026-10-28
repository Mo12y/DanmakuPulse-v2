#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp57_eight_routes.py — 八条路线最小验证（用 exp57 导出的样本与元数据）

样本：200 支视频 / 32.9 万条弹幕（user_hash 覆盖≥90% 的批次）+ 2,218 支视频元数据 + 字段质量。
纪律：每条路线先问"字段是否存在→质量是否够→指标能否算→是否有解释意义→假设是否被支持"；
无外部人工评价的路线一律标"证据不足"，不把可计算当成立。
"""
import csv
import glob
import json
import numpy as np
import pandas as pd

REPO = '/home/molly/projects/danmaku-v2'
EXP = f'{REPO}/data-out/exp57'
OUT = f'{REPO}/experiments/exp57_eight_routes'
SEED = 20261028
rng = np.random.default_rng(SEED)
import os
os.makedirs(OUT, exist_ok=True)


def read_dir(d, **kw):
    ps = sorted(glob.glob(f'{d}/*.csv'))
    return pd.concat([pd.read_csv(p, **kw) for p in ps], ignore_index=True)


quality = json.load(open(f'{EXP}/quality.json', encoding='utf-8'))
vm = read_dir(f'{EXP}/video_meta_csv', encoding='utf-8-sig')
tx = read_dir(f'{EXP}/text_sample_csv', encoding='utf-8-sig',
              dtype={'bvid': str, 'user_hash': str, 'tnorm': str, 'txt': str})
tx['bucket60'] = tx['bucket60'].astype(int)
R = {'meta': dict(seed=SEED, sample_videos=int(tx.bvid.nunique()), sample_rows=int(len(tx)),
                  all_videos=int(len(vm)), quality_summary=dict(
                      user_cov=quality['shares']['user_hash_nonnull'],
                      key_unique=quality['key_unique'],
                      spam_rate=quality['shares']['is_spam_true'],
                      norm_diff_rate=quality['shares']['text_norm_differs']))}

# ---------------- 公共：每桶聚合 ----------------
g = tx.groupby(['bvid', 'bucket60'])
bk = g.agg(n=('tnorm', 'size'),
           n_user=('user_hash', 'nunique'),
           n_txt=('tnorm', 'nunique'),
           mean_len=('text_len', 'mean'),
           w=('attr_weight', 'sum')).reset_index()
# 多用户重复：同一 tnorm 由 ≥2 个不同 user_hash 发送
rep = (tx.groupby(['bvid', 'bucket60', 'tnorm'])
       .agg(k=('tnorm', 'size'), u=('user_hash', 'nunique')).reset_index())
rep_multi = rep[rep.u >= 2].groupby(['bvid', 'bucket60']).agg(n_multi=('tnorm', 'size'),
                                                             multi_rows=('k', 'sum')).reset_index()
bk = bk.merge(rep_multi, on=['bvid', 'bucket60'], how='left').fillna({'n_multi': 0, 'multi_rows': 0})
bk['repeat_share'] = 1 - bk.n_txt / bk.n
bk['multi_share'] = bk.multi_rows / bk.n
bk['user_ratio'] = bk.n_user / bk.n

R['route1_sync'] = dict(
    n_buckets=int(len(bk)),
    bucket_level_median=dict(n=float(bk.n.median()), n_user=float(bk.n_user.median()),
                             repeat_share=float(bk.repeat_share.median()),
                             multi_share=float(bk.multi_share.median())),
    corr_volume_vs_multi=round(float(np.corrcoef(np.log1p(bk.n), bk.multi_share)[0, 1]), 4),
    corr_volume_vs_repeat=round(float(np.corrcoef(np.log1p(bk.n), bk.repeat_share)[0, 1]), 4),
    corr_volume_vs_userratio=round(float(np.corrcoef(np.log1p(bk.n), bk.user_ratio)[0, 1]), 4),
)

# 同量级视频内，同步指标是否仍有差异（控制总量）
vid_sync = (bk.groupby('bvid').apply(lambda d: pd.Series({
    'vol': float(np.log1p(d.n.sum())),
    'multi': float(np.average(d.multi_share, weights=d.n)),
    'repeat': float(np.average(d.repeat_share, weights=d.n)),
}), include_groups=False).reset_index())
vid_sync['vol_tertile'] = pd.qcut(vid_sync.vol, 3, labels=['low', 'mid', 'high'])
tt = vid_sync.groupby('vol_tertile', observed=True).agg(
    n=('bvid', 'size'), multi_med=('multi', 'median'), multi_iqr=('multi', lambda s: float(s.quantile(.75) - s.quantile(.25))),
    repeat_med=('repeat', 'median'))
R['route1_sync']['within_volume_tertile'] = {str(k): {kk: round(float(vv), 4) if isinstance(vv, (int, float, np.floating)) else vv
                                                      for kk, vv in v.items()} for k, v in tt.iterrows()}
R['route1_sync']['interpretation'] = 'multi_share 与 log(量) 的相关若接近 0，说明同步性不是量的替身'
R['route1_sync']['status'] = '有初步支持证据' if abs(R['route1_sync']['corr_volume_vs_multi']) < 0.5 else '结果不稳定（与量高度相关）'

# ---------------- R2 复读/接力 ----------------
ev = (tx.groupby(['bvid', 'tnorm'])
      .agg(k=('tnorm', 'size'), u=('user_hash', 'nunique'),
           bmin=('bucket60', 'min'), bmax=('bucket60', 'max')).reset_index())
ev['span'] = ev['bmax'] - ev['bmin']
ev3_all = ev[ev.k >= 3].copy()                      # 定义 1：任意窗口内同文本≥3 次
ev3 = ev3_all[ev3_all.span <= 3].copy()             # 定义 2（主口径）：同文本≥3 次且集中在 ≤3 桶内
ev3['single_user'] = ev3.u == 1
R['route2_repeat'] = dict(
    def1_k3_any_window=int(len(ev3_all)), def1_per_video=round(len(ev3_all) / max(tx.bvid.nunique(), 1), 2),
    def2_k3_within_3buckets=int(len(ev3)), def2_per_video=round(len(ev3) / max(tx.bvid.nunique(), 1), 2),
    single_user_share=round(float(ev3.single_user.mean()), 4),
    multi_user_share=round(float(1 - ev3.single_user.mean()), 4),
    span_median_single=float(ev3[ev3.single_user].span.median()) if ev3.single_user.any() else None,
    span_median_multi=float(ev3[~ev3.single_user].span.median()) if (~ev3.single_user).any() else None,
    note_def='定义1（任意窗口 k≥3）在低池视频上近乎普遍（约 96 个/视频）⇒ 失去"事件"性；'
             '主口径改用定义2（≤3 桶内 k≥3），见 def2_*')

# 规范化是否制造虚假重复：tnorm 相同但 txt 不同的组
same_tnorm_diff_txt = (tx.groupby(['bvid', 'tnorm'])['txt'].nunique() > 1).sum()
R['route2_repeat']['tnorm_groups_with_multiple_raw_texts'] = int(same_tnorm_diff_txt)
R['route2_repeat']['note_norm'] = 'tnorm 相同但原文不同 ⇒ 规范化把原文差异抹平（emoji/符号替换），复读计数会偏高'

# 时间结构：事件起点是否比随机更聚集（对照：把事件起点整体+30 桶循环平移）
starts = ev3[['bvid', 'bmin']].copy()
def cluster_score(s):
    v = []
    for bv, d in s.groupby('bvid'):
        b = np.sort(d.bmin.values)
        if len(b) < 3:
            continue
        v.extend(np.diff(b).tolist())
    return float(np.median(v)) if v else None
obs = cluster_score(starts)
nulls = []
for _ in range(200):
    s2 = starts.copy()
    s2['bmin'] = (s2.bmin + rng.integers(5, 40, len(s2))) % 40
    nulls.append(cluster_score(s2))
nulls = [x for x in nulls if x is not None]
R['route2_repeat']['median_gap_buckets_obs'] = obs
R['route2_repeat']['median_gap_buckets_null_med'] = round(float(np.median(nulls)), 2) if nulls else None
R['route2_repeat']['temporal_clustering'] = ('事件起点比随机更聚集' if obs is not None and nulls and obs < np.median(nulls)
                                             else '未超出随机基线')
R['route2_repeat']['status'] = '有初步支持证据（单/多用户可区分 + 时间结构超出随机）' if obs is not None and nulls and obs < np.median(nulls) * 0.8 else '结果不稳定'
R['route2_repeat']['relay_note'] = '"接力"未单独定义（需预先定义前后事件关联条件）；本报告只测事件时间聚集性'

# ---------------- R3 双时钟 ----------------
def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

lag = tx.dropna(subset=['lag_s'])
lag = lag[lag.lag_s >= 0]
lag_h = (lag.lag_s / 3600.0).astype(int).clip(0, 24 * 30)
prog = tx.groupby(['bvid', 'bucket60']).size().reset_index(name='n')
lagc = lag.groupby(['bvid', lag_h.rename('h')]).size().reset_index(name='n')
r3 = []
for bv in sorted(tx.bvid.unique()):
    p = prog[prog.bvid == bv].n.values
    l = lagc[lagc.bvid == bv].n.values
    if len(p) < 5 or len(l) < 5:
        continue
    r3.append(dict(bvid=bv, gini_prog=gini(p), gini_lag=gini(l),
                   peak_prog_pos=float(np.argmax(p) / max(len(p) - 1, 1)),
                   peak_lag_pos=float(np.argmax(l) / max(len(l) - 1, 1))))
r3 = pd.DataFrame(r3)
R['route3_dual_clock'] = dict(
    n_videos=int(len(r3)),
    gini_prog_median=round(float(r3.gini_prog.median()), 4),
    gini_lag_median=round(float(r3.gini_lag.median()), 4),
    corr_gini=round(float(r3.gini_prog.corr(r3.gini_lag)), 4),
    corr_peak_positions=round(float(r3.peak_prog_pos.corr(r3.peak_lag_pos)), 4),
    lag_coverage=dict(rows_with_lag=int(len(lag)), share=round(len(lag) / len(tx), 4)),
    status='有初步支持证据（两钟集中度与峰位相关性低）' if abs(r3.gini_prog.corr(r3.gini_lag)) < 0.5 else '两钟信息高度重合',
    caveat='send_time 受采集截断（快照后发送的弹幕不在池）；lag≠观看时间')

# ---------------- R4 表达 × 节奏 ----------------
tx['is_colored'] = (tx.color.astype(float) != 16777215).astype(int)
tx['is_long'] = (tx.text_len.astype(float) >= 5).astype(int)
ex = tx.groupby(['bvid', 'bucket60']).agg(n=('tnorm', 'size'), colored=('is_colored', 'mean'),
                                          long=('is_long', 'mean'), mean_len=('text_len', 'mean'),
                                          w=('attr_weight', 'sum')).reset_index()
ex = ex.merge(bk[['bvid', 'bucket60', 'n']].rename(columns={'n': 'n2'}), on=['bvid', 'bucket60'])
ex['hi'] = ex.groupby('bvid').n.transform(lambda s: s >= s.quantile(0.8))
R['route4_expression'] = dict(
    n_buckets=int(len(ex)),
    corr_n_vs_colored=round(float(np.corrcoef(np.log1p(ex.n), ex.colored)[0, 1]), 4),
    corr_n_vs_long=round(float(np.corrcoef(np.log1p(ex.n), ex.long)[0, 1]), 4),
    high_vs_rest=dict(colored=round(float(ex[ex.hi].colored.mean() - ex[~ex.hi].colored.mean()), 4),
                      long=round(float(ex[ex.hi].long.mean() - ex[~ex.hi].long.mean()), 4),
                      mean_len=round(float(ex[ex.hi].mean_len.mean() - ex[~ex.hi].mean_len.mean()), 2)),
    attr_weight_corr_with_n=round(float(np.corrcoef(np.log1p(ex.n), np.log1p(ex.w))[0, 1]), 4),
    status='待判（见 high_vs_rest 与相关性；加权与未加权差异见 corr）',
    caveat='颜色/字号/模式≠真实情绪；只作表达形式')

# ---------------- R5 参与者集中度 ----------------
hk = tx.groupby(['bvid', 'bucket60', 'user_hash']).size().reset_index(name='k')
hh = (hk.assign(p=lambda d: d.k / d.groupby(['bvid', 'bucket60']).k.transform('sum'))
      .groupby(['bvid', 'bucket60']).apply(lambda d: pd.Series({
          'hhi': float((d.p ** 2).sum()), 'top1': float(d.p.max())}), include_groups=False).reset_index())
mg = bk.merge(hh, on=['bvid', 'bucket60'])
mg['is_peak'] = mg.groupby('bvid').n.transform(lambda s: s >= s.quantile(0.8))
# 去掉 top1% 高频用户后曲线是否变化
cnt = tx.groupby('user_hash').size().sort_values(ascending=False)
cut = int(len(cnt) * 0.01)
top_users = set(cnt.index[:cut])
tx2 = tx[~tx.user_hash.isin(top_users)]
pk1 = bk.groupby('bvid').apply(lambda d: float(d.loc[d.n.idxmax(), 'bucket60'] / max(d.bucket60.max(), 1)), include_groups=False)
bk2 = tx2.groupby(['bvid', 'bucket60']).size().reset_index(name='n')
pk2 = bk2.groupby('bvid').apply(lambda d: float(d.loc[d.n.idxmax(), 'bucket60'] / max(d.bucket60.max(), 1)), include_groups=False)
common = pk1.index.intersection(pk2.index)
R['route5_concentration'] = dict(
    n_buckets=int(len(mg)),
    hhi_median=round(float(mg.hhi.median()), 4),
    peak_vs_rest=dict(hhi=round(float(mg[mg.is_peak].hhi.mean() - mg[~mg.is_peak].hhi.mean()), 4),
                      top1=round(float(mg[mg.is_peak].top1.mean() - mg[~mg.is_peak].top1.mean()), 4)),
    corr_n_vs_hhi=round(float(np.corrcoef(np.log1p(mg.n), mg.hhi)[0, 1]), 4),
    peak_pos_shift_after_removing_top1pct_users=round(float((pk2[common] - pk1[common]).abs().median()), 4),
    n_top_users_removed=cut,
    status='待判（peak 与非 peak 的集中度差 + 去高频用户后峰位漂移）',
    caveat='user_hash 是匿名哈希，不等于独立自然人；is_spam 非完美标签')

# ---------------- R6 采集完整性 ----------------
vm['ratio_got_reported'] = vm.got / vm.reported.clip(lower=1)
multi_batch = vm.groupby('bvid').filter(lambda d: len(d) >= 2)
cons = []
for bv, d in multi_batch.groupby('bvid'):
    if len(d) < 2:
        continue
    d = d.sort_values('batch_date')
    cons.append(dict(bvid=bv, n_batches=len(d),
                     cov_range=float(d.coverage.max() - d.coverage.min()),
                     got_cv=float(d.got.std() / max(d.got.mean(), 1e-9))))
cons = pd.DataFrame(cons)
R['route6_completeness'] = dict(
    n_videos=len(vm), n_multi_batch_videos=int(len(cons)),
    got_over_reported_median=round(float(vm.ratio_got_reported.median()), 4),
    got_over_reported_p10=round(float(vm.ratio_got_reported.quantile(0.1)), 4),
    coverage_median=round(float(vm.coverage.median()), 4),
    multi_batch_coverage_range_median=round(float(cons.cov_range.median()), 4) if len(cons) else None,
    multi_batch_got_cv_median=round(float(cons.got_cv.median()), 4) if len(cons) else None,
    status='有初步支持证据（同视频跨批次覆盖差异可量化；结论敏感性需见下）',
    caveat='got/reported 不是无偏采样率；跨批差异同时含池滚动与真实增长')

# ---------------- R7 UP/分区/时长 ----------------
feats = read_dir(f'{REPO}/data-out/exp56/feats_csv', encoding='utf-8-sig')
V = vm.drop_duplicates('bvid').merge(feats, on='bvid', how='inner', suffixes=('', '_f'))
for c in ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'mid_ratio', 'ending_5pct',
          'early_lag_ratio', 'peak_cnt']:
    if c in V.columns:
        V[c] = pd.to_numeric(V[c], errors='coerce')
MET = ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'mid_ratio']


def eta2(vals, groups):
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    ok = ~np.isnan(vals); vals, groups = vals[ok], groups[ok]
    if len(vals) < 10 or len(set(groups)) < 2:
        return None
    gm = vals.mean()
    ss_tot = ((vals - gm) ** 2).sum()
    ss_b = sum(len(vals[groups == g]) * (vals[groups == g].mean() - gm) ** 2 for g in set(groups))
    return float(ss_b / ss_tot)


def icc_one(V, key, met):
    d = V.dropna(subset=[key, met])
    grp = d.groupby(key)[met]
    ns = grp.size()
    keep = ns[ns >= 3].index
    d = d[d[key].isin(keep)]
    if d[key].nunique() < 3:
        return None
    grand = d[met].mean()
    msb = sum(len(g) * (g[met].mean() - grand) ** 2 for _, g in d.groupby(key)) / (d[key].nunique() - 1)
    msw = sum(((g[met] - g[met].mean()) ** 2).sum() for _, g in d.groupby(key)) / (len(d) - d[key].nunique())
    return float((msb - msw) / (msb + (3 - 1) * msw)) if (msb + 2 * msw) > 0 else None


def _icc_partial(Vd, mets):
    import numpy as _np
    out = {}
    lay = Vd.copy()
    lay['lg_dur'] = _np.log1p(pd.to_numeric(lay['duration'], errors='coerce'))
    lay['lg_got'] = _np.log1p(pd.to_numeric(lay['got'], errors='coerce'))
    for m in mets:
        sub = lay[['up_mid', m, 'lg_dur', 'lg_got']].dropna()
        if len(sub) < 30:
            out[m] = None
            continue
        A = _np.column_stack([_np.ones(len(sub)), sub.lg_dur.values, sub.lg_got.values])
        beta, *_ = _np.linalg.lstsq(A, sub[m].values, rcond=None)
        lay.loc[sub.index, m + '_r'] = sub[m].values - A @ beta
        out[m] = round(icc_one(lay, 'up_mid', m + '_r'), 4) if icc_one(lay, 'up_mid', m + '_r') is not None else None
    return out


up_n = V.groupby('up_mid').size()
R['route7_up_partition'] = dict(
    n_videos=int(len(V)), n_ups=int(up_n.size), ups_with_3plus=int((up_n >= 3).sum()),
    eta2_tname_from_exp56='见 exp56 route_C（分区 21 组、1,710 支：η² 0.091–0.151，偏 η² 0.090–0.133）',
    icc_up={m: (round(icc_one(V, 'up_mid', m), 4) if icc_one(V, 'up_mid', m) is not None else None) for m in MET},
    icc_up_controlling_duration_volume=_icc_partial(V, MET),
    icc_note='偏 ICC = 先把指标对 log(duration)+log(got) 残差化，再算组内相关；'
             '若偏 ICC 明显小于原始 ICC，说明"同 UP 更相似"主要由时长/体量解释',
    status='待判（UP 内 ICC 与分区 η² 比较：ICC 明显 >0 才说明同 UP 内部更相似）',
    caveat='热榜样本非全体视频代表；同 UP 视频可能共享题材与粉丝结构')

# ---------------- R8 多维表示 ----------------
mv = (tx.groupby('bvid').agg(n=('tnorm', 'size'), users=('user_hash', 'nunique')).reset_index())
mv = mv.merge(bk.groupby('bvid').apply(lambda d: pd.Series({
    'multi': float(np.average(d.multi_share, weights=d.n)),
    'repeat': float(np.average(d.repeat_share, weights=d.n))}), include_groups=False).reset_index(), on='bvid')
mv = mv.merge(hh.groupby('bvid').agg(hhi=('hhi', 'mean'), top1=('top1', 'mean')).reset_index(), on='bvid')
mv = mv.merge(ex.groupby('bvid').agg(colored=('colored', 'mean'), long=('long', 'mean')).reset_index(), on='bvid')
mv = mv.merge(vm[['bvid', 'got', 'coverage', 'user_cov', 'up_mid', 'tname', 'duration']], on='bvid', how='left')
NEW = ['multi', 'repeat', 'hhi', 'top1', 'colored', 'long']
mv = mv.merge(feats, on='bvid', how='inner')
red = {}
for c in NEW:
    sub = mv[[c] + MET].dropna()
    if len(sub) < 20:
        continue
    X = np.column_stack([np.ones(len(sub))] + [sub[m].values for m in MET])
    beta, *_ = np.linalg.lstsq(X, sub[c].values, rcond=None)
    pred = X @ beta
    ss_res = ((sub[c].values - pred) ** 2).sum()
    ss_tot = ((sub[c].values - sub[c].mean()) ** 2).sum()
    red[c] = dict(R2_from_existing=round(float(1 - ss_res / ss_tot), 4) if ss_tot > 0 else None)
R['route8_representation'] = dict(n_videos=int(len(mv)), redundancy=red,
                                  status='待判（R2 接近 1 ⇒ 与现有 8 维重复；明显 <1 ⇒ 携带新信息）',
                                  caveat='无独立人工评价，只能判"是否重复"，不能判"是否更有解释力"')

# ---------------- 交叉验证：新指标之间的相关性（判断是否重复） ----------------
cc = mv[['multi', 'repeat', 'hhi', 'top1', 'colored', 'long']].corr()
R['cross'] = dict(
    corr_matrix={a: {b: round(float(cc.loc[a, b]), 3) for b in cc.columns} for a in cc.index},
    redundant_pairs=[f'{a}~{b}' for a in cc.columns for b in cc.columns
                     if a < b and abs(cc.loc[a, b]) >= 0.8],
    note='|r|≥0.8 视为重复信息；R1（multi）与 R5（hhi/top1）是否互补看 multi~hhi 与 multi~top1')

with open(f'{OUT}/results.json', 'w', encoding='utf-8') as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1)[:6000])
print('EXP57-DONE →', f'{OUT}/results.json')
