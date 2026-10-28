#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp56_four_routes.py — 四条新路线最小可行性验证（A 相似检索 / B 峰生命周期 / C 条件差异 / D 表示学习）

数据：data-out/exp56/（Spark 导出：units/feats/curves1pct）+ data-out/c1/（C1 标注与 60s 曲线）。
原则：不预设任何路线成功；评价标准在看结果前写死（见各节 PRE）；缺外部评价就报"尚不能判断"。

PRE（预注册，先于结果）：
 A：查询集 = C1 池按 tname 分层抽 10 支（seed=20261028）；候选池 = C1 池全部（254 支）；
    距离 = 原始计数欧氏 / 归一化组成欧氏 / DTW(归一化, 带半径10) / 8 维特征欧氏(z 分数)；
    一致性 = 各方法 top-10 的 Jaccard；幅度混淆 = 检索名次与 |Δlog(got)| 的 Spearman。
 B：峰检测规则 = c[t] 为 ±2 窗内局部最大 且 c[t] ≥ base+2·sqrt(base+1)（base=±6 窗中位）且 峰间距≥3；
    生命周期 = 上升时间/峰宽(半高)/衰减时间/峰后 10 桶累计/是否二次反弹(10 桶内、≥50% 峰高)。
    稳健性网格 = smooth{1,3,5} × thr_k{1.5,2,2.5} × min_dist{2,3,5}（仅稳健性检查，不用于选参）。
    人工参照 = C1 裁决中的 B 段（爆发段）作为峰区真值（±1 桶容差）。
 C：分组变量预选 = tname(样本≥20 的分区) / 时长三档 / 热榜位次三档；指标 = 5 个节奏特征；
    效应量 η² + 视频级 bootstrap CI；多重比较 15 检验 Bonferroni α=0.05/15。
 D：内部充分性 = PCA(4/8/16) 曲线表示与全曲线检索名次的 Kendall τ；8 维特征 vs 曲线的 τ。
"""
import csv
import glob
import json
import os

import numpy as np

REPO = '/home/molly/projects/danmaku-v2'
EXP = f'{REPO}/data-out/exp56'
C1 = f'{REPO}/data-out/c1'
OUT = f'{REPO}/experiments/exp56_four_routes'
SEED = 20261028
os.makedirs(OUT, exist_ok=True)
rng_global = np.random.default_rng(SEED)


def read_csv_dir(d):
    rows = []
    for p in sorted(glob.glob(f'{d}/*.csv')):
        with open(p, encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f))
    return rows


# ---------------------------------------------------------------- 数据装载
units = {r['bvid']: r for r in read_csv_dir(f'{EXP}/units_csv')}
for bv, r in units.items():
    for k in ('duration', 'danmaku_got', 'n_buckets_expected', 'n_buckets_nonzero', 'rank_proxy'):
        r[k] = int(float(r[k])) if r[k] not in ('', None) else None
    r['coverage_ratio'] = float(r['coverage_ratio'])
    r['c1_eligible'] = (r['c1_eligible'] == 'true')

feats = {r['bvid']: r for r in read_csv_dir(f'{EXP}/feats_csv')}
FKEYS_ALL = ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'ending_5pct', 'mid_ratio',
             'early_lag_ratio', 'peak_cnt', 'peak_max_z']
for bv, r in feats.items():
    for k in FKEYS_ALL:
        r[k] = float(r[k]) if r[k] not in ('', None) else np.nan
# 只用非空率 ≥90% 的特征列（peak_max_z 实测全空，剔除并记录）
_n = len(feats)
FKEYS = [k for k in FKEYS_ALL if np.mean([not np.isnan(feats[b][k]) for b in feats]) >= 0.90]
R_DROPPED = [k for k in FKEYS_ALL if k not in FKEYS]

curve_sparse = {}
for r in read_csv_dir(f'{EXP}/curves1pct_csv'):
    curve_sparse.setdefault(r['bvid'], {})[int(r['bucket1pct'])] = float(r['cnt'])


def dense100(bv):
    s = curve_sparse.get(bv, {})
    return np.array([s.get(i, 0.0) for i in range(100)], float)


POOL = sorted([bv for bv, r in units.items() if r['c1_eligible'] and bv in curve_sparse and bv in feats])
R = {'meta': dict(seed=SEED, n_units=len(units), n_c1_pool=len(POOL))}
print(f'units={len(units)} C1 池(可用曲线+特征)={len(POOL)}', flush=True)

# ================================================================ A：相似性检索
print('[A] 相似性检索…', flush=True)
CURVES = {bv: dense100(bv) for bv in POOL}
COMP = {bv: (v / v.sum() if v.sum() > 0 else v) for bv, v in CURVES.items()}
FZ = {}
for k in FKEYS:
    vals = np.array([feats[bv][k] for bv in POOL], float)
    mu, sd = np.nanmean(vals), np.nanstd(vals)
    FZ[k] = {bv: ((feats[bv][k] - mu) / sd if sd > 0 and not np.isnan(feats[bv][k]) else 0.0) for bv in POOL}


def d_euclid_raw(a, b):
    return float(np.linalg.norm(CURVES[a] - CURVES[b]))


def d_euclid_norm(a, b):
    return float(np.linalg.norm(COMP[a] - COMP[b]))


def d_feat(a, b):
    return float(np.sqrt(sum((FZ[k][a] - FZ[k][b]) ** 2 for k in FKEYS)))


def dtw(a, b, band=10):
    x, y = COMP[a], COMP[b]
    n = 100
    INF = 1e18
    prev = np.full(n + 1, INF); prev[0] = 0.0
    D = np.full((n + 1, n + 1), INF); D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(n, i + band)
        for j in range(lo, hi + 1):
            c = abs(x[i - 1] - y[j - 1])
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, n])


# 分层抽 10 支查询（按 tname 轮转，避免只挑明显样本）
by_t = {}
for bv in POOL:
    by_t.setdefault(units[bv]['tname'], []).append(bv)
queries, ti = [], 0
tl = sorted(by_t, key=lambda t: -len(by_t[t]))
while len(queries) < 10:
    t = tl[ti % len(tl)]
    cand = [b for b in sorted(by_t[t]) if b not in queries]
    if cand:
        queries.append(cand[int(rng_global.integers(0, len(cand)))])
    ti += 1
    if ti > 200:
        break


def rank_by(dfun, q, k=10):
    sc = [(dfun(q, b), b) for b in POOL if b != q]
    sc.sort()
    return [b for _, b in sc[:k]]


top = {name: {} for name in ('raw', 'norm', 'dtw', 'feat')}
for q in queries:
    top['raw'][q] = rank_by(d_euclid_raw, q)
    top['norm'][q] = rank_by(d_euclid_norm, q)
    top['dtw'][q] = rank_by(dtw, q)
    top['feat'][q] = rank_by(d_feat, q)

names = ['raw', 'norm', 'dtw', 'feat']
jac = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        v = [len(set(top[a][q]) & set(top[b][q])) / len(set(top[a][q]) | set(top[b][q])) for q in queries]
        jac[f'{a}~{b}'] = round(float(np.mean(v)), 4)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


# 幅度混淆：检索名次 vs |Δlog(got)|
conf = {}
for name, dfun in (('raw', d_euclid_raw), ('norm', d_euclid_norm), ('feat', d_feat)):
    vals = []
    for q in queries:
        sc = sorted([(dfun(q, b), b) for b in POOL if b != q])
        rank = np.arange(len(sc), 0, -1)                # 越靠前名次越高
        dg = np.array([abs(np.log1p(units[b]['danmaku_got']) - np.log1p(units[q]['danmaku_got'])) for _, b in sc])
        vals.append(spearman(rank, dg))
    conf[name] = round(float(np.mean(vals)), 4)

# ---------- D 的内部充分性（与 A 共用数据）
pca_suff = {}
Xmat = np.array([COMP[bv] for bv in POOL])
Xc = Xmat - Xmat.mean(0)
U_, S_, Vt = np.linalg.svd(Xc, full_matrices=False)
for k in (4, 8, 16):
    Z = Xc @ Vt[:k].T
    taus = []
    for q in queries:
        qi = POOL.index(q)
        dd = np.linalg.norm(Z - Z[qi], axis=1)
        order_pca = [POOL[i] for i in np.argsort(dd)[:10] if POOL[i] != q]
        taus.append(len(set(order_pca) & set(top['norm'][q])) / 10.0)
    pca_suff[f'PCA{k}'] = round(float(np.mean(taus)), 4)
pca_suff['feat8_vs_curve_top10_overlap'] = round(
    float(np.mean([len(set(top['feat'][q]) & set(top['norm'][q])) / 10 for q in queries])), 4)

R['route_A'] = dict(n_pool=len(POOL), queries=queries, inter_metric_top10_jaccard=jac,
                    magnitude_confound_spearman_with_dlog_got=conf,
                    external_human_eval='缺失（无成对相似性人工判断；C1 只是阶段边界标注）',
                    status='尚不能判断（定义可复现、内部一致性已量化；缺独立人工评价）')
R['route_D'] = dict(pca_top10_overlap_with_curve=pca_suff,
                    supervision='无监督信号（无相似性配对标签）；30 支仅边界标注、非相似性判断',
                    status='尚不能判断（缺独立评价；见 A 的外部评价缺口）')
print(f"[A] Jaccard={jac} 幅度混淆={conf} | [D] {pca_suff}", flush=True)

# ================================================================ B：峰生命周期
print('[B] 峰生命周期…', flush=True)
cur30 = {}
for p in sorted(glob.glob(f'{C1}/curves_csv/*.csv')):
    with open(p, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            cur30.setdefault(r['bvid'], {})[int(r['bucket60'])] = (float(r['cnt']), r['bucket_state'])
ann = {}
with open(f'{C1}/ann_final.csv', encoding='utf-8-sig', newline='') as f:
    for r in csv.DictReader(f):
        ann.setdefault(r['bvid'], []).append((int(r['start_bucket']), int(r['end_bucket']), r['category']))


def series_of(bv):
    s = cur30[bv]
    n = max(s) + 1
    c = np.array([s.get(i, (0.0, 'unknown_fetch'))[0] for i in range(n)], float)
    return c


def smooth(c, k):
    if k <= 1:
        return c.copy()
    ker = np.ones(k) / k
    return np.convolve(c, ker, mode='same')


def peaks(c, k_smooth=3, thr_k=2.0, min_dist=3, half=6, min_excess=3.0):
    '''v1.1（2026-10-28 修复）：首版对平滑序列用严格不等号做 ±2 局部极大，
    低计数平台上并列值导致永不触发（581 桶仅 13 峰，召回 0.056）。
    修复：局部极大改为"≥ 窗内最大"（允许并列，用原始计数比较），并加最小超出量 min_excess。
    属实现缺陷修复，不改预注册的阈值族（base+thr_k·sqrt(base+1)）。'''
    n = len(c)
    out = []
    for t in range(2, n - 2):
        # v1.2（2026-10-28）：局部极大改用原始计数比较。平滑会把峰位挪到 t±1，
        # 使尖峰自身不满足"平滑序列的 ±2 极大"（实测 BV16uJy6CEuX t=6 被 t=7 压过）。
        if c[t] < float(np.max(c[t - 2:t + 3])) - 1e-9:
            continue
        lo, hi = max(0, t - half), min(n, t + half + 1)
        base = float(np.median(c[lo:hi]))
        if c[t] < base + thr_k * np.sqrt(base + 1.0) or (c[t] - base) < min_excess:
            continue
        if out and t - out[-1] < min_dist:
            if c[t] > c[out[-1]]:
                out[-1] = t
            continue
        out.append(t)
    return out


def lifecycle(c, p):
    n = len(c)
    lo, hi = max(0, p - 6), min(n, p + 7)
    base = float(np.median(c[lo:hi]))
    pk = float(c[p])
    rise = 0
    i = p
    while i > 0 and c[i - 1] <= c[i]:
        i -= 1; rise += 1
    half = base + (pk - base) / 2.0
    w = 1
    i = p
    while i > 0 and c[i - 1] >= half:
        i -= 1; w += 1
    i = p
    while i < n - 1 and c[i + 1] >= half:
        i += 1; w += 1
    decay, i = 0, p
    while i < n - 1 and c[i + 1] > base + 1.0:
        i += 1; decay += 1
    post = float(c[p + 1:p + 11].sum())
    rebound = 0
    j = min(n - 1, p + decay + 1)
    seg = c[j:min(n, j + 11)]
    if len(seg) and float(seg.max()) >= 0.5 * pk:
        rebound = 1
    return dict(rise=rise, width_half=w, decay=decay, post10=round(post, 1), rebound=rebound,
                peak=round(pk, 1), base=round(base, 1))


# 主口径（预注册参数）
peaks_main = {bv: peaks(series_of(bv)) for bv in sorted(cur30)}
# 人工参照：C1 B 段
bsegs = {bv: [(s, e) for s, e, c in ss if c == 'B'] for bv, ss in ann.items()}
tp = fp = fn = 0
for bv in sorted(cur30):
    pd_ = peaks_main[bv]; tr = bsegs.get(bv, [])
    used = set()
    for p in pd_:
        hit = [k for k, (s, e) in enumerate(tr) if k not in used and s - 1 <= p <= e + 1]
        if hit:
            used.add(hit[0]); tp += 1
        else:
            fp += 1
    fn += len(tr) - len(used)
prec = tp / (tp + fp) if tp + fp else 0.0
rec = tp / (tp + fn) if tp + fn else 0.0

# 生命周期指标分布（仅主口径峰）
lc = [lifecycle(series_of(bv), p) for bv in sorted(cur30) for p in peaks_main[bv]]
lc_med = {k: (round(float(np.median([x[k] for x in lc])), 2) if lc else None) for k in
          ('rise', 'width_half', 'decay', 'post10', 'peak', 'base')}
lc_rebound = round(float(np.mean([x['rebound'] for x in lc])), 3) if lc else None

# 稳健性：9 参数组合（smooth{1,3,5} × thr_k{1.5,2,2.5}），min_dist 固定 3
grid = {}
for ks in (1, 3, 5):
    for tk in (1.5, 2.0, 2.5):
        pv = {bv: peaks(series_of(bv), k_smooth=ks, thr_k=tk) for bv in sorted(cur30)}
        nn = sum(len(v) for v in pv.values())
        grid[f's{ks}_k{tk}'] = dict(n_peaks=nn, per_video_med=round(float(np.median([len(v) for v in pv.values()])), 1))
counts = [g['n_peaks'] for g in grid.values()]
stab_count = round(float(np.std(counts) / np.mean(counts)), 3) if counts else None

# 窗口充分性：峰前后各有 ≥5 个观测桶
ok = tot = 0
for bv in sorted(cur30):
    c = series_of(bv); n = len(c)
    for p in peaks_main[bv]:
        tot += 1
        if p >= 5 and p + 5 < n:
            ok += 1
ball = [(bv, s0, e0) for bv, ss in ann.items() for s0, e0, c0 in ss if c0 == 'B']
head = [x for x in ball if x[1] <= 1]
inter = [x for x in ball if x[1] > 1]
tp_i = 0
for bv, s0, e0 in inter:
    if any(s0 - 1 <= p <= e0 + 1 for p in peaks_main.get(bv, [])):
        tp_i += 1
R['route_B'] = dict(n_videos=len(cur30), n_peaks=sum(len(v) for v in peaks_main.values()),
                    human_B_segments=len(ball), head_bursts=len(head), interior_bursts=len(inter),
                    head_burst_share=round(len(head) / max(len(ball), 1), 3),
                    interior_recall=round(tp_i / max(len(inter), 1), 3),
                    head_unmeasurable_note='片头 B 段（起于 bucket 0-1）无峰前历史、且局部极大需 ±2 窗，'
                                           '结构上不可检测/不可测峰前过程',
                    human_ref='C1 裁决 B 段（爆发段）±1 桶', precision=round(prec, 4),
                    recall=round(rec, 4), tp=tp, fp=fp, fn=fn,
                    lifecycle_median=lc_med, rebound_share=lc_rebound,
                    robustness_grid=grid, peak_count_cv_across_grid=stab_count,
                    window_adequacy=dict(peaks_with_5_buckets_each_side=ok, total=tot,
                                         share=round(ok / tot, 3) if tot else None),
                    pre_registered='smooth=3, thr_k=2.0, min_dist=3, half=6（见文件头 PRE）',
                    status='待判（检测可靠性可测；生命周期指标稳定性见 robustness）')
print(f"[B] peaks={R['route_B']['n_peaks']} P={prec:.3f} R={rec:.3f} CV={stab_count}", flush=True)

# ================================================================ C：条件差异
print('[C] 条件差异…', flush=True)
CANDS = [bv for bv in units if bv in feats]
MET = ['rhythm_entropy', 'burst_cv', 'opening_5pct', 'mid_ratio', 'peak_cnt'][:5]
MET = [m for m in MET if m in FKEYS] + [m for m in ('peak_max_z',) if m in FKEYS]


def eta2(vals, groups):
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    ok = ~np.isnan(vals)
    vals, groups = vals[ok], groups[ok]
    if len(set(groups)) < 2 or len(vals) < 10:
        return None
    gm = vals.mean()
    ss_tot = ((vals - gm) ** 2).sum()
    ss_b = sum(len(vals[groups == g]) * (vals[groups == g].mean() - gm) ** 2 for g in set(groups))
    return float(ss_b / ss_tot) if ss_tot > 0 else None


def boot_eta(vals, groups, B=300):
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    idx = np.arange(len(vals)); out = []
    for _ in range(B):
        pick = rng_global.choice(idx, len(idx), True)
        e = eta2(vals[pick], groups[pick])
        if e is not None:
            out.append(e)
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)] if out else None


def dur_bin(d):
    return '<5min' if d < 300 else ('5-15min' if d <= 900 else '>15min')


def rank_bin(rk):
    if rk is None:
        return None
    return 'top50' if rk < 50 else ('51-100' if rk < 100 else '101-200')


tc = {}
for bv in CANDS:
    tc.setdefault(units[bv]['tname'], 0)
    tc[units[bv]['tname']] += 1
big_t = {t for t, n in tc.items() if n >= 20}

res_c = {}
for gname, gfun in (('tname', lambda bv: units[bv]['tname'] if units[bv]['tname'] in big_t else None),
                    ('duration', lambda bv: dur_bin(units[bv]['duration'])),
                    ('rank', lambda bv: rank_bin(units[bv]['rank_proxy']))):
    bvs = [bv for bv in CANDS if gfun(bv) is not None]
    grp = [gfun(bv) for bv in bvs]
    cell = dict(n=len(bvs), n_groups=len(set(grp)))
    for m in MET:
        vals = [feats[bv][m] for bv in bvs]
        e = eta2(vals, grp)
        cell[m] = dict(eta2=round(e, 4) if e is not None else None,
                       ci=boot_eta(vals, grp) if e is not None else None)
    # 混淆参照：log(got) 本身的 η²
    lg = np.array([np.log1p(units[bv]['danmaku_got']) for bv in bvs], float)
    e_got = eta2(lg, grp)
    cell['log_got_eta2_reference'] = round(e_got, 4) if e_got is not None else None
    # 混杂控制：各指标对 log(got) 做线性回归取残差后再算 η²（偏 η²）
    cell['partial_eta2_controlling_log_got'] = {}
    for m in MET:
        vals = np.array([feats[bv][m] for bv in bvs], float)
        ok = ~np.isnan(vals)
        if ok.sum() < 10:
            cell['partial_eta2_controlling_log_got'][m] = None
            continue
        A = np.column_stack([np.ones(ok.sum()), lg[ok]])
        beta, *_ = np.linalg.lstsq(A, vals[ok], rcond=None)
        resid = vals[ok] - A @ beta
        e = eta2(resid, np.array(grp)[ok])
        cell['partial_eta2_controlling_log_got'][m] = round(e, 4) if e is not None else None
    res_c[gname] = cell
R['route_C'] = dict(n_videos=len(CANDS), metrics=MET, by=res_c,
                    correction='Bonferroni 15 检验 α=0.05/15=0.0033（η² 无 p 值，用 bootstrap CI 是否含 0 判读）',
                    status='待判（效应量与 CI 已给；注意 log_got 参照）')
print('[C]', json.dumps({k: {m: (v[m]['eta2'] if isinstance(v[m], dict) else v[m]) for m in list(MET) + ['log_got_eta2_reference']} for k, v in res_c.items()}, ensure_ascii=False), flush=True)

with open(f'{OUT}/results.json', 'w', encoding='utf-8') as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print('EXP56-DONE →', f'{OUT}/results.json')
