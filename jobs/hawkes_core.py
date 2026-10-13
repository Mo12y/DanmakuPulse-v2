#!/usr/bin/env python3
"""
hawkes_core.py — m2_l3_hawkes 的纯函数核心（08 §5.1；可单测、可被 RDD 闭包分发）

模型：λ(t) = μ(t) + α·Σ_{t_j<t} e^(−β(t−t_j))；α = σ·β（σ=分支比∈(0,0.95) 保平稳）
  - μ(t) 分段：first_batch 日前后 μ0/μ1（μ1≥μ0 由参数化 μ1=μ0+e^η 强制）
  - 似然 O(n) 向量化（cumsum 递推）；优化 L-BFGS-B（scipy），bounds 由调用方给
  - 外源冲击：似然比检验 H0: μ1=μ0（边界检验 → p = 0.5·χ²₁ 生存函数）
  - 质检：时间变换残差 τ_i=Λ(t_i) 的一阶差分 ~ Exp(1) 的 KS 检验（p>0.05 → pass）
  - 失败降级：日弹幕量幂律拟合 c_d ~ A·(d−d*)^(−p)

时间单位 = 天（自 pubdate 起）；λ/μ/α 单位 events/day；β 单位 1/day（半衰期 = ln2/β 天）。
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, kstest

DAY = 86400.0
BOUNDS = {'mu0': (1e-6, 1e5), 'eta': (-20.0, 20.0), 'sigma': (1e-6, 0.95), 'beta': (0.01, 5.0)}


def unpack(x, has_split):
    mu0, eta, sigma, beta = x
    mu1 = mu0 + np.exp(eta) if has_split else mu0
    return mu0, mu1, sigma, beta


def recursion_A(t, beta):
    """A_i = Σ_{j<i} e^(−β(t_i−t_j))，cumsum 向量化（数值域：βT ≤ 5·21=105 < ln float64 max）"""
    u = np.exp(beta * (t - t[0]))
    cs = np.cumsum(u)
    A = np.empty_like(t)
    A[0] = 0.0
    A[1:] = np.exp(-beta * (t[1:] - t[0])) * cs[:-1]
    return A


def nll(x, t, T, has_split, T0):
    """负对数似然（含 −∫λ）。任何非法域返回大值（优化器会绕开）。"""
    mu0, mu1, sigma, beta = unpack(x, has_split)
    if mu0 <= 0 or mu1 <= 0 or not (0 < sigma < 0.95) or beta <= 0:
        return 1e12
    alpha = sigma * beta
    with np.errstate(over='ignore', invalid='ignore'):
        A = recursion_A(t, beta)
        mu_t = np.where(t < T0, mu0, mu1) if has_split else np.full_like(t, mu0)
        lam = mu_t + alpha * A
        if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
            return 1e12
        ll = np.sum(np.log(lam))
        # μ 积分（分段）
        if has_split:
            ll -= mu0 * min(T, T0) + mu1 * max(0.0, T - T0)
        else:
            ll -= mu0 * T
        # α·Σ ∫ e^{−β(s−t_i)} ds over [t_i, T]
        ll -= (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - t)))
    return -ll


def fit_mle(t, T, has_split, T0, x0_list):
    """多初值 L-BFGS-B；返回最优 x 或 None。"""
    bounds = [BOUNDS['mu0'], BOUNDS['eta'], BOUNDS['sigma'], BOUNDS['beta']]
    best = None
    for x0 in x0_list:
        try:
            res = minimize(nll, np.asarray(x0, dtype=float), args=(t, T, has_split, T0),
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 400, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun >= 1e11 or not np.all(np.isfinite(res.x)):
                continue
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue
    return best


def ks_pvalue(t, T, has_split, T0, x):
    """时间变换（compensator）残差 KS：τ 差分 ~ Exp(1)，p>0.05 → pass"""
    mu0, mu1, sigma, beta = unpack(x, has_split)
    alpha = sigma * beta
    A = recursion_A(t, beta)
    k = np.arange(1, len(t) + 1, dtype=float)
    mu_part = np.where(t < T0, mu0 * t, mu0 * min(T, T0) + mu1 * (t - T0)) if has_split \
        else mu0 * t
    zed = mu_part + (alpha / beta) * ((k - 1.0) - A)
    diffs = np.diff(np.concatenate([[0.0], zed]))
    diffs = diffs[np.isfinite(diffs) & (diffs >= 0)]
    if len(diffs) < 10:
        return 0.0
    return float(kstest(diffs, 'expon').pvalue)


def powerlaw_fit(day_idx, counts):
    """日弹幕量幂律：c_d ~ A·(d−d*)^(−p)，d* = argmax(c_d)，拟合 d>d* 的衰减段。"""
    counts = np.asarray(counts, dtype=float)
    if counts.size < 4 or counts.max() <= 0:
        return None
    dstar = int(np.argmax(counts))
    xs, ys = [], []
    for d in range(dstar + 1, counts.size):
        if counts[d] > 0:
            xs.append(np.log(d - dstar))
            ys.append(np.log(counts[d]))
    if len(xs) < 3:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1.0 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else float('nan')
    return {'p': float(-slope), 'A': float(np.exp(intercept)), 'd_star': dstar,
            'n_fit_points': len(xs), 'r2': float(r2)}


def powerlaw_halflife(p):
    """幂律降级行的半衰期（天）：峰后衰减至半值 2^(1/p)（05 §2.3.2「仍产出衰减指数与半衰期」）。"""
    return float(2.0 ** (1.0 / p)) if (p is not None and p > 0) else None


def fit_video(send_times, pubdate, first_batch_epoch, cfg):
    """单视频完整拟合（RDD 闭包入口）。返回 dict 或 {'skip': reason}。

    send_times: 已排序的 epoch 秒（仅该视频最新快照的弹幕）；pubdate: epoch 秒
    cfg: {'min_events', 'window_days', 'ks_p'}
    """
    if pubdate is None or pubdate <= 0:
        return {'skip': 'no_pubdate'}
    t_all = np.asarray(send_times, dtype=float)
    t_all = np.unique(t_all[(t_all >= pubdate) & (t_all <= pubdate + cfg['window_days'] * DAY)])
    if t_all.size == 0:
        return {'skip': 'no_events_in_window'}
    # 窗口 [pubdate, min(末事件, pubdate+21d)]（已在上面截断到 21d；末事件即窗口内最后一条）
    t = (t_all - pubdate) / DAY
    T = float(t[-1]) + 1e-9
    if t.size < cfg['min_events']:
        return {'skip': 'below_min_events', 'n_events': int(t.size)}
    if first_batch_epoch is None or first_batch_epoch <= 0:
        has_split, T0 = False, 0.0
    else:
        T0 = (first_batch_epoch - pubdate) / DAY
        has_split = 0.0 < T0 < T
        T0 = min(max(T0, 0.0), T)
    rate = t.size / max(T, 1e-6)
    x0s = [(rate * 0.3, 0.0, 0.3, 0.5), (rate * 0.6, 0.0, 0.5, 1.0),
           (max(rate * 0.1, 1e-5), 0.0, 0.2, 0.2)]
    base = {'n_events': int(t.size), 'T_days': T}
    res = fit_mle(t, T, has_split, T0, x0s)
    if res is None:
        out = {'fit_quality': 'powerlaw', 'method': 'powerlaw'}
    else:
        mu0, mu1, sigma, beta = unpack(res.x, has_split)
        ks_p = ks_pvalue(t, T, has_split, T0, res.x) if has_split else ks_pvalue(t, T, False, 0.0, res.x)
        out = {
            'mu0': float(mu0), 'mu1': float(mu1), 'alpha': float(sigma * beta),
            'beta': float(beta), 'branching_ratio': float(sigma),
            'decay_halflife_d': float(np.log(2) / beta),
            'fit_ks_p': float(ks_p),
            'method': 'mle' if ks_p > cfg['ks_p'] else 'powerlaw',
            'fit_quality': 'pass' if ks_p > cfg['ks_p'] else 'powerlaw',
            'exo_jump': float(mu1 - mu0) if has_split else 0.0,
        }
        if has_split:
            # LR 检验 H0: μ1=μ0（η→−∞，边界 → 混合分布 → p = 0.5·χ²₁ 生存函数）
            res0 = fit_mle(t, T, False, 0.0, [x0s[0][:2] + x0s[0][2:], (rate * 0.5, 0.0, 0.3, 0.5)])
            if res0 is not None and res.fun < 1e11:
                lr = 2.0 * (res0.fun - res.fun)
                out['exo_sig'] = float(0.5 * chi2.sf(max(lr, 0.0), 1))
            else:
                out['exo_sig'] = None
        else:
            out['exo_sig'] = None
    # 幂律（失败降级或附加信息：无论成败都算，便于论文层随时对照）
    day_idx = np.floor(t).astype(int)
    counts = np.bincount(day_idx, minlength=int(np.floor(T)) + 1).astype(float)
    pl = powerlaw_fit(np.arange(counts.size), counts)
    out['pl_p'] = pl['p'] if pl else None
    out['pl_r2'] = pl['r2'] if pl else None
    # 降级行的「半衰期」语义（05 §2.3.2「退回幂律…仍产出衰减指数与半衰期」）：
    #   method=powerlaw → decay_halflife_d = 峰后衰减至半值天数 2^(1/p)（p>0 时）；
    #   method=mle → 保持 ln2/β（指数核半衰期）。逐行口径由 method 列自解释。
    if out.get('method') == 'powerlaw' and pl and pl['p'] > 0:
        out['decay_halflife_d'] = powerlaw_halflife(pl['p'])
    return {**base, **out}


def to_row(bvid, src_batch, r, engine_version):
    """拟合结果 dict → 输出表行（跳过视频返回 None）。"""
    if 'skip' in r:
        return None
    def g(k):
        v = r.get(k)
        return None if v is None else (float(v) if not isinstance(v, int) else v)
    return (bvid, src_batch,
            g('mu0'), g('mu1'), g('alpha'), g('beta'), g('branching_ratio'),
            g('decay_halflife_d'), g('exo_jump'), g('exo_sig'), g('fit_ks_p'),
            r.get('method'), int(r['n_events']), g('pl_p'), g('pl_r2'),
            engine_version)
