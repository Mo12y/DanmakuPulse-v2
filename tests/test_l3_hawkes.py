
# ── L3 Hawkes 单测（2026-10-13；B4.1 验收「自实现 MLE + KS + 幂律降级」）──
# 运行：~/danmaku-venv/bin/python -m pytest tests/test_l3_hawkes.py -q
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'jobs'))


def _sim_hawkes(mu, sigma, beta, T, seed, mu1=None, T0=None):
    """Ogata thinning 合成指数核 Hawkes（可选 μ 在 T0 处阶跃）。"""
    import numpy as np
    rng = np.random.RandomState(seed)
    alpha = sigma * beta
    t, s = [], 0.0
    while True:
        mu_t = mu if (T0 is None or s < T0) else (mu1 if mu1 is not None else mu)
        lam = mu_t + alpha * np.sum(np.exp(-beta * (s - np.array(t)))) if t else mu_t
        M = lam * 1.5 + 1e-9
        s += rng.exponential(1.0 / M)
        if s > T:
            break
        mu_s = mu if (T0 is None or s < T0) else (mu1 if mu1 is not None else mu)
        lam_s = mu_s + alpha * np.sum(np.exp(-beta * (s - np.array(t)))) if t else mu_s
        if rng.rand() < lam_s / M:
            t.append(s)
    return np.array(t)


CFG = {'min_events': 100, 'window_days': 21.0, 'ks_p': 0.05}


def test_recursion_A_matches_bruteforce():
    import numpy as np
    from hawkes_core import recursion_A
    tt = np.sort(np.random.RandomState(0).rand(40)) * 8 + 0.01
    A = recursion_A(tt, 0.7)
    A_bf = np.array([np.sum(np.exp(-0.7 * (tt[i] - tt[:i]))) if i > 0 else 0.0
                     for i in range(len(tt))])
    assert np.abs(A - A_bf).max() < 1e-10


def test_unpack_enforces_mu1_ge_mu0():
    from hawkes_core import unpack
    for eta in (-10.0, 0.0, 3.0):
        mu0, mu1, _, _ = unpack([2.0, eta, 0.3, 1.0], True)
        assert mu1 >= mu0
    mu0, mu1, _, _ = unpack([2.0, 0.0, 0.3, 1.0], False)
    assert mu1 == mu0


def test_fit_stationary_recovers_and_ks_pass():
    from hawkes_core import fit_video
    t = _sim_hawkes(mu=5.0, sigma=0.5, beta=1.0, T=30, seed=42)
    r = fit_video(list(t * 86400.0 + 1e9), 1e9, None, CFG)
    assert r['n_events'] >= 100
    assert 0.0 < r['branching_ratio'] < 0.95
    assert r['method'] == 'mle' and r['fit_ks_p'] > 0.05, f"KS 应通过：{r}"


def test_fit_jump_recovers_mu_and_flags_exo():
    from hawkes_core import fit_video
    t = _sim_hawkes(mu=5.0, sigma=0.4, beta=0.8, T=30, seed=7, mu1=15.0, T0=10.0)
    pub = 1e9
    r = fit_video(list(t * 86400.0 + pub), pub, pub + 10 * 86400.0, CFG)
    assert abs(r['mu0'] - 5.0) < 1.5, r
    assert abs(r['mu1'] - 15.0) < 3.0, r
    assert r['exo_jump'] > 5.0 and r['exo_sig'] < 0.05, r


def test_powerlaw_recovery():
    import numpy as np
    from hawkes_core import powerlaw_fit
    d = np.arange(20)
    c = 500.0 * np.maximum(d - 3, 1e-9) ** -1.3
    c[:4] = [50, 200, 500, 300]           # 峰值前的上升段（不参与拟合）
    pl = powerlaw_fit(d, c)
    assert pl is not None
    assert abs(pl['p'] - 1.3) < 0.15, pl
    assert pl['r2'] > 0.85, pl


def test_gate_skip_below_min_events():
    import numpy as np
    from hawkes_core import fit_video
    # 50 个事件落在窗口内（0.4d 间隔 → 20d ≤ 21d 窗口；无窗口截断）
    t = np.arange(50) * 0.4 * 86400.0 + 1e9
    r = fit_video(list(t), 1e9, None, CFG)
    assert r == {'skip': 'below_min_events', 'n_events': 50}, r
    # 窗口截断：50 天事件（1d 间隔）→ 窗口 [0,21d] 内仅 22 个
    t2 = np.arange(50) * 86400.0 + 1e9
    r2 = fit_video(list(t2), 1e9, None, CFG)
    assert r2 == {'skip': 'below_min_events', 'n_events': 22}, r2


def test_powerlaw_halflife_semantics_on_fallback():
    """降级行 decay_halflife_d = 2^(1/p)（05 §2.3.2「仍产出衰减指数与半衰期」）。"""
    from hawkes_core import powerlaw_halflife
    assert abs(powerlaw_halflife(1.3) - 2.0 ** (1.0 / 1.3)) < 1e-12
    assert powerlaw_halflife(1.0) == 2.0
    assert powerlaw_halflife(-0.5) is None and powerlaw_halflife(None) is None
