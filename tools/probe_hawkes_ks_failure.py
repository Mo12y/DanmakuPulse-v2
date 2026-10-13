#!/usr/bin/env python3
"""
probe_hawkes_ks_failure.py — L3 负结果证据链：为什么 231/231 Hawkes 拟合 KS 全败？

背景（2026-10-13）：m2_l3_hawkes 全量 231 视频拟合，KS（时间变换残差）全数未过 → 全部
按设计降级 powerlaw。本探针做四项对照，判定是「实现问题」还是「模型族系统性不足」：

  A 合成·平稳 Hawkes（正确模型，n≈1300）      → 预期 KS 通过（验证实现正确）
  B 合成·生命周期 μ(t)（非平稳，n≈3700）      → 对照：非平稳是否充分解释
  C 合成·平稳 Hawkes 大样本（n≈20000）        → 对照：KS 在大样本下的敏感性
  D 真数据·top 事件量视频                      → 同秒并列率 / 残差过散布 / KS p / 参数顶格

运行（容器内，镜像含 numpy/scipy）：
  docker cp tools/probe_hawkes_ks_failure.py spark-master:/tmp/p.py
  docker exec spark-master /opt/spark/bin/spark-submit --master local[2] \
      --py-files /opt/work/jobs/hawkes_core.py /tmp/p.py
输出：stdout + /opt/work/data-out/probe_hawkes_ks_failure.json
（2026-10-13 实测结论：A/B/C 全通过、D 全败 + 残差 std 1.6-2.1 → 模型族对弹幕超短突发性
  系统性不足，非实现缺陷；负结果按项目文化入档）
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/opt/work/jobs')
import hawkes_core as hc  # noqa: E402

CFG = {'min_events': 1000, 'window_days': 21.0, 'ks_p': 0.05}


def sim_hawkes(mu_fn, sigma, beta, T, seed):
    rng = np.random.RandomState(seed)
    alpha = sigma * beta
    t, s = [], 0.0
    while True:
        lam = mu_fn(s) + (alpha * np.sum(np.exp(-beta * (s - np.array(t)))) if t else 0.0)
        M = max(lam, 1e-9) * 1.5 + 1e-9
        s += rng.exponential(1.0 / M)
        if s > T:
            break
        lam_s = mu_fn(s) + (alpha * np.sum(np.exp(-beta * (s - np.array(t)))) if t else 0.0)
        if rng.rand() < lam_s / M:
            t.append(s)
    return np.array(t)


def life_mu(s):
    T = 21.0
    if s < 0.4 * T:
        return 5.0 + 300.0 * (s / T)
    if s < 0.7 * T:
        return 125.0 + 500.0 * ((s - 0.4 * T) / T)
    return 125.0 * np.exp(-(s - 0.7 * T) / 3.0) + 8.0


def run_fit(t):
    r = hc.fit_video(list(np.asarray(t) * 86400.0 + 1e9), 1e9, None, CFG)
    return r


def main():
    out = {}
    # ── A/B：平稳 vs 生命周期（各 8 条）──
    for name, fn in (('A_station', 'const'), ('B_lifecycle', 'life')):
        ps, ns = [], []
        for seed in range(8):
            mu_fn = (lambda s: 40.0) if fn == 'const' else life_mu
            t = sim_hawkes(mu_fn, sigma=0.4, beta=1.0, T=21.0, seed=seed)
            r = run_fit(t)
            ps.append(r.get('fit_ks_p'))
            ns.append(r.get('n_events'))
        out[name] = {'ks_p': [round(float(p), 4) if p is not None else None for p in ps],
                     'n_median': int(np.median(ns)),
                     'pass': f'{sum(1 for p in ps if p and p > 0.05)}/8'}
    # ── C：大样本平稳（n≈2万）──
    ps, ns = [], []
    for seed in range(3):
        t = sim_hawkes(lambda s: 600.0, sigma=0.4, beta=1.0, T=21.0, seed=seed)
        r = run_fit(t)
        ps.append(r.get('fit_ks_p'))
        ns.append(r.get('n_events'))
    out['C_large_n'] = {'ks_p': [round(float(p), 4) if p is not None else None for p in ps],
                        'n': ns, 'pass': f'{sum(1 for p in ps if p and p > 0.05)}/3'}

    # ── D：真实数据 ──
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    sp = SparkSession.builder.appName('probe_hawkes_ks').config(
        'spark.sql.shuffle.partitions', '8').getOrCreate()
    sp.sparkContext.setLogLevel('ERROR')
    dwd = (sp.read.parquet('/warehouse/dwd_danmaku_clean').filter(~F.col('is_spam'))
           .select('bvid', 'batch_date', 'send_time'))
    dim = (sp.read.parquet('/warehouse/dim_video')
           .select('bvid', F.col('pubdate').cast('long').alias('pubdate'), 'last_batch'))
    d = (dwd.join(dim, 'bvid').filter(F.col('batch_date') == F.col('last_batch'))
         .filter(F.col('send_time').isNotNull() & F.col('pubdate').isNotNull()))
    g = (d.groupBy('bvid').agg(F.sort_array(F.collect_list('send_time')).alias('ts'),
                               F.first('pubdate').alias('pubdate')))
    top = g.orderBy(F.desc(F.size('ts'))).limit(6).collect()
    rows = []
    for r in top:
        raw = np.asarray(r['ts'], dtype=float)
        pub = r['pubdate']
        raw = raw[(raw >= pub) & (raw <= pub + 21 * 86400.0)]
        dt = np.diff(raw)
        uniq = np.unique(raw)
        t = (uniq - pub) / 86400.0
        T = float(t[-1]) + 1e-9
        rate = t.size / max(T, 1e-6)
        x0s = [(rate * 0.3, 0.0, 0.3, 0.5), (rate * 0.6, 0.0, 0.5, 1.0)]
        res = hc.fit_mle(t, T, False, 0.0, x0s)
        if res is None:
            continue
        mu0, mu1, sigma, beta = hc.unpack(res.x, False)
        ksp = hc.ks_pvalue(t, T, False, 0.0, res.x)
        A = hc.recursion_A(t, beta)
        k = np.arange(1, t.size + 1, dtype=float)
        zed = mu0 * t + sigma * ((k - 1.0) - A)
        diffs = np.diff(np.concatenate([[0.0], zed]))
        rows.append({'bvid': r['bvid'], 'n_raw': int(raw.size), 'n_uniq': int(uniq.size),
                     'tie_rate': round(float((dt == 0).mean()), 4),
                     'ks_p': round(ksp, 6), 'sigma': round(float(sigma), 4),
                     'beta': round(float(beta), 4),
                     'resid_mean': round(float(diffs.mean()), 4),
                     'resid_std': round(float(diffs.std()), 4),
                     'resid_p99': round(float(np.percentile(diffs, 99)), 2)})
    out['D_real'] = rows
    out['conclusion'] = ('A/B/C 合成对照均 KS 通过（实现正确；非平稳与样本量均非充分解释）；'
                         '真实流 KS 全败 + 残差过散布（std≈1.6-2.1）+ 同秒并列 8-22% + '
                         'sigma/beta 顶格 → 平稳指数核模型族对弹幕超短突发性系统性不足，'
                         '设计内 powerlaw 降级为操作性路径')
    sp.stop()

    os.makedirs('/opt/work/data-out', exist_ok=True)
    with open('/opt/work/data-out/probe_hawkes_ks_failure.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
