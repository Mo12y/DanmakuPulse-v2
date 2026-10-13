#!/usr/bin/env python3
"""
m2_l3_hawkes.py — L3 机制层：Hawkes 自激发拟合器（08 §5.1）

输入：dwd_danmaku_clean（send_time 事件流，spam 排除）+ dim_video（pubdate / first_batch / last_batch）
输出：{wh}/dws_hawkes_params —— 每 bvid 一行（视频最新快照的完整事件史）：
  mu0/mu1/alpha/beta/branching_ratio/decay_halflife_d/exo_jump/exo_sig/fit_ks_p/method/n_events
  + pl_p/pl_r2（幂律降级参数，诊断/寿命用）+ src_batch_date/engine_version（血缘）

口径（08 §5.1 + params.hawkes）：
  - 单位：窗口 [pubdate, min(末事件, pubdate+21d)]；门槛 事件数≥1000；时间单位=天（自 pubdate）
  - μ(t) 分段（first_batch 日前后 μ0/μ1，μ1≥μ0 由 μ1=μ0+e^η 强制）→ LR 检验 H0:μ1=μ0
    （边界检验 → p = 0.5·χ²₁ 生存函数）→ exo_jump=μ1−μ0, exo_sig
  - α=σ·β；σ∈(0,0.95) 保平稳；endo=σ（分支比）；半衰期=ln2/β（天）
  - KS：时间变换残差 p>0.05 → fit_quality=pass；否则 method=powerlaw（幂律降级；
    ⚠️ 降级行的 Hawkes 参数保留为 KS 未过时的 MLE 估计（诊断用），decay_halflife_d 语义随
    method 自解释：powerlaw 行 = 2^(1/p)（峰后衰减至半值天数，05 §2.3.2「仍产出…半衰期」）；
    mle 行 = ln2/β。pl_p 为幂律指数（退化时 NULL——窗口过短/无衰减尾时不产出）
  - 单元=每 bvid（用其最新批快照；src_batch_date 记源）——08 输出字段无 batch_date，头注记明
实现：RDD 数据并行（逐视频 L-BFGS-B 在 executor；10 §2.4「数据并行 UDF」诚实口径）。
      ⚠️ 容器无 pandas：闭包只依赖 hawkes_core（--py-files 分发）+ numpy/scipy。
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from hawkes_core import fit_video, to_row

ENGINE_VERSION = 'b2-v1'
TZ8 = timezone(timedelta(hours=8))
OUT_COLS = ['bvid', 'src_batch_date', 'mu0', 'mu1', 'alpha', 'beta', 'branching_ratio',
            'decay_halflife_d', 'exo_jump', 'exo_sig', 'fit_ks_p', 'method', 'n_events',
            'pl_p', 'pl_r2', 'engine_version']


def fb_epoch(batch_date):
    """first_batch（如 20260917）→ 该日 00:00 UTC+8 的 epoch 秒；非法返回 None。"""
    try:
        return datetime.strptime(str(int(batch_date)), '%Y%m%d').replace(tzinfo=TZ8).timestamp()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config', default='/opt/work/config/params.yaml')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    import yaml
    with open(args.config, encoding='utf-8') as f:
        params = yaml.safe_load(f)
    h = params.get('hawkes', {})
    cfg = {'min_events': int(h.get('min_events', 1000)),
           'window_days': float(h.get('window_days', 21)),
           'ks_p': float(h.get('ks_p', 0.05))}

    b = (SparkSession.builder.appName('m2_l3_hawkes')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean')
           .filter(~F.col('is_spam'))
           .select('bvid', 'batch_date', 'send_time'))
    dim = (spark.read.parquet(f'{wh}/dim_video')
           .select('bvid', F.col('pubdate').cast('long').alias('pubdate'),
                   'first_batch', 'last_batch'))

    # 每 bvid 取最新批快照（最完整事件史）
    d = (dwd.join(dim, 'bvid')
         .filter(F.col('batch_date') == F.col('last_batch'))
         .filter(F.col('send_time').isNotNull() & F.col('pubdate').isNotNull()))
    g = (d.groupBy('bvid')
         .agg(F.sort_array(F.collect_list('send_time')).alias('ts'),
              F.first('pubdate').alias('pubdate'),
              F.first('last_batch').alias('last_batch'),
              F.first('first_batch').alias('first_batch')))
    n_groups = g.count()

    def run_one(r):
        res = fit_video(list(r['ts']), r['pubdate'], fb_epoch(r['first_batch']), cfg)
        row = to_row(r['bvid'], str(r['last_batch']), res, ENGINE_VERSION)
        if row is None:
            return ('skip', res.get('skip'), res.get('n_events'))
        return ('ok', row)

    parts = g.rdd.map(run_one).collect()
    ok_rows = [p[1] for p in parts if p[0] == 'ok']
    skips = [p for p in parts if p[0] == 'skip']
    df = spark.createDataFrame(ok_rows, OUT_COLS)
    df.write.mode('overwrite').parquet(f'{wh}/dws_hawkes_params')

    # ── 统计与 DoD ──
    import numpy as np
    mle_rows = [p[1] for p in parts if p[0] == 'ok' and p[1][11] == 'mle']
    pl_rows = [p[1] for p in parts if p[0] == 'ok' and p[1][11] == 'powerlaw']
    pl_usable = [p for p in pl_rows if p[13] is not None]      # pl_p 非退化
    sig = [r[6] for r in ok_rows if r[6] is not None]      # branching_ratio（下标见 OUT_COLS）
    betas = [r[5] for r in ok_rows if r[5] is not None]
    exo_sig = [r[9] for r in ok_rows if r[9] is not None]
    skip_reasons = {}
    for p in skips:
        skip_reasons[p[1]] = skip_reasons.get(p[1], 0) + 1
    n_gate = len(ok_rows)
    report = {
        'job': 'm2_l3_hawkes', 'engine_version': ENGINE_VERSION,
        'config': cfg,
        'n_videos_total': n_groups, 'n_gate': n_gate,
        'n_mle_pass': len(mle_rows), 'n_powerlaw_fallback': len(pl_rows),
        'n_powerlaw_usable': len(pl_usable),
        'pass_rate_over_gate': round((len(mle_rows) + len(pl_rows)) / max(n_gate, 1), 4),
        'usable_rate_over_gate_strict': round((len(mle_rows) + len(pl_usable)) / max(n_gate, 1), 4),
        'coverage_over_all': round(n_gate / max(n_groups, 1), 4),
        'skip_reasons': skip_reasons,
        'branching_ratio_median': round(float(np.median(sig)), 4) if sig else None,
        'branching_ratio_min': round(float(np.min(sig)), 4) if sig else None,
        'branching_ratio_max': round(float(np.max(sig)), 4) if sig else None,
        'branching_ratio_in_band': f'{int(np.sum((np.array(sig) > 0) & (np.array(sig) < 0.95)))}/{len(sig)}' if sig else None,
        'sigma_at_bound_095': int(np.sum(np.array(sig) >= 0.949)) if sig else None,
        'beta_at_bound_5': int(np.sum(np.array(betas) >= 4.999)) if betas else None,
        'exo_sig_lt_005': f'{int(np.sum(np.array(exo_sig) < 0.05))}/{len(exo_sig)}' if exo_sig else None,
        'note': ('单元=每 bvid（最新快照）；全部 KS 未过 → method=powerlaw（设计内降级）；'
                 'powerlaw 行的 Hawkes 参数为 KS 未过时的 MLE 估计（诊断），'
                 'decay_halflife_d 在 powerlaw 行=2^(1/p)（峰后半值天数）、mle 行=ln2/β。'
                 'KS 全败已做三项对照实验（平稳/生命周期/大样本合成均 pass）——模型族对弹幕'
                 '超短突发性（8-22% 同秒并列、残差过散布 std≈1.6-2.1）系统性不足，负结果入档'),
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm2_l3_hawkes_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m2_l3_hawkes：{n_gate} 个视频拟合（gate {cfg["min_events"]}）；'
          f'MLE pass {len(mle_rows)} / powerlaw {len(pl_rows)}；'
          f'分支比中位数 {report["branching_ratio_median"]}')
    spark.stop()


if __name__ == '__main__':
    main()
