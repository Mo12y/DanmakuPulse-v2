#!/usr/bin/env python3
"""figure_perf_report.py — B5.1 性能报告图表（3 张，报告级）

数据源：data-out/perf_results.jsonl（本轮干净网格 12 条）
输出：<out-dir>/fig1_density.png / fig2_events.png / fig3_hawkes_skew.png
运行：~/danmaku-venv/bin/python tools/figure_perf_report.py --out-dir ~/projects/danmaku/docs/05-新系统重构/assets
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

C1, C2, C3 = '#2E5E8C', '#C0392B', '#27AE60'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='data-out/perf_results.jsonl')
    ap.add_argument('--out-dir', default='/home/molly/projects/danmaku/docs/05-新系统重构/assets')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(args.results, encoding='utf-8')]

    den, evt, haw, skew = {}, {}, [], {}
    for r in rows:
        if r['mode'] == 'density':
            den.setdefault(r['amp'], {})[r['par']] = r
        elif r['mode'] == 'events':
            evt[r['amp']] = r
        elif r['mode'] == 'hawkes':
            haw.append(r)
        elif r['mode'] == 'skew':
            skew[r.get('amp', 1)] = r

    # ── fig1：density（成本曲线 + 加速比）──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    amps = sorted(den)
    p1 = [den[a].get('1', {}).get('wall_s') for a in amps]
    p2 = [den[a].get('2', {}).get('wall_s') for a in amps]
    x = np.arange(len(amps))
    ax1.bar(x - 0.18, p1, 0.36, label='par=1', color=C1)
    ax1.bar(x + 0.18, p2, 0.36, label='par=2', color=C3)
    for xi, (a, b) in enumerate(zip(p1, p2)):
        ax1.text(xi - 0.18, a + 0.4, f'{a}', ha='center', fontsize=9)
        ax1.text(xi + 0.18, b + 0.4, f'{b}', ha='center', fontsize=9)
    ax1.set_xticks(x, [f'{a}×' for a in amps])
    ax1.set_xlabel('放大倍率（合成放大）')
    ax1.set_ylabel('wall 时间（秒）')
    ax1.set_title('密度引擎：成本曲线（local[1] vs local[2]）')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    sp = [p1i / p2i for p1i, p2i in zip(p1, p2)]
    ax2.bar([f'{a}×' for a in amps], sp, 0.5, color=C2)
    for xi, v in enumerate(sp):
        ax2.text(xi, v + 0.02, f'{v:.2f}×', ha='center', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 2.2)
    ax2.axhline(2.0, ls='--', lw=1, color='gray', alpha=0.6)
    ax2.text(len(amps) - 1, 2.04, '理论上限 2×', ha='right', fontsize=8, color='gray')
    ax2.set_xlabel('放大倍率')
    ax2.set_ylabel('加速比（par 1→2）')
    ax2.set_title('密度引擎：并行加速比')
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig1_density.png'), dpi=150)
    plt.close(fig)

    # ── fig2：events 规模扩展 ──
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    amps_e = sorted(evt)
    walls = [evt[a]['wall_s'] for a in amps_e]
    nevs = [evt[a]['n_events'] for a in amps_e]
    xs = np.arange(len(amps_e))
    ax1.bar(xs, walls, 0.45, color=C1, label='wall 时间（s）')
    ax1.set_xticks(xs, [f'{a}×' for a in amps_e])
    ax1.set_xlabel('放大倍率（合成放大）')
    ax1.set_ylabel('wall 时间（秒）', color=C1)
    for xi, v in enumerate(walls):
        ax1.text(xi, v + 0.2, f'{v}s', ha='center', fontsize=9, color=C1)
    ax2 = ax1.twinx()
    ax2.plot(xs, nevs, 'o-', color=C2, lw=2, label='检测事件数')
    for xi, v in enumerate(nevs):
        ax2.text(xi, v * 1.04, f'{v:,}', ha='center', fontsize=9, color=C2)
    ax2.set_ylabel('检测事件数', color=C2)
    ax1.set_title('事件检测：规模扩展（par=2；合成放大）')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig2_events.png'), dpi=150)
    plt.close(fig)

    # ── fig3：Hawkes 加速比 + skew 对照 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ['首测\n(18:29)', '复测 #1', '复测 #2']
    vals = [1.38] + [h['speedup'] for h in haw[:2]]
    colors = ['#95A5A6', C3, C3]
    bars = ax1.bar(labels, vals, 0.5, color=colors)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.02, f'{v:.2f}×',
                 ha='center', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 2.2)
    ax1.axhline(2.0, ls='--', lw=1, color='gray', alpha=0.6)
    ax1.text(2, 2.04, '理论上限 2×', ha='right', fontsize=8, color='gray')
    ax1.set_ylabel('加速比（serial→parallel，par=2）')
    ax1.set_title('Hawkes 分组并行：加速比（803 组）')
    ax1.grid(axis='y', alpha=0.3)
    # skew 对照
    sk = skew.get(1) or list(skew.values())[0]
    ax2.bar(['普通分区\n(plain)', '加盐分区\n(salted)'],
            [sk['plain_skew'], sk['salted_skew']], 0.5, color=[C1, C3])
    for xi, v in enumerate([sk['plain_skew'], sk['salted_skew']]):
        ax2.text(xi, v + 0.01, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 1.6)
    ax2.axhline(1.0, ls='--', lw=1, color='gray', alpha=0.6)
    ax2.text(1, 1.02, '完全均匀 = 1.0', ha='right', fontsize=8, color='gray')
    ax2.set_ylabel('分区倾斜度（max/中位）')
    ax2.set_title('倾斜对照：加盐前后无差异')
    ax2.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, 'fig3_hawkes_skew.png'), dpi=150)
    plt.close(fig)

    print('saved: fig1_density.png / fig2_events.png / fig3_hawkes_skew.png ->', args.out_dir)


if __name__ == '__main__':
    main()
