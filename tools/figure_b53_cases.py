#!/usr/bin/env python3
"""figure_b53_cases.py — B5.3 价值案例图：3 视频「系统高能段 vs 观众锚词弹幕」对照

数据源：data-out/cases_b53.json
输出：<out>/fig_b53_events_vs_anchors.png（3 面板时间线）
说明：锚词弹幕 = M5 判定的"预警型（前方高能）/ 回溯型（名场面等）"弹幕——
      作为"观众共识高能点"的**代理信号**（非人工标注；人工确认列文档末尾）。
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

C_EVENT = '#C0392B'
C_ANC_F = '#2E5E8C'   # 前向（预警）
C_ANC_A = '#27AE60'   # 后向（回溯）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='data-out/cases_b53.json')
    ap.add_argument('--out-dir', default='/home/molly/projects/danmaku/docs/05-新系统重构/assets')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = json.load(open(args.cases, encoding='utf-8'))

    vids = data['videos']
    fig, axes = plt.subplots(len(vids), 1, figsize=(12, 3.4 * len(vids)))
    for ax, v in zip(axes, vids):
        dur = v['duration'] or 1
        lb = v['last_batch']
        dens = sorted([(r['bucket5s'] * 5 / dur * 100, r['wcnt'] or 0)
                       for r in v['density'] if r['batch_date'] == lb])
        evs = [e for e in v['events'] if e['batch_date'] == lb]
        ancs = [a for a in v['anchors'] if a['batch_date'] == lb]
        if not dens:
            dens = sorted([(r['bucket5s'] * 5 / dur * 100, r['wcnt'] or 0) for r in v['density']])
        xs = [p[0] for p in dens]
        ys = [p[1] for p in dens]
        ax.plot(xs, ys, lw=1.2, color='#555555', label='弹幕密度（wcnt，5s 桶）')
        ymax = max(ys) if ys else 1
        # 系统事件带
        for i, e in enumerate(evs):
            s = e['start_sec'] / dur * 100
            t = e['end_sec'] / dur * 100
            ax.axvspan(s, t, alpha=0.22, color=C_EVENT)
            ax.text((s + t) / 2, ymax * 1.02, f"E{e['event_id']+1}\n{e['score_fused']:.2f}",
                    ha='center', fontsize=7.5, color=C_EVENT)
        # 锚词弹幕 rug（上方两行）
        n_f = sum(1 for a in ancs if a['anchor_type'] == 'fore')
        n_a = sum(1 for a in ancs if a['anchor_type'] == 'aft')
        for a in ancs:
            yy = ymax * (1.14 if a['anchor_type'] == 'fore' else 1.22)
            ax.plot([a['offset_pct'] * 100], [yy], marker='|', ms=8, mew=1.4,
                    color=C_ANC_F if a['anchor_type'] == 'fore' else C_ANC_A)
        ax.set_ylim(0, ymax * 1.3)
        ax.set_xlim(0, 100)
        ax.set_title(f"{v['title'][:26]}…（{v['tname']}｜{v['up_name']}）  "
                     f"事件 ×{len(evs)}｜锚词：预警 {n_f} / 回溯 {n_a}（批 {lb}）", fontsize=10)
        ax.set_ylabel('密度')
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel('视频进度（%）')
    # 图例（用最后一格）
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color='#555555', lw=1.2, label='弹幕密度'),
               Line2D([], [], color=C_EVENT, alpha=0.4, lw=8, label='系统检测高能段'),
               Line2D([], [], marker='|', color=C_ANC_F, lw=0, ms=9, label='锚词-预警（前方高能）'),
               Line2D([], [], marker='|', color=C_ANC_A, lw=0, ms=9, label='锚词-回溯（名场面等）')]
    axes[0].legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(args.out_dir, 'fig_b53_events_vs_anchors.png')
    fig.savefig(out, dpi=150)
    print('saved:', out)


if __name__ == '__main__':
    main()
