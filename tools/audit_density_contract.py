#!/usr/bin/env python3
"""audit_density_contract.py — 密度表"稀疏存储+稠密化契约"消费方审计（证据固化，2026-10-13）

背景：dws_density_5s / dws_density_1pct 为**稀疏落盘**（缺行 ⇔ cnt=0，契约见 m2_l1_density 头注）。
凡"时间轴语义"消费（曲线、std/mean/峰检测、分段）**必须先稠密化**（域内缺行补 0），
否则静默剔除安静时段 → cv/峰位系统性失真。
本脚本对全部消费方做静态审计，输出：判定（✅遵守 / ❌违规 / ⚠️数学等价）+ 证据串。

用法：~/danmaku-venv/bin/python tools/audit_density_contract.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')

# (相对路径, 稠密化证据特征串, 说明, 判定)
CHECKS = [
    ('jobs/m2_l2_events.py', 'np.zeros(int(mx) + 1',
     'RDD 序列重建（mx+1 全零初始化后回填）', 'OK'),
    ('jobs/m2_l2_novelty.py', 'np.zeros(int(mx) + 1',
     '同上（分段检测前重建）', 'OK'),
    ('jobs/m2_l1_cluster.py', '[0.0] * N_BUCKETS',
     '曲线域填充（采集后按域补零）', 'OK'),
    ('jobs/m2_l2_lag.py', 'coalesce(F.col(\'s_cnt\'), F.lit(0))',
     'cal_cnt 位移源缺失补 0（保持稀疏存储一致）', 'OK'),
    ('jobs/m2_l1_fingerprint.py', 'F.sequence(F.lit(0)',
     'burst_cv 稠密化（2026-10-13 修复——原实现违反契约）', 'FIXED'),
    ('jobs/m2_l1_fingerprint.py', 'sum(F.col(\'cnt\') * F.log(F.col(\'cnt\')))',
     'rhythm_entropy：p=0 项贡献为 0 ⇒ 稀疏/稠密数学等价（分母 ln(100) 固定）', 'EQUIV'),
]


def main():
    print('═' * 72)
    print('密度稠密化契约 · 消费方审计（2026-10-13 快照）')
    print('═' * 72)
    all_ok = True
    for rel, needle, desc, verdict in CHECKS:
        p = os.path.join(ROOT, rel)
        s = io.open(p, encoding='utf-8').read()
        found = needle in s
        mark = {'OK': '✅', 'FIXED': '✅(修复)', 'EQUIV': '✅(等价)'}[verdict]
        status = mark if found else '❌ 证据缺失'
        if not found:
            all_ok = False
        print(f'{status}  {rel}')
        print(f'      {desc}')
        print(f'      证据串: {needle[:60]}')
    print('─' * 72)
    print('判定汇总:', '全部通过' if all_ok else '存在缺失，需人工复核')
    print('复跑此脚本即得最新审计；契约原文见 jobs/m2_l1_density.py 头注。')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
