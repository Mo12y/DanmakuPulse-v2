#!/usr/bin/env python3
"""
compare_canonical.py — 重跑一致性比对（幂等验收，host 运行，纯 stdlib）

    python3 tools/compare_canonical.py data-out/canonical data-out/canonical_run2

规则（与 parity_check.py 同）：整数列精确一致；浮点列 |diff| ≤ 1e-9。
不做 md5：Spark 聚合的浮点位级结果受 shuffle 顺序影响（同输入重跑差 ~1e-16 属正常），
幂等的验收口径 = 容差内一致 + 整数精确。
"""
import csv
import math
import sys

INT_COLS = {'peak_count', 'peak_fwhm', 'danmaku_count', 'bucket', 'n_videos'}
FLOAT_TOL = 1e-9


def load(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.reader(f))


def compare(fa, fb, name):
    a, b = load(fa), load(fb)
    if a[0] != b[0]:
        return f'❌ {name}: 表头不一致\n  {a[0]}\n  {b[0]}'
    if len(a) != len(b):
        return f'❌ {name}: 行数不一致 {len(a)-1} vs {len(b)-1}'
    header = a[0]
    int_idx = [i for i, c in enumerate(header) if c in INT_COLS]
    bad = 0
    max_d = 0.0
    for ra, rb in zip(a[1:], b[1:]):
        for i in range(len(header)):
            va, vb = ra[i], rb[i]
            if i in int_idx:
                if int(float(va)) != int(float(vb)):
                    bad += 1
            else:
                try:
                    d = abs(float(va) - float(vb))
                except ValueError:
                    if va != vb:
                        bad += 1
                    continue
                if d > FLOAT_TOL:
                    bad += 1
                max_d = max(max_d, d)
    if bad:
        return f'❌ {name}: {bad} 个字段超容差（浮点最大差 {max_d:.3e}）'
    return f'✅ {name}: {len(a)-1} 行一致（浮点最大差 {max_d:.3e} ≤ {FLOAT_TOL}）'


def main():
    d1, d2 = sys.argv[1], sys.argv[2]
    ok = True
    for name in ['metrics.csv', 'curve.csv']:
        r = compare(f'{d1}/{name}', f'{d2}/{name}', name)
        print(r)
        ok = ok and r.startswith('✅')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
