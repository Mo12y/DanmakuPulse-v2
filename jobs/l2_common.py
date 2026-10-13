#!/usr/bin/env python3
"""l2_common.py — L2 共享纯函数（m2_l2_events / m2_l2_novelty / tests 共用）

spark-submit 时须随 --py-files 分发（workers 反序列化闭包时按模块名 import）。
"""


def platform_peaks(c):
    """局部极大（平台取中点；端点 run 不算峰——scipy find_peaks 口径，沿用 m2_l1_min 实测）。

    c: 一维序列（list 或 np.ndarray）。返回峰下标列表（升序）。
    """
    idx = []
    n = len(c)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and c[j + 1] == c[i]:
            j += 1
        left_ok = i > 0 and c[i - 1] < c[i]
        right_ok = j < n - 1 and c[j] > c[j + 1]
        if left_ok and right_ok:
            idx.append((i + j) // 2)
        i = j + 1
    return idx