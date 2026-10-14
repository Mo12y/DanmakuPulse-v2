"""m4_track_common 单测 —— 2026-10-14 收口批次 ④

验收命令（宿主 venv）：
  ~/danmaku-venv/bin/python -m pytest tests/test_m4_track_common.py -q
预期：4 passed（轨迹排序剔除 / 几何增长 / 命名规则 / 度量半衰期数值复算）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jobs'))

from m4_track_common import (build_trajectory, geom_growth, metric_halflife,
                             name_types_v2, trajectory_features)


def test_build_trajectory_drops_inverted_obs():
    # 正常两观测（1013 在 1012 之后）→ 全保留
    normal = [{'batch': 20261012, 't': 1000.0, 'play': 1},
              {'batch': 20261013, 't': 2000.0, 'play': 2}]
    kept, dropped, assumed = build_trajectory(normal)
    assert len(kept) == 2 and dropped == 0 and not assumed

    # 倒挂（1012 第三窗口：1012 观测晚于 1013）→ 剔除早批晚采的观测
    inverted = [{'batch': 20261012, 't': 5000.0, 'play': 1},
                {'batch': 20261013, 't': 2000.0, 'play': 2}]
    kept2, dropped2, _ = build_trajectory(inverted)
    assert dropped2 == 1 and len(kept2) == 1 and kept2[0]['batch'] == 20261013

    # t 缺失 → 退化按批序、不剔除、标注 assumed
    kept3, dropped3, assumed3 = build_trajectory(
        [{'batch': 20261013, 't': None, 'play': 2},
         {'batch': 20261012, 't': None, 'play': 1}])
    assert [o['batch'] for o in kept3] == [20261012, 20261013]
    assert dropped3 == 0 and assumed3


def test_geom_growth():
    assert abs(geom_growth(100, 121, 2) - 0.1) < 1e-12   # 121=100*1.1^2
    assert geom_growth(0, 100, 3) is None
    assert geom_growth(100, 200, 0) is None
    assert geom_growth(None, 100, 3) is None


def test_trajectory_features_and_naming():
    obs = [{'batch': 20261012, 't': 0.0, 'play': 100, 'like': 10, 'reported': 50,
            'rank': 100, 'sat': 0.0},
           {'batch': 20261013, 't': 86400.0, 'play': 200, 'like': 25, 'reported': 75,
            'rank': 80, 'sat': 0.0}]
    fx = trajectory_features(obs)
    assert abs(fx['g_play'] - 1.0) < 1e-9 and fx['rank_delta'] == 20
    assert fx['n_obs'] == 2
    assert trajectory_features([obs[0]]) is None

    centers = [[0.5, 0, 0, 0, 0, 0, 0], [2.0, 0, 0, 0, 0, 0, 0], [1.0, 0, 0, 0, 0, 0, 0]]
    m = name_types_v2(centers)
    assert m == {1: '快增型', 2: '稳增型', 0: '缓增型'}


def test_metric_halflife_exact():
    # 日增率按 0.5, 0.25, 0.125 衰减（每步减半）→ τ 应恰为 1.0 天
    obs = [{'t': 0.0, 'play': 100.0}, {'t': 86400.0, 'play': 150.0},
           {'t': 172800.0, 'play': 187.5}, {'t': 259200.0, 'play': 210.9375}]
    tau, n = metric_halflife(obs, 'play')
    assert n == 3 and abs(tau - 1.0) < 1e-6
    # 不足 3 个 rate → (None, n)
    tau2, n2 = metric_halflife(obs[:3], 'play')
    assert tau2 is None and n2 == 2
