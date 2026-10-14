#!/usr/bin/env python3
"""test_eval_l2_peaks.py — 21 号 4 表2 评估逻辑单测（IoU / 峰解析 / 贪心匹配 P-R-F / AP）。"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')

sys.argv = ['eval_l2_peaks', '--truth', 'none']
spec = importlib.util.spec_from_file_location('eval_l2_peaks', os.path.join(ROOT, 'tools', 'eval_l2_peaks.py'))
ev = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(ev)
except SystemExit:
    pass


def test_iou_exact_partial_disjoint():
    assert ev.iou((0, 10), (0, 10)) == 1.0
    assert abs(ev.iou((0, 10), (5, 15)) - 1.0 / 3) < 1e-12
    assert ev.iou((0, 10), (10, 20)) == 0.0
    assert ev.iou((0, 10), (100, 110)) == 0.0


def test_parse_peaks_formats():
    assert ev.parse_peaks('120-165; 890-940; 2100-2185') == [(120.0, 165.0), (890.0, 940.0), (2100.0, 2185.0)]
    assert ev.parse_peaks('120-165' + chr(0xFF1B) + '890-940') == [(120.0, 165.0), (890.0, 940.0)]
    assert ev.parse_peaks('') == []
    assert ev.parse_peaks('nan') == []
    assert ev.parse_peaks('10~20') == [(10.0, 20.0)]
    assert ev.parse_peaks('  120 - 165  ') == [(120.0, 165.0)]


def test_prf_perfect():
    truth = {('BV1', 'd'): [(100.0, 200.0), (500.0, 600.0)]}
    cand = {'D1': {('BV1', 'd'): [(100.0, 200.0, 1.0), (500.0, 600.0, 0.9)]}}
    P, R, F, TP, FP, FN = ev.prf(cand['D1'], truth, 0.5)
    assert (TP, FP, FN) == (2, 0, 0)
    assert P == R == F == 1.0


def test_prf_partial():
    truth = {('BV1', 'd'): [(100.0, 200.0), (500.0, 600.0)]}
    cand = {'D1': {('BV1', 'd'): [(100.0, 200.0, 1.0), (900.0, 950.0, 0.8)]}}
    P, R, F, TP, FP, FN = ev.prf(cand['D1'], truth, 0.5)
    assert (TP, FP, FN) == (1, 1, 1)
    assert abs(P - 0.5) < 1e-12 and abs(R - 0.5) < 1e-12


def test_prf_no_double_count():
    truth = {('BV1', 'd'): [(100.0, 200.0)]}
    cand = {'D1': {('BV1', 'd'): [(100.0, 200.0, 0.9), (105.0, 195.0, 0.8)]}}
    P, R, F, TP, FP, FN = ev.prf(cand['D1'], truth, 0.5)
    assert (TP, FP, FN) == (1, 1, 0)


def test_prf_missing_video_is_fn():
    truth = {('BV1', 'd'): [(1.0, 10.0)], ('BV2', 'd'): [(1.0, 10.0)]}
    cand = {'D1': {('BV1', 'd'): [(1.0, 10.0, 1.0)]}}
    P, R, F, TP, FP, FN = ev.prf(cand['D1'], truth, 0.5)
    assert (TP, FP, FN) == (1, 0, 1)
    assert abs(R - 0.5) < 1e-12


def test_iou_threshold_matters():
    truth = {('BV1', 'd'): [(0.0, 100.0)]}
    cand = {'D1': {('BV1', 'd'): [(50.0, 150.0, 1.0)]}}   # IoU = 50/150 = 1/3
    assert ev.prf(cand['D1'], truth, 0.3)[3] == 1
    assert ev.prf(cand['D1'], truth, 0.5)[3] == 0


def test_ap_perfect_and_empty():
    truth = {('BV1', 'd'): [(100.0, 200.0)]}
    cand = {'x': {('BV1', 'd'): [(100.0, 200.0, 1.0)]}}
    assert ev.ap(cand['x'], truth, [0.1, 0.3, 0.5, 0.7, 0.9]) == 1.0
    assert ev.ap({}, {('BV1', 'd'): [(1.0, 2.0)]}, [0.3, 0.5]) == 0.0
