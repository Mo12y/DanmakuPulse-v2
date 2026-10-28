#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_exp50_gatea.py — Gate A 一致性计算器的测试（合成数据，不触真实标注）"""
import csv
import importlib.util
import json
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e50 = _load("exp50_gatea", "experiments/exp50_gatea_consistency.py")
tools = _load("c1_tools", "tools/c1_gatea_tools.py")

HDR = e50.SHEET_COLUMNS


def _write_sheet(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(HDR)
        w.writerows(rows)


def _make_env(tmp_path, a_rows, b_rows):
    sdir = tmp_path / 'sample30_csv'
    sdir.mkdir()
    with open(sdir / 'part.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['bvid', 'n_buckets_expected'])
        for bv in ('BV1', 'BV2', 'BV3'):
            w.writerow([bv, 10])
    a, b = tmp_path / 'ann_A.csv', tmp_path / 'ann_B.csv'
    _write_sheet(a, a_rows)
    _write_sheet(b, b_rows)
    return str(a), str(b), str(tmp_path / 'sample30_csv'), str(tmp_path)


def test_load_sheet_rejects_bad_category(tmp_path):
    p = tmp_path / 'ann.csv'
    _write_sheet(p, [['BV1', '1', '0', '4', 'Z', 'high', '']])
    with pytest.raises(ValueError, match='不在六类内'):
        e50.load_sheet(str(p))


def test_load_sheet_rejects_end_lt_start(tmp_path):
    p = tmp_path / 'ann.csv'
    _write_sheet(p, [['BV1', '1', '4', '0', 'L', 'high', '']])
    with pytest.raises(ValueError, match='end_bucket'):
        e50.load_sheet(str(p))


def test_load_sheet_rejects_empty(tmp_path):
    p = tmp_path / 'ann.csv'
    _write_sheet(p, [])
    with pytest.raises(ValueError, match='标注未完成'):
        e50.load_sheet(str(p))


def test_run_perfect_agreement_is_pass(tmp_path):
    rows = [['BV1', '1', '0', '4', 'L', 'high', ''], ['BV1', '2', '5', '9', 'H', 'high', ''],
            ['BV2', '1', '0', '9', 'L', 'low', 'whole'],
            ['BV3', '1', '0', '3', 'B', 'high', ''], ['BV3', '2', '4', '9', 'D', 'high', '']]
    a, b, s30, out = _make_env(tmp_path, rows, [r[:] for r in rows])
    rc = e50.run(a, b, s30, out)
    g = json.load(open(f'{out}/gateA_results.json', encoding='utf-8'))['gate']
    assert g['value'] == 1.0 and g['verdict'] == 'PASS' and g['category_agreement'] == 1.0
    assert rc == 0


def test_run_disagreement_drops_f1_and_writes_disagreements(tmp_path):
    a = [['BV1', '1', '0', '4', 'L', 'high', ''], ['BV1', '2', '5', '9', 'H', 'high', ''],
         ['BV2', '1', '0', '9', 'L', 'low', ''], ['BV3', '1', '0', '9', 'L', 'high', '']]
    b = [['BV1', '1', '0', '5', 'L', 'high', ''],   # 边界差 1 桶（容差内）
         ['BV1', '2', '6', '9', 'H', 'high', ''],   # 边界差 1 桶（容差内）
         ['BV2', '1', '0', '9', 'H', 'low', ''],    # 类别分歧
         ['BV3', '1', '0', '4', 'L', 'high', '']]   # 边界差 5 桶（超出容差）
    a_p, b_p, s30, out = _make_env(tmp_path, a, b)
    e50.run(a_p, b_p, s30, out)
    g = json.load(open(f'{out}/gateA_results.json', encoding='utf-8'))['gate']
    assert 0 < g['value'] < 1.0
    dis = list(csv.DictReader(open(f'{out}/disagreements.csv', encoding='utf-8-sig')))
    assert any(d['type'] == 'category' for d in dis)
    assert any(d['type'] == 'boundary' for d in dis)


def test_run_rejects_out_of_range_bucket(tmp_path):
    a = [['BV1', '1', '0', '15', 'L', 'high', '']]      # 桶域只有 [0,10)
    b = [['BV1', '1', '0', '9', 'L', 'high', '']]
    a_p, b_p, s30, out = _make_env(tmp_path, a, b)
    with pytest.raises(ValueError, match='超出桶域'):
        e50.run(a_p, b_p, s30, out)


def test_tools_constants_and_sha():
    assert tools.SHEET_COLUMNS == e50.SHEET_COLUMNS
    assert 'algo' in 'algo_smoke_segments.csv'
    p = '/tmp/_sha_probe.txt'
    open(p, 'w').write('x')
    assert len(tools._sha256(p)) == 64
