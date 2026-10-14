"""fetch_times sidecar（视频级采集时刻）单测 —— 2026-10-14 收口批次 ⑤

验收命令（宿主 venv）：
  ~/danmaku-venv/bin/python -m pytest tests/test_fetch_times_sidecar.py -q
预期：2 passed（写→读回→合并；异常不外抛）。
"""
import csv
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_bili():
    spec = importlib.util.spec_from_file_location(
        'bili_under_test', os.path.join(ROOT, 'collect', 'bili.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _new_manifest(mod, tmp_path, batch='20990101'):
    csv_path = os.path.join(str(tmp_path), f'hotlist_top200_{batch}.csv')
    m = mod.BatchManifest(csv_path, batch, '2099-01-01T00:00:00', 300, 200)
    m.start_window()
    return m


def test_note_time_write_and_reload(tmp_path):
    mod = load_bili()
    m = _new_manifest(mod, tmp_path)
    m.note_time('BV1aaaaaaaaa', '2099-01-01T00:01:02')
    m.note_time('BV1bbbbbbbbb', '2099-01-01T00:03:04')
    p = m._times_path()
    assert os.path.isfile(p), 'sidecar 未写出'
    rows = {r['bvid']: r['fetched_at'] for r in csv.DictReader(open(p, encoding='utf-8'))}
    assert rows == {'BV1aaaaaaaaa': '2099-01-01T00:01:02',
                    'BV1bbbbbbbbb': '2099-01-01T00:03:04'}

    # 重载（模拟断点续采）：已有时刻保留，新增合并（幂等重写）
    m2 = _new_manifest(mod, tmp_path)
    assert m2._times.get('BV1aaaaaaaaa', ('', ''))[0] == '2099-01-01T00:01:02'
    m2.note_time('BV1cccccccccc', '2099-01-01T00:05:06')
    rows2 = {r['bvid']: r['fetched_at'] for r in csv.DictReader(open(p, encoding='utf-8'))}
    assert len(rows2) == 3 and rows2['BV1cccccccccc'] == '2099-01-01T00:05:06'


def test_note_time_never_raises(tmp_path):
    mod = load_bili()
    m = _new_manifest(mod, tmp_path)
    # 写入目标不可用场景：只应告警，不得抛出（采集完整性优先）
    m._times_path = lambda: '/proc/nonexistent_dir/fetch_times_x.csv'
    m.note_time('BV1yyyyyyyyy', '2099-01-01T00:00:00')  # 无异常即通过
