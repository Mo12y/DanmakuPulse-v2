#!/usr/bin/env python3
"""
stage_raw_local.py — data/cli_collected/ 三件套 → data/raw-local/batch_date=*/ 暂存
（host 运行，纯 stdlib；ingest_raw.sh 从 raw-local 同步到 HDFS——批次的"映射唯一事实源"在本文件）

历史 4 批固定映射（2026-10-12 勘察结论，与 docs/05-新系统重构/STATUS.md 数据契约一致）：
  20260902/20260907：快照永久丢失 → 只暂存 danmaku.csv + meta.json
  20260914：快照 = 旧仓库 legacy hotlist_videos.json（fetched_at=2026-09-14T13:12 与 meta 一致）
  20260917：快照 = hotlist_videos_20260917.json
日频新批次：自动发现（CSV + hotlist_videos_<date>.json 齐全才算完整批次）
幂等：覆盖复制。
"""
import csv
import glob
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'data', 'cli_collected')
DST = os.path.join(ROOT, 'data', 'raw-local')

HISTORICAL = {
    '20260902': {'csv': 'hotlist_top100_20260902.csv', 'snapshot': None},
    '20260907': {'csv': 'hotlist_top200_20260907.csv', 'snapshot': None},
    '20260914': {'csv': 'hotlist_top200_20260914.csv', 'snapshot': 'hotlist_videos.json'},
    '20260917': {'csv': 'hotlist_top100_20260917.csv', 'snapshot': 'hotlist_videos_20260917.json'},
}


def stage(bd, csv_name, snapshot_name):
    d = os.path.join(DST, f'batch_date={bd}')
    os.makedirs(d, exist_ok=True)
    shutil.copy2(os.path.join(SRC, csv_name), os.path.join(d, 'danmaku.csv'))
    shutil.copy2(os.path.join(SRC, csv_name + '.meta.json'), os.path.join(d, 'meta.json'))
    if snapshot_name:
        shutil.copy2(os.path.join(SRC, snapshot_name), os.path.join(d, 'hotlist.json'))
    else:
        # 明确不存在（占位，防 ingest 侧误判缺失文件为事故）
        marker = os.path.join(d, 'SNAPSHOT_LOST')
        open(marker, 'w').write('snapshot permanently lost (see config/batch_lineage.json)\n')
    print(f'  [ok] {bd}: danmaku.csv + meta.json'
          + (f' + {snapshot_name}' if snapshot_name else '（快照缺失，占位标记）'))


def main():
    assert os.path.isdir(SRC), f'采集目录不存在：{SRC}（先 make seed-data / 采集）'
    staged = set()

    for bd, spec in HISTORICAL.items():
        assert os.path.isfile(os.path.join(SRC, spec['csv'])), f'历史批次缺失：{spec["csv"]}'
        stage(bd, spec['csv'], spec['snapshot'])
        staged.add(bd)

    # 日频新批次自动发现
    for f in sorted(glob.glob(os.path.join(SRC, 'hotlist_top*.csv'))):
        m = re.search(r'(\d{8})', os.path.basename(f))
        if not m or m.group(1) in staged:
            continue
        bd = m.group(1)
        snap = f'hotlist_videos_{bd}.json'
        if os.path.isfile(os.path.join(SRC, snap)):
            stage(bd, os.path.basename(f), snap)
        else:
            print(f'  [skip] {bd}: 快照缺失（采集可能中断），不入 raw')

    print('✅ raw-local 暂存完成')


if __name__ == '__main__':
    main()
