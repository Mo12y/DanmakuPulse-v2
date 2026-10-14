#!/usr/bin/env python3
"""
reconstruct_fetch_times.py — 从采集日志回溯「视频级采集时刻」sidecar（一次性工具，2026-10-14）

背景（收口批次 ⑤）：fetch_times sidecar 自 2026-10-14 起才由采集端写入；此前批次只有
meta 窗口级信息。本工具解析 logs/collect_*.log 的组头时间戳
（"-- 批次 N/M [HH:MM:SS] (i-j/NOT) --"，10 视频一组、组内逐个采集），
为每个视频回填**最后一次尝试**的组起始时刻（精度 ≈ ±2min，src 标注 log_group_reconstructed）。

日志 → 批次映射（2026-10-14 实测）：
  collect_20261012.log  → 20261012 窗口①（10 视频 @16:42:21）
  collect_resume3.log   → 20261012 窗口②（160 视频；18:20 → 次日 08:37，中间 VM 挂起）
  collect_20261013.log  → 段A 20261013（200 视频 @13:12:32→17:39:30）
                          段B 20261012 窗口③（45 视频 @17:48:50→18:44:03，晚于 1013 批 → 倒挂）

输出（幂等覆盖，可重跑）：
  data/cli_collected/fetch_times_<bd>.csv（bvid,fetched_at,src —— 与采集端 sidecar 同构）
  data-out/fetch_times_reconstruction_report.json（窗口计数 / 倒挂集 / 校验断言）

用法：~/danmaku-venv/bin/python tools/reconstruct_fetch_times.py
（随后 make stage → make ingest → make m0-daily 收编进 ods_video_fetch_times）
"""
import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUP_RE = re.compile(r'--\s*批次\s*\d+/\d+\s*\[(\d{2}):(\d{2}):(\d{2})\]')
VIDEO_RE = re.compile(r'\[(\d+)/\d+\]\s*(BV[0-9A-Za-z]{10})')
WRAP_RE = re.compile(r'^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*日频采集开始')
HEAD_RE = re.compile(r'B站热门榜弹幕采集 — (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
BD_RE = re.compile(r'批次日:\s*(\d{8})')
ARG_BD_RE = re.compile(r'--batch-date\s+(\d{8})')


def parse_sections(lines):
    """按「日频采集开始」wrapper 切段；无 wrapper 的整文件成一段（start 取自 HEAD_RE）。"""
    sections, cur = [], None
    for ln in lines:
        m = WRAP_RE.match(ln)
        if m:
            if cur is not None:
                sections.append(cur)
            cur = {'start': datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S'),
                   'head': ln, 'lines': []}
            continue
        if cur is None:
            cur = {'start': None, 'head': '', 'lines': []}
        cur['lines'].append(ln)
    if cur is not None:
        sections.append(cur)
    for s in sections:
        if s['start'] is None:
            for ln in s['lines'][:80]:
                m = HEAD_RE.search(ln)
                if m:
                    s['start'] = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                    break
    return [s for s in sections if s['start'] is not None]


def section_batch(s):
    for ln in [s['head']] + s['lines'][:80]:
        m = ARG_BD_RE.search(ln)
        if m:
            return m.group(1)
    for ln in [s['head']] + s['lines'][:80]:
        m = BD_RE.search(ln)
        if m:
            return m.group(1)
    return s['start'].strftime('%Y%m%d')


def extract_section(s):
    """返回 (attempts: list[datetime], last_by_bvid: dict[str, datetime], n_lines_video)。"""
    date = s['start'].date()
    last_hms = None
    cur_group = None
    attempts, last_by_bvid = [], {}
    for ln in s['lines']:
        g = GROUP_RE.search(ln)
        if g:
            hms = (int(g.group(1)), int(g.group(2)), int(g.group(3)))
            if last_hms is not None and hms < last_hms:
                date = date + timedelta(days=1)     # 跨天（含 VM 挂起后跳变）→ +1 天
            last_hms = hms
            cur_group = datetime.combine(date, datetime.min.time()).replace(
                hour=hms[0], minute=hms[1], second=hms[2])
            continue
        vids = VIDEO_RE.findall(ln)
        if vids and cur_group is not None:
            for bv in {b for _idx, b in vids}:     # 多P 行同 BV 出现两次 → 按行去重
                attempts.append(cur_group)
                last_by_bvid[bv] = cur_group
    return attempts, last_by_bvid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs='*', default=[
        'logs/collect_20261012.log', 'logs/collect_resume3.log', 'logs/collect_20261013.log'],
        help='按时间顺序的日志文件（相对仓库根）')
    args = ap.parse_args()

    all_last = {}          # batch -> {bvid: datetime}（跨文件取最大值=最后一次尝试）
    all_attempts = {}      # batch -> list[datetime]（全部尝试，用于窗口校验）
    section_info = []      # 供报告

    for rel in args.logs:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            print(f'❌ 日志缺失：{p}')
            sys.exit(2)
        with open(p, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        for s in parse_sections(lines):
            bd = section_batch(s)
            attempts, last = extract_section(s)
            all_attempts.setdefault(bd, []).extend(attempts)
            d = all_last.setdefault(bd, {})
            for bv, t in last.items():
                if bv not in d or t > d[bv]:
                    d[bv] = t
            section_info.append({
                'file': rel, 'batch': bd,
                'start': s['start'].isoformat(timespec='seconds'),
                'n_attempts': len(attempts), 'n_unique': len(last),
                'min_t': min(attempts).isoformat(timespec='seconds') if attempts else None,
                'max_t': max(attempts).isoformat(timespec='seconds') if attempts else None,
                'is_window3_candidate': (bd == '20261012' and '20261013' in rel),
            })

    # ── 校验（对照 meta：1012 attempted=215/ok=198/unique=200；1013 attempted=200）──
    checks, failures = {}, []
    def chk(name, cond, detail):
        checks[name] = {'ok': bool(cond), 'detail': detail}
        if not cond:
            failures.append(f'{name}: {detail}')

    n_1012 = len(all_last.get('20261012', {}))
    n_1013 = len(all_last.get('20261013', {}))
    chk('1012_unique_200', n_1012 == 200, f'{n_1012}')
    chk('1013_unique_200', n_1013 == 200, f'{n_1013}')
    chk('1012_attempts_215', len(all_attempts.get('20261012', [])) == 215,
        f'{len(all_attempts.get("20261012", []))}')
    chk('1013_attempts_200', len(all_attempts.get('20261013', [])) == 200,
        f'{len(all_attempts.get("20261013", []))}')

    # 窗口级校验（对照 meta windows：①10/②160/③45）与 1013（200）
    by_file = {}
    for si in section_info:
        by_file.setdefault((si['file'], si['batch']), 0)
        by_file[(si['file'], si['batch'])] += si['n_attempts']
    chk('w1_attempts_10', by_file.get(('logs/collect_20261012.log', '20261012'), -1) == 10,
        str(by_file.get(('logs/collect_20261012.log', '20261012'))))
    chk('w2_attempts_160', by_file.get(('logs/collect_resume3.log', '20261012'), -1) == 160,
        str(by_file.get(('logs/collect_resume3.log', '20261012'))))
    chk('w3_attempts_45', by_file.get(('logs/collect_20261013.log', '20261012'), -1) == 45,
        str(by_file.get(('logs/collect_20261013.log', '20261012'))))
    chk('b1013_attempts_200', by_file.get(('logs/collect_20261013.log', '20261013'), -1) == 200,
        str(by_file.get(('logs/collect_20261013.log', '20261013'))))

    # 窗口③（1012 最后一段，10-13 晚）：bvid 清单 + 与其 1013 观测的倒挂数
    v3 = set()
    for rel in args.logs:
        p = os.path.join(ROOT, rel)
        if '20261013' not in rel:
            continue
        with open(p, encoding='utf-8', errors='replace') as f:
            for s in parse_sections(f.readlines()):
                if section_batch(s) != '20261012':
                    continue
                _, last = extract_section(s)
                v3.update(last)
    w3_bvids = sorted(v3)
    chk('window3_45_unique', len(w3_bvids) == 45, f'{len(w3_bvids)}')

    # 倒挂集合：1012 时刻 > 其 1013 时刻（可配对的）
    d1012, d1013 = all_last.get('20261012', {}), all_last.get('20261013', {})
    inverted = sorted([bv for bv in w3_bvids
                       if bv in d1013 and d1012[bv] > d1013[bv]])
    chk('window3_inverted_ge0', True, f'win3∩1013={len([b for b in w3_bvids if b in d1013])}, '
                                     f'倒挂={len(inverted)}')

    # ── 输出：cli_collected/fetch_times_<bd>.csv（与采集端同构，幂等覆盖）──
    for bd, d in sorted(all_last.items()):
        out = os.path.join(ROOT, 'data', 'cli_collected', f'fetch_times_{bd}.csv')
        tmp = out + '.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(['bvid', 'fetched_at', 'src'])
            for bv in sorted(d):
                w.writerow([bv, d[bv].isoformat(timespec='seconds'),
                            'log_group_reconstructed'])
        os.replace(tmp, out)
        print(f'✅ {out}（{len(d)} 行）')

    report = {
        'tool': 'reconstruct_fetch_times', 'ts': datetime.now().isoformat(timespec='seconds'),
        'sections': section_info,
        'checks': checks, 'failures': failures,
        'window3_bvids': w3_bvids, 'window3_inverted_pairs': inverted,
        'precision_note': '时刻 = 组起始（10 视频/组），精度 ≈ ±2min；仅用于排序与 Δt 记账',
    }
    rp = os.path.join(ROOT, 'data-out', 'fetch_times_reconstruction_report.json')
    os.makedirs(os.path.dirname(rp), exist_ok=True)
    with open(rp, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ('checks', 'failures')},
                     ensure_ascii=False, indent=2))
    if failures:
        print('❌ 校验未全过，见报告')
        sys.exit(1)
    print('✅ 全部校验通过')


if __name__ == '__main__':
    main()
