#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""c1_gatea_tools.py — Gate A 标注交付的核验与分发工具（只读核验 + 生成独立副本）

子命令：
  verify    对 data-out/c1/ 做交付核验（V1–V10，全部通过才允许分发）
  dispatch  生成 annotator_A / annotator_B 两个互相独立的分发副本（含 sha256 清单）
            —— 算法产物（algo_smoke_segments.csv 等）物理隔离在 data-out/c1_internal/，
               分发包不包含；一方文件与另一方无任何交集路径

纪律：本工具不改协议、不生成任何标签、不评估一致性（评估在 exp50_gatea_consistency.py）。
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sys

REPO = '/home/molly/projects/danmaku-v2'
C1 = f'{REPO}/data-out/c1'
INTERNAL = f'{REPO}/data-out/c1_internal'
PROTOCOL = f'{REPO}/docs/c1_annotation_protocol.md'

SHEET_COLUMNS = ['bvid', 'seg_id', 'start_bucket', 'end_bucket', 'category', 'confidence', 'note']
CATEGORY_SET = {'L', 'R', 'H', 'B', 'D', 'X'}
CONF_SET = {'high', 'low'}
CURVE_WHITELIST = {'bvid', 'bucket60', 't_start_s', 't_end_s', 'cnt', 'wcnt', 'bucket_state'}
BUCKET_STATES = {'observed_nonzero', 'observed_zero', 'post_snapshot', 'unknown_fetch'}
SAMPLE_COLUMNS = {'bvid', 'batch_date', 'tname', 'duration', 'danmaku_got', 'n_buckets_expected',
                  'n_buckets_nonzero', 'n_zero_observed', 'n_post_snapshot', 'n_interior_holes', 'video_url'}


def _read_one_csv(dirpath):
    parts = sorted(glob.glob(f'{dirpath}/*.csv'))
    if not parts:
        return None
    rows = []
    for p in parts:
        with open(p, encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f))
    return rows


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def read_sample30():
    rows = _read_one_csv(f'{C1}/sample30_csv')
    for r in rows:
        for k in ('duration', 'danmaku_got', 'n_buckets_expected', 'n_buckets_nonzero',
                  'n_zero_observed', 'n_post_snapshot', 'n_interior_holes'):
            r[k] = int(r[k])
    return rows


def read_curves():
    rows = _read_one_csv(f'{C1}/curves_csv')
    for r in rows:
        r['bucket60'] = int(r['bucket60'])
        r['cnt'] = int(float(r['cnt']))
    return rows


def read_sheet(path):
    """读标注表；返回 (rows, header)。文件不存在 → (None, None)。"""
    if not os.path.exists(path):
        return None, None
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f)
        header = next(rd, None)
        rows = [r for r in rd]
    return rows, header


def verify():
    checks = []

    def chk(cid, ok, detail=''):
        checks.append(dict(id=cid, ok=bool(ok), detail=detail))

    # V1 样本清单
    s30 = read_sample30()
    chk('V1_sample30_rows30', s30 is not None and len(s30) == 30, f'n={0 if not s30 else len(s30)}')
    bvids = [r['bvid'] for r in s30]
    chk('V1_sample30_bvid_unique', len(set(bvids)) == len(bvids), f'unique={len(set(bvids))}')
    chk('V1_sample30_columns', set(s30[0].keys()) >= (SAMPLE_COLUMNS - {'batch_date'}),
        f'cols={len(s30[0])}')

    # V2 曲线与样本一一对应
    curves = read_curves()
    cbv = sorted(set(r['bvid'] for r in curves))
    chk('V2_curve_covers_sample30', cbv == sorted(bvids),
        f'curves={len(cbv)} sample={len(set(bvids))} diff={set(cbv) ^ set(bvids) or "无"}')

    # V3 时间轴/桶数/连续性/质量标记
    by_bv = {}
    for r in curves:
        by_bv.setdefault(r['bvid'], []).append(r)
    bad_n, bad_cont, bad_state, bad_t0 = [], [], [], []
    sby = {r['bvid']: r for r in s30}
    for bv, rs in by_bv.items():
        rs.sort(key=lambda x: x['bucket60'])
        n_exp = sby[bv]['n_buckets_expected']
        if len(rs) != n_exp:
            bad_n.append((bv, len(rs), n_exp))
        if [x['bucket60'] for x in rs] != list(range(len(rs))):
            bad_cont.append(bv)
        if any(x['bucket_state'] not in BUCKET_STATES for x in rs):
            bad_state.append(bv)
        if any(int(x['t_start_s']) != 60 * int(x['bucket60']) for x in rs):
            bad_t0.append(bv)
    chk('V3_bucket_count_matches_expected', not bad_n, f'不一致={bad_n[:5]}')
    chk('V3_bucket_index_contiguous', not bad_cont, f'异常={bad_cont[:5]}')
    chk('V3_bucket_state_enum', not bad_state, f'非法值={bad_state[:5]}')
    chk('V3_t_start_equals_60x', not bad_t0, f'异常={bad_t0[:5]}')

    # V4 曲线质量标记与 sample30 汇总一致（零桶/采集后/内部空洞可重建）
    bad_q = []
    for bv, rs in by_bv.items():
        n_zero = sum(1 for x in rs if x['bucket_state'] == 'observed_zero')
        n_post = sum(1 for x in rs if x['bucket_state'] == 'post_snapshot')
        if n_zero != sby[bv]['n_zero_observed'] or n_post != sby[bv]['n_post_snapshot']:
            bad_q.append((bv, n_zero, sby[bv]['n_zero_observed'], n_post, sby[bv]['n_post_snapshot']))
    chk('V4_quality_flags_consistent', not bad_q, f'不一致={bad_q[:4]}')

    # V5 曲线列白名单（防算法列混入）
    chk('V5_curve_columns_whitelist', set(curves[0].keys()) == CURVE_WHITELIST,
        f'cols={sorted(curves[0].keys())}')

    # V6 pool ↔ sample 对应
    pool = _read_one_csv(f'{C1}/c1_pool_csv')
    pin = {r['bvid']: r for r in pool if r['included'] == 'true'}
    missing = [b for b in bvids if b not in pin]
    chk('V6_sample_in_pool_included', not missing, f'缺失={missing[:5]}')

    # V7 双人模板：存在、列=协议、内容为空、互相独立（A/B 无共享数据行）
    for who in ('A', 'B'):
        rows, header = read_sheet(f'{C1}/ann_{who}.csv')
        chk(f'V7_ann{who}_exists_header', header == SHEET_COLUMNS, f'header={header}')
        chk(f'V7_ann{who}_empty', rows == [], f'rows={0 if rows is None else len(rows)}')
    a, _ = read_sheet(f'{C1}/ann_A.csv')
    b, _ = read_sheet(f'{C1}/ann_B.csv')
    chk('V7_annAB_separate_files', os.path.exists(f'{C1}/ann_A.csv') and os.path.exists(f'{C1}/ann_B.csv')
        and not os.path.samefile(f'{C1}/ann_A.csv', f'{C1}/ann_B.csv'), '两个独立文件')

    # V8 算法产物隔离：c1/ 下不得有算法输出；内部区必须有
    leak = [p for p in glob.glob(f'{C1}/**', recursive=True)
            if os.path.isfile(p) and any(k in os.path.basename(p).lower()
                                         for k in ('algo', 'smoke', 'segment', 'gateb'))]
    chk('V8_no_algo_output_in_c1', not leak, f'泄露={leak[:3]}')
    chk('V8_algo_quarantined', os.path.exists(f'{INTERNAL}/algo_smoke_segments.csv'),
        INTERNAL)

    # V9 协议冻结与门槛
    txt = open(PROTOCOL, encoding='utf-8').read()
    chk('V9_protocol_v1_frozen', 'v1.0' in txt and '冻结' in txt, PROTOCOL)
    chk('V9_protocol_disclosure', '利益冲突' in txt and '诚实披露' in txt)
    chk('V9_protocol_gates', '0.70' in txt and '15s' in txt and '60%' in txt)

    # V10 身份与利益冲突登记（登记表存在；第二标注者行留空待填 = 不伪造）
    reg_p = f'{C1}/annotators.csv'
    reg_ok = os.path.exists(reg_p)
    if reg_ok:
        rows = list(csv.DictReader(open(reg_p, encoding='utf-8-sig')))
        roles = {r['role']: r for r in rows}
        reg_ok = ('A' in roles and 'B' in roles
                  and '看过' in roles['A'].get('conflict_note', '')
                  and roles['B'].get('name', '').strip() == ''
                  and roles['B'].get('conflict_note', '').strip() == '')
    chk('V10_annotators_registry', reg_ok,
        'A 已披露冲突、B 留空待本人填写（不代填、不伪造）' if reg_ok else '登记表缺失或字段不符')

    rep = dict(generated_by='tools/c1_gateA_verify.py',
               n_checks=len(checks),
               n_pass=sum(1 for c in checks if c['ok']),
               n_fail=sum(1 for c in checks if not c['ok']),
               checks=checks)
    os.makedirs(f'{C1}', exist_ok=True)
    with open(f'{C1}/gateA_verify_report.json', 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    for c in checks:
        print(f"[{'ok' if c['ok'] else '!!'}] {c['id']}  {c['detail']}")
    print(f"\nVERIFY-{'OK' if rep['n_fail'] == 0 else 'FAIL'} "
          f"pass={rep['n_pass']}/{rep['n_checks']}")
    return 0 if rep['n_fail'] == 0 else 1


INSTRUCTIONS = """# C1 Gate A 标注操作说明（一页）

协议：docs/c1_annotation_protocol.md v1.0（随包附一份只读副本；**不得修改协议**）。

## 你要做的
1. 先看 2 支练习视频对齐标准（练习视频不在本包 30 支内，由组织者另发）。
2. 逐支打开 `sample30.csv` 里的 `video_url` **完整观看**（正常速度，不跳播）。
3. 回看 `curves.csv` 中该 bvid 的 60s 曲线（辅助；**冲突时以观感为准**，note 记 curve-mismatch）。
4. 按协议 §2–§4 标阶段：每段一行写进 **只属于你的** `ann_{A|B}.csv`。

## 提交格式（列序固定，UTF-8-SIG）
`bvid, seg_id, start_bucket, end_bucket, category, confidence, note`
- `start_bucket`/`end_bucket`：0 起始的分钟桶序号，`end_bucket` 为闭区间最后一桶；
- `category ∈ {L,R,H,B,D,X}`（协议 §3）；`confidence ∈ {high,low}`；`note ≤ 10 字`；
- 每视频至少 1 段；整片无变化就标 1 段；`X` 类用于 post_snapshot/unknown_fetch 区间。

## 纪律（违反即数据作废）
- **独立标注**：不得看另一位标注者的文件、进度或结论；
- **不看任何算法输出**：本包没有也不该有算法分段文件（`algo_smoke_segments.csv` 在隔离区）；
- 不要修改协议、不要修改 `curves.csv` / `sample30.csv`；
- 遇到协议未覆盖的情形：照常标注 + note 记 `edge-case`，并记入你随包交回的 `issues.txt`。

## 文件校验
包内 `MANIFEST.sha256` 列出每个文件的校验和。收到包后与交回前各跑一次：
```
sha256sum -c MANIFEST.sha256
```
任何 mismatch 说明文件被改动，需在 `issues.txt` 说明。

## 交回
交回且仅交回：你的 `ann_{A|B}.csv`（原始，勿重命名）+ `issues.txt`（可空）。
"""

DISPATCH_README = """# 分发副本说明

本目录是给标注者的独立副本。生成自 tools/c1_gateA_dispatch.py：
- `sample30.csv` / `curves.csv` 两位标注者内容一致（同一批数据）；
- `ann_{A|B}.csv` 各自独立，路径互不相同，一方无法覆盖另一方；
- `MANIFEST.sha256` 供 sha256sum -c 校验；
- **不含**：算法分段（隔离于 data-out/c1_internal/）、另一位标注者的文件、评估脚本输出。
"""


def dispatch():
    made = []
    for who in ('A', 'B'):
        dst = f'{C1}/dispatch/annotator_{who}'
        os.makedirs(dst, exist_ok=True)
        # 数据文件（两人一致）
        for src in (f'{C1}/sample30_csv', f'{C1}/curves_csv'):
            name = os.path.basename(src).replace('_csv', '.csv')
            rows = _read_one_csv(src)
            cols = sorted(rows[0].keys())
            with open(f'{dst}/{name}', 'w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        # 自己的标注表（独立路径 + 独立文件）
        shutil.copyfile(f'{C1}/ann_{who}.csv', f'{dst}/ann_{who}.csv')
        # 协议只读副本
        shutil.copyfile(PROTOCOL, f'{dst}/c1_annotation_protocol_v1.0.md')
        # 一页说明
        with open(f'{dst}/INSTRUCTIONS.md', 'w', encoding='utf-8') as f:
            f.write(INSTRUCTIONS.replace('{A|B}', who).replace('ann_{A|B}', f'ann_{who}'))
        with open(f'{dst}/README.txt', 'w', encoding='utf-8') as f:
            f.write(DISPATCH_README)
        # sha256 清单（排除清单自身）
        lines = []
        for p in sorted(glob.glob(f'{dst}/*')):
            if p.endswith('MANIFEST.sha256'):
                continue
            lines.append(f"{_sha256(p)}  {os.path.basename(p)}")
        with open(f'{dst}/MANIFEST.sha256', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        made.append(dst)
        print(f"[ok] 分发副本 {dst}: " + ", ".join(os.path.basename(x) for x in sorted(glob.glob(f'{dst}/*'))))
    # 隔离自检：分发包内不得有算法产物
    leak = [p for p in glob.glob(f'{C1}/dispatch/**', recursive=True)
            if os.path.isfile(p) and ('algo' in os.path.basename(p).lower() or 'smoke' in os.path.basename(p).lower())]
    print(f"[{'ok' if not leak else '!!'}] 分发包算法泄露检查: {'无' if not leak else leak}")
    return 0 if not leak else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['verify', 'dispatch'])
    args = ap.parse_args()
    sys.exit(verify() if args.cmd == 'verify' else dispatch())
