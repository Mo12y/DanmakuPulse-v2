#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp50 · Gate A 一致性计算（双人标注交回后运行）

输入：两位标注者各自独立交回的 ann_A.csv / ann_B.csv（协议 v1.0 §5 列冻结）
输出：{out}/gateA_results.json（含 Gate 通过判定）+ {out}/disagreements.csv（分歧清单）

口径（照抄协议 §6，不得在此处改）：
  - 边界 = 每段 start_bucket + 末段 end_bucket+1（视频右边界）
  - 匹配 = 双向贪心最近邻，|Δ| ≤ 1 桶（60s），1:1
  - precision = 命中/#预测边界；recall = 命中/#真值边界；F1 = 调和均值
  - **Gate 门槛 F1 ≥ 0.70**（预设候选门槛，非结果）
  - 区间估计：视频级 bootstrap（重采样视频后重算微观 F1），B=2000，种子固定
  - 类别一致率：按配对成功的段起点对齐后比较 category

用法：
  ~/danmaku-venv/bin/python experiments/exp50_gatea_consistency.py \
      --a data-out/c1/ann_A.csv --b data-out/c1/ann_B.csv \
      --sample30 data-out/c1/sample30_csv --out data-out/c1
"""
import argparse
import csv
import glob
import importlib.util
import json
import os

import numpy as np

REPO = '/home/molly/projects/danmaku-v2'
SPEC = importlib.util.spec_from_file_location('exp49', f'{REPO}/experiments/exp49_c1_gateb.py')
exp49 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp49)

SHEET_COLUMNS = ['bvid', 'seg_id', 'start_bucket', 'end_bucket', 'category', 'confidence', 'note']
CATEGORY_SET = {'L', 'R', 'H', 'B', 'D', 'X'}
CONF_SET = {'high', 'low'}
GATE_F1 = 0.70
SEED = 20261028
BOOT = 2000


def load_sheet(path):
    """读并校验标注表。返回 {bvid: [(s,e,category)]}；校验失败抛 ValueError。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'标注表不存在：{path}')
    with open(path, encoding='utf-8-sig', newline='') as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if header != SHEET_COLUMNS:
            raise ValueError(f'{path}: 列序必须是 {SHEET_COLUMNS}，实际 {header}')
        rows = list(rd)
    if not rows:
        raise ValueError(f'{path}: 没有任何标注行——标注未完成，不得运行本评估')
    out, errs = {}, []
    for i, r in enumerate(rows, start=2):
        if len(r) != len(SHEET_COLUMNS):
            errs.append(f'第{i}行 列数 {len(r)} != {len(SHEET_COLUMNS)}')
            continue
        bvid, seg, s, e, cat, conf, note = (x.strip() for x in r)
        if cat not in CATEGORY_SET:
            errs.append(f'第{i}行 category={cat} 不在六类内')
            continue
        if conf and conf not in CONF_SET:
            errs.append(f'第{i}行 confidence={conf} 非法')
        try:
            s, e = int(s), int(e)
        except ValueError:
            errs.append(f'第{i}行 桶序号非整数: {s},{e}')
            continue
        if e < s:
            errs.append(f'第{i}行 end_bucket({e}) < start_bucket({s})')
            continue
        out.setdefault(bvid, []).append((s, e, cat))
    if errs:
        raise ValueError(f'{path} 校验失败 {len(errs)} 处，前 5 条：' + '；'.join(errs[:5]))
    return out


def load_sample(path_glob):
    p = sorted(glob.glob(f'{path_glob}/*.csv'))
    rows = []
    for x in p:
        with open(x, encoding='utf-8-sig', newline='') as f:
            rows.extend(csv.DictReader(f))
    return {r['bvid']: int(r['n_buckets_expected']) for r in rows}


def run(a_path, b_path, sample_glob, out_dir, boot=BOOT, seed=SEED):
    A = load_sheet(a_path)
    B = load_sheet(b_path)
    n_buckets = load_sample(sample_glob)
    # 桶越界检查（协议 §1：边界必须落在视频桶域内）
    errs = []
    for tag, S in (('A', A), ('B', B)):
        for bv, segs in S.items():
            if bv not in n_buckets:
                errs.append(f'{tag}: bvid {bv} 不在 sample30')
                continue
            nb = n_buckets[bv]
            for s, e, _ in segs:
                if s < 0 or e >= nb:
                    errs.append(f'{tag}: {bv} 段 [{s},{e}] 超出桶域 [0,{nb})')
    if errs:
        raise ValueError('桶域校验失败：' + '；'.join(errs[:5]))

    truth = {bv: exp49.boundaries_of_segments([(s, e) for s, e, _ in segs], n_buckets[bv])
             for bv, segs in B.items()}
    pred = {bv: exp49.boundaries_of_segments([(s, e) for s, e, _ in segs], n_buckets[bv])
            for bv, segs in A.items()}
    res = exp49.evaluate(truth, pred, tol=1)

    # 类别一致率（按配对成功的段起点）
    cat_same = cat_n = 0
    disagreements = []
    for bv in sorted(set(A) & set(B)):
        bmap = {s: (s, e, c) for s, e, c in B[bv]}
        for sa, ea, ca in A[bv]:
            cand = [d for d in bmap if abs(d - sa) <= 1]
            if cand:
                d = min(cand, key=lambda x: abs(x - sa))
                sb, eb, cb = bmap[d]
                cat_n += 1
                if cb == ca:
                    cat_same += 1
                else:
                    disagreements.append(dict(bvid=bv, type='category',
                                              detail=f'A={ca}[{sa},{ea}] vs B={cb}[{sb},{eb}]'))
    # 边界分歧：未匹配的真值/预测边界
    for bv in sorted(set(truth) | set(pred)):
        tb, pb = set(truth.get(bv, [])), set(pred.get(bv, []))
        hit_t = {p for (t, p, d) in exp49.match_boundaries(truth.get(bv, []), pred.get(bv, []))['pairs']}
        for t in sorted(tb - {x[0] for x in exp49.match_boundaries(truth.get(bv, []), pred.get(bv, []))['pairs']}):
            disagreements.append(dict(bvid=bv, type='boundary',
                                      detail=f'真值边界 t={t} 无匹配预测（候选预测边界={sorted(pb)}）'))
    micro = res['micro']
    # 视频级 bootstrap CI（重采样视频，重算微观 F1）
    vids = sorted(set(truth) | set(pred))
    rng = np.random.default_rng(seed)
    f1s = []
    for _ in range(boot):
        pick = rng.choice(vids, len(vids), True)
        T = {v: truth.get(v, []) for v in pick}
        Pp = {v: pred.get(v, []) for v in pick}
        f1s.append(exp49.evaluate(T, Pp, tol=1)['micro']['f1'])
    ci = [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]

    gate = dict(metric='F1@60s', value=round(micro['f1'], 4), threshold=GATE_F1,
                verdict='PASS' if micro['f1'] >= GATE_F1 else 'FAIL',
                precision=round(micro['precision'], 4), recall=round(micro['recall'], 4),
                TP=micro['TP'], FP=micro['FP'], FN=micro['FN'],
                med_err_buckets=micro['med_err_buckets'],
                ci95=[round(ci[0], 4), round(ci[1], 4)], boot=boot, seed=seed,
                category_agreement=round(cat_same / cat_n, 4) if cat_n else None,
                n_videos=len(vids), n_truth_bounds=sum(len(v) for v in truth.values()),
                n_pred_bounds=sum(len(v) for v in pred.values()))
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/gateA_results.json', 'w', encoding='utf-8') as f:
        json.dump(dict(gate=gate, per_video=res['per_video'],
                       note='口径=docs/c1_annotation_protocol.md v1.0 §6；门槛是预设候选门槛，不是结果'), f,
                  ensure_ascii=False, indent=1)
    with open(f'{out_dir}/disagreements.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['bvid', 'type', 'detail'])
        w.writeheader()
        w.writerows(disagreements)
    print(json.dumps(gate, ensure_ascii=False, indent=1))
    print(f"GATE-A-{'PASS' if gate['verdict'] == 'PASS' else 'FAIL'}  "
          f"F1={gate['value']} CI95={gate['ci95']} 分歧 {len(disagreements)} 条")
    return 0 if gate['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--sample30', default=f'{REPO}/data-out/c1/sample30_csv')
    ap.add_argument('--out', default=f'{REPO}/data-out/c1')
    ap.add_argument('--boot', type=int, default=BOOT)
    args = ap.parse_args()
    sys_exit = run(args.a, args.b, args.sample30, args.out, boot=args.boot)
    raise SystemExit(sys_exit)
