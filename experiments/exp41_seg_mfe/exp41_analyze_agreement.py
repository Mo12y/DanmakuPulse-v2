# -*- coding: utf-8 -*-
"""exp41 · 双人盲标一致性分析（预注册口径，标注完成后运行）
用法：~/danmaku-venv/bin/python analyze_agreement.py annotator_A.json annotator_B.json
预注册指标（对应任务40 Gate A 口径，探索性 5 支仅报告、不算正式验证）：
  - 边界：排除 0 与 duration 的段间边界；60s 容差贪心匹配（两人按时间就近配对，一对一只用一次）
  - 边界一致率 = 匹配数 / min(nA, nB)；同时报 匹配对时间差中位/最大
  - 段类别：两人区间 IoU>0.5 对齐后类别一致率（uncertain 对不判一致也不判分歧，单列）
  - 分歧分布：A 类别 × B 类别 计数矩阵（对齐对）
"""
import json
import sys

TOL = 60.0


def bounds(it):
    segs = it.get("segments", [])
    return sorted(s["start_s"] for s in segs[1:]) if len(segs) >= 2 else []


def greedy(a, b, tol):
    pairs, used = [], set()
    for x in a:
        best, bj = None, None
        for j, y in enumerate(b):
            if j in used:
                continue
            d = abs(x - y)
            if d <= tol and (best is None or d < best):
                best, bj = d, j
        if bj is not None:
            used.add(bj)
            pairs.append((x, b[bj], best))
    return pairs


def seg_align(A, B):
    out = []
    for sa in A:
        best, bs = 0.0, None
        for sb in B:
            lo, hi = max(sa["start_s"], sb["start_s"]), min(sa["end_s"], sb["end_s"])
            iou = max(0.0, hi - lo) / max(1e-9, min(sa["end_s"] - sa["start_s"], sb["end_s"] - sb["start_s"]))
            if iou > best:
                best, bs = iou, sb
        if bs is not None and best > 0.5:
            out.append((sa, bs))
    return out


def main(fa, fb):
    A = {it["bvid"]: it for it in json.load(open(fa, encoding="utf-8"))["items"]}
    B = {it["bvid"]: it for it in json.load(open(fb, encoding="utf-8"))["items"]}
    lines = ["# exp41 双人一致性（探索性 5 支）", ""]
    tot_m, tot_min, diffs, cat_ok, cat_bad, uncer, matrix = 0, 0, [], 0, 0, 0, {}
    for bv in sorted(set(A) & set(B)):
        ba, bb = bounds(A[bv]), bounds(B[bv])
        pairs = greedy(ba, bb, TOL)
        m, mn = len(pairs), max(1, min(len(ba), len(bb)))
        tot_m += m; tot_min += mn
        diffs += [p[2] for p in pairs]
        aligned = seg_align(A[bv].get("segments", []), B[bv].get("segments", []))
        ok = bad = un = 0
        for sa, sb in aligned:
            ca, cb = sa.get("category"), sb.get("category")
            if ca == "uncertain" or cb == "uncertain":
                un += 1
            elif ca == cb:
                ok += 1
            else:
                bad += 1
                key = f"{ca}→{cb}"
                matrix[key] = matrix.get(key, 0) + 1
        cat_ok += ok; cat_bad += bad; uncer += un
        lines.append(f"- {bv}: 边界 A={len(ba)} B={len(bb)} 匹配={m} (率 {m/mn:.2f})；"
                     f"时间差中位 {sorted(diffs)[-1] if diffs else 0:.0f}s 内；"
                     f"段对齐 {len(aligned)}：一致 {ok} / 分歧 {bad} / 含不确定 {un}"
                     f"{' ｜ 无明显分段 A=' + str(A[bv].get('no_obvious')) + ' B=' + str(B[bv].get('no_obvious')) if (A[bv].get('no_obvious') or B[bv].get('no_obvious')) else ''}")
    lines += ["", f"## 汇总", f"- 边界一致率（60s 容差，Gate A 参考口径）：**{tot_m}/{tot_min} = {tot_m/tot_min:.2f}**（≥0.70 为任务40 参考线）",
              f"- 匹配边界时间差：中位 {sorted(diffs)[len(diffs)//2] if diffs else 0:.0f}s，最大 {max(diffs) if diffs else 0:.0f}s",
              f"- 段类别一致 {cat_ok} / 分歧 {cat_bad} / 含不确定 {uncer}；"
              f"一致率（不含不确定）= {cat_ok/max(1,cat_ok+cat_bad):.2f}",
              f"- 分歧矩阵：{json.dumps(matrix, ensure_ascii=False)}"]
    txt = "\n".join(lines)
    print(txt)
    open("agreement_report.md", "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
