# -*- coding: utf-8 -*-
'''标注者 A（泊松规则）vs 标注者 B（多模态模型看图）一致性'''
import csv, collections
A = collections.defaultdict(list)
for r in csv.DictReader(open('data-out/annotator_A_poisson.csv', encoding='utf-8-sig')):
    A[(r['bvid'], r['batch_date'])].append(float(r['peak_sec']))
# B：看图视觉判断（秒）——数据文件 tools/annotator_B_visual.csv（2026-10-15 去硬编码，复核 F5）
# No.15/No.20 只看了第 1 屏，故限定在该范围内比较（viewed_upto_sec 列）
B = {}
for r in csv.DictReader(open('data-out/annotator_B_visual.csv', encoding='utf-8-sig')):
    bv = r['bvid']
    if bv not in B:
        B[bv] = [[], None]
    B[bv][0].append(int(float(r['peak_sec'])))
    vu = (r['viewed_upto_sec'] or '').strip()
    if vu:
        B[bv][1] = float(vu)
B = {k: (v[0], v[1]) for k, v in B.items()}
batch = {}
for k in A:
    if k[0] in B: batch[k[0]] = k
TOL = 15.0   # ±15 秒内算匹配
print('%-14s %6s %6s %8s %8s %8s' % ('bvid', 'A峰数', 'B峰数', 'A命中B', 'B命中A', '并集'))
tot = collections.Counter()
for bv, (bpeaks, rng) in B.items():
    k = batch.get(bv)
    if not k: print(bv, 'A 中无此视频'); continue
    a = [x for x in A[k] if rng is None or x <= rng]
    b = list(bpeaks)
    ah = sum(1 for x in a if any(abs(x - y) <= TOL for y in b))
    bh = sum(1 for y in b if any(abs(x - y) <= TOL for x in a))
    print('%-14s %6d %6d %8d %8d %8d' % (bv, len(a), len(b), ah, bh, len(a) + len(b) - bh))
    tot['a'] += len(a); tot['b'] += len(b); tot['ah'] += ah; tot['bh'] += bh
print()
print('合计: A=%d  B=%d  A被B覆盖=%d(%.1f%%)  B被A覆盖=%d(%.1f%%)'
      % (tot['a'], tot['b'], tot['ah'], 100*tot['ah']/max(tot['a'],1), tot['bh'], 100*tot['bh']/max(tot['b'],1)))
print()
print('=== A 在每个视频上标了几个峰（对照 B）===')
for bv, (bpeaks, rng) in B.items():
    k = batch.get(bv)
    if not k: continue
    a = sorted(x for x in A[k] if rng is None or x <= rng)
    print('  %-14s A: %s' % (bv, [int(x) for x in a][:14]))
    print('  %-14s B: %s' % ('', bpeaks))