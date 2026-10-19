#!/usr/bin/env python3
"""
highlight_core.py — 精彩时刻检测纯函数库（PLAN_HIGHLIGHTS §2.1–2.5）

本模块只做「纯计算」：不 import pyspark、不读 HDFS、无随机数、无全局可变状态。
用途：
  * jobs/m2_l4_highlights.py 以 RDD 逐视频调用（spark-submit 需 --py-files 分发本文件）
  * tests/test_highlight_core.py 直接用合成数据单测（不依赖 Spark 集群）

算法口径（唯一依据 = PLAN_HIGHLIGHTS.md §2.1–2.5，词典/阈值规则，不训练模型）：
  1. 复读桶统计 → 平台峰检测：局部极大（平台取中点、端点不算）+ s(b)>=3 + n_rep_texts(b)>=1
  2. 峰 → 段：以峰为中心按 25% 回落定界；相邻峰「谷值 ÷ 较小峰 > 0.5」合并为同一爆发区
  3. 情绪类型：词典规则（感人/好笑/燃/愤怒/中性，命中即停），仅供参照
  4. 批判弹幕：confirmed 词子串命中 + len(text_norm)>=8 + sent_score<0；候选桶 n_crit>=2、
     间隔<=2 桶合并、段 Σ n_crit>=3 才保留（宁缺毋滥）
  5. 从众/中立/批判分类、视频反馈画像、百分位/稳健 z、切片/高光选取

诚实边界（写进报告 JSON 与页面，不许删）：
  * 复读近似 = text_norm 完全相等（不做编辑距离/语义相似）；「哈哈哈」与「哈哈哈哈哈」不合并
  * 情绪类型是词典规则判定，标注「参考」，不宣称准确
  * 批判层可能很薄：未检出就写未检出，不为凑数放宽阈值（破例规则仅在检出率<5% 时允许并写入报告）
"""
import bisect
import hashlib
import math
import re

ENGINE_VERSION = 'hl-v1'           # 算法/表口径版本（改口径必须换版本号）
DICT_VERSION = 'critique-v1.1'     # 质疑词词典版本（Lead 定稿；文件内容哈希另记报告 JSON）

BUCKET_SEC = 5                     # 复读检测网格 5s（沿用 v1 复读聚集口径）
OPENING_SEC = 60                   # 片头噪声提示阈值（start_sec<60 → in_opening）
LONG_VIDEO_SEC = 1800              # 长视频/短视频分界（切片建议 vs 高光定位）

PEAK_MIN_REPEAT = 3                # 峰约束：s(b)=Σmax(n-1,0) >= 3
PEAK_MIN_REP_TEXTS = 1             # 峰约束：n_rep_texts(b)（同文本 5s 内 >=3 条）>= 1
SEG_FALLBACK_RATIO = 0.25          # 峰→段定界：向两侧扩展到 s(b) < max(1, 0.25*s_peak)
MERGE_VALLEY_RATIO = 0.5           # 相邻峰合并：谷值 ÷ 较小峰 > 0.5 → 同一爆发区
TOP_DANMAKU_MAX = 5                # 共振段代表弹幕最多 5 个

CRITIQUE_MIN_LEN = 8               # 批判弹幕：text_norm 长度 >= 8
CRITIQUE_CAND_MIN = 2              # 批判段候选桶：n_crit(b) >= 2
CRITIQUE_MERGE_GAP = 2             # 批判段：候选桶间隔 <= 2 桶合并
CRITIQUE_SEG_MIN = 3               # 批判段：段 Σ n_crit >= 3
CRITIQUE_TOP_N = 3                 # 批判段最多取分数前 3 段
CRITIQUE_TOP_DANMAKU = 3           # 批判段代表弹幕最多 3 条
CRITIQUE_RELAXED = {'cand_min': 1, 'merge_gap': 2, 'min_sum': 2}  # §2.4 破例（检出率<5% 才启用）

# 情绪规则（PLAN §2.3；顺序=命中即停：感人→好笑→燃→愤怒→中性默认）
EMOTION_RULES = [
    ('moving', '感人', re.compile('泪|哭|破防|感动|意难平|心疼|呜呜|泪目|绷不住泪')),
    ('funny', '好笑', re.compile('哈{2,}|笑死|笑不活|笑麻|绷不住|233|草|乐死|乐|神金|什么鬼|太草了')),
    ('hype', '燃', re.compile('燃|帅|牛|强|高能|名场面|超神|卧槽|我操|炸|起飞|封神|6{3,}|awsl|燃爆')),
    ('angry', '愤怒', re.compile('怒|气死|退钱|恶心|凭什么|翻车|烂|垃圾|尬|拉胯|硬吹|洗地|离谱')),
]

# 从众语气词表（PLAN §2.5；弱表达 → 从众侧）
HERD_TONE_RE = re.compile('哈|233|666|好耶|草|泪目|前排|打卡|来了|啊啊|呜呜|哇|卧槽|牛|第一|空降|\\?|？')


# ══════════════════════ 一、基础工具 ══════════════════════

def platform_peaks(seq):
    """局部极大（平台取中点；端点 run 不算峰——scipy find_peaks 口径，沿用 l2_common.platform_peaks）。

    seq: 一维序列（list/np.ndarray）。返回峰下标列表（升序）。
    """
    idx = []
    n = len(seq)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and seq[j + 1] == seq[i]:
            j += 1
        left_ok = i > 0 and seq[i - 1] < seq[i]
        right_ok = j < n - 1 and seq[j] > seq[j + 1]
        if left_ok and right_ok:
            idx.append((i + j) // 2)
        i = j + 1
    return idx


def bucket_midpoint_sec(bucket):
    """5s 桶中点秒（峰定位用）：b 桶覆盖 [b*5, (b+1)*5)，中点 = b*5+2.5。"""
    return float(bucket) * BUCKET_SEC + BUCKET_SEC / 2.0


def clip_segment_secs(start_bucket, end_bucket, duration):
    """段桶区间 → 秒区间，并 clip 到视频时长（duration 为空/<=0 时只做桶换算）。"""
    start_sec = int(start_bucket) * BUCKET_SEC
    end_sec = (int(end_bucket) + 1) * BUCKET_SEC
    if duration is not None and float(duration) > 0:
        start_sec = min(start_sec, int(float(duration)))
        end_sec = min(end_sec, int(float(duration)))
    return int(start_sec), int(end_sec)


# ══════════════════════ 二、复读聚集 → 共振峰/段（§2.2） ══════════════════════

def detect_peak_buckets(n_repeat, n_rep_texts):
    """复读峰检测：平台峰 + s(b)>=3 + n_rep_texts(b)>=1。

    n_repeat:   每桶复读量 s(b)=Σ_text max(n-1,0)（整数序列，下标=桶号）
    n_rep_texts: 每桶「同文本 5s 内 >=3 条」的文本数（整数序列，与 n_repeat 等长）
    返回峰桶号列表（升序）。纯函数、无随机。
    """
    if not n_repeat:
        return []
    if len(n_repeat) != len(n_rep_texts):
        raise ValueError('n_repeat 与 n_rep_texts 长度必须一致：%d vs %d'
                         % (len(n_repeat), len(n_rep_texts)))
    peaks = []
    for b in platform_peaks(n_repeat):
        if float(n_repeat[b]) >= PEAK_MIN_REPEAT and int(n_rep_texts[b]) >= PEAK_MIN_REP_TEXTS:
            peaks.append(int(b))
    return peaks


def _merge_expanded(prev, seg):
    """把 seg 并入 prev（区间取并，峰分数取较大）。"""
    prev['end_bucket'] = max(prev['end_bucket'], seg['end_bucket'])
    prev['peak_score'] = max(prev['peak_score'], seg['peak_score'])


def expand_segments(n_repeat, peaks, fallback_ratio=SEG_FALLBACK_RATIO):
    """峰 → 候选段：以峰为中心向左右扩展到 s(b) < max(1, fallback_ratio*s_peak) 或端点。

    返回按 (start,end,peak) 升序的候选段列表（纯函数，供单测与 build_segments 复用）。
    """
    n = len(n_repeat)
    expanded = []
    for p in peaks:
        p = int(p)
        if p < 0 or p >= n:
            continue
        s_peak = float(n_repeat[p])
        if s_peak <= 0:
            continue
        thr = max(1.0, float(fallback_ratio) * s_peak)
        left = p
        while left > 0 and float(n_repeat[left - 1]) >= thr:
            left -= 1
        right = p
        while right < n - 1 and float(n_repeat[right + 1]) >= thr:
            right += 1
        expanded.append({'start_bucket': int(left), 'end_bucket': int(right),
                         'peak_bucket': p, 'peak_score': int(round(s_peak))})
    expanded.sort(key=lambda s: (s['start_bucket'], s['end_bucket'], s['peak_bucket']))
    return expanded


def merge_expanded_segments(n_repeat, segments, merge_ratio=MERGE_VALLEY_RATIO):
    """相邻候选段合并（PLAN §2.2-5②）：区间重叠/相邻 → 合并；
    否则「谷值 ÷ 较小峰 > merge_ratio」→ 合并，反之各自成段。

    实现说明（论文口径复核用）：25% 回落定界下，相邻段之间必有桶低于较小峰阈值的 25%，
    所以「谷值比」分支实际主要起 defense-in-depth 作用；真正的合并主要来自重叠/相邻。
    保留该分支以严格对齐 PLAN 文字口径（可单测）。
    """
    merged = []
    for seg in segments:
        seg = dict(seg)
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if seg['start_bucket'] <= prev['end_bucket'] + 1:
            _merge_expanded(prev, seg)                 # 区间重叠/相邻 → 同一爆发区
            continue
        valley = min(float(n_repeat[b]) for b in range(prev['end_bucket'] + 1, seg['start_bucket']))
        peak_min = min(float(prev['peak_score']), float(seg['peak_score']))
        if peak_min > 0 and valley / peak_min > float(merge_ratio):
            _merge_expanded(prev, seg)                 # 谷值 > 较小峰×0.5 → 合并
        else:
            merged.append(seg)
    return merged


def finalize_segments(n_repeat, segments):
    """段内最高峰：peak_score=段内 max s(b)，peak_bucket=首个达到 max 的桶（并列取靠前）。"""
    out = []
    for seg in segments:
        s, e = int(seg['start_bucket']), int(seg['end_bucket'])
        pb = max(range(s, e + 1), key=lambda b: (float(n_repeat[b]), -b))  # 首个最大值
        out.append({'start_bucket': s, 'end_bucket': e,
                    'peak_bucket': int(pb), 'peak_score': int(round(float(n_repeat[pb])))})
    return out


def build_segments(n_repeat, peaks):
    """峰 → 共振段：25% 回落定界 + 相邻峰谷值比合并（PLAN §2.2-5）。

    返回按 start_bucket 升序的段列表：
      {start_bucket, end_bucket, peak_bucket, peak_score}
    """
    if len(n_repeat) == 0 or not peaks:
        return []
    expanded = expand_segments(n_repeat, peaks)
    if not expanded:
        return []
    return finalize_segments(n_repeat, merge_expanded_segments(n_repeat, expanded))


def segment_top_danmaku(start_bucket, end_bucket, bucket_top_texts, bucket_top_n, top_k=TOP_DANMAKU_MAX):
    """段内代表弹幕：各桶 top_text 按累计 top_n 排序（去重），返回 (texts, counts)。

    排序键：累计次数 desc → 文本长度 desc → 字典序 asc（可复现）。
    """
    agg = {}
    for b in range(int(start_bucket), int(end_bucket) + 1):
        if b < 0 or b >= len(bucket_top_texts):
            continue
        text = bucket_top_texts[b]
        if not text:
            continue
        n = int(bucket_top_n[b]) if b < len(bucket_top_n) and bucket_top_n[b] is not None else 0
        agg[text] = agg.get(text, 0) + n
    items = sorted(agg.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[:int(top_k)]
    return [t for t, _ in items], [int(n) for _, n in items]


# ══════════════════════ 三、情绪类型（§2.3，参考级） ══════════════════════

def classify_emotion(top_text, pos_ratio=None, neg_ratio=None):
    """段情绪类型（词典规则，命中即停）。返回 (emotion_type, emotion_label)。

    顺序：感人 → 好笑 → 燃 → 愤怒 → 中性默认；中性时若 neg_ratio>0.5 → 愤怒（弱证据，仍标参考）。
    """
    text = top_text or ''
    for typ, label, rx in EMOTION_RULES:
        if rx.search(text):
            return typ, label
    if neg_ratio is not None and float(neg_ratio) > 0.5:
        return 'angry', '愤怒'
    return 'neutral', '中性'


# ══════════════════════ 四、情感曲线覆盖（§2.2-8） ══════════════════════

def curve_ratios(curve, start_sec, end_sec, duration):
    """段覆盖的 1% 桶情感占比均值：返回 (pos_ratio, neg_ratio, neu_ratio)。

    curve: [{bucket, pos_ratio, neg_ratio, neu_ratio}, ...]（dws_sentiment_curve_1pct 的子集）
    覆盖 = 桶时间区间 [b*d/100, (b+1)*d/100) 与 [start_sec, end_sec) 相交。
    无曲线/无覆盖桶/时长非法 → (None, None, None)（页面显示为空，不伪造）。
    """
    if not curve or duration is None or float(duration) <= 0:
        return None, None, None
    if start_sec is None or end_sec is None or float(end_sec) <= float(start_sec):
        return None, None, None
    d = float(duration)
    b_lo = max(0, min(99, int(math.floor(float(start_sec) / d * 100.0))))
    b_hi = max(0, min(99, int(math.floor((float(end_sec) - 1e-9) / d * 100.0))))
    if b_hi < b_lo:
        return None, None, None
    acc = [0.0, 0.0, 0.0]
    cnt = [0, 0, 0]
    keys = ('pos_ratio', 'neg_ratio', 'neu_ratio')
    for row in curve:
        b = row.get('bucket')
        if b is None or b < b_lo or b > b_hi:
            continue
        for i, k in enumerate(keys):
            v = row.get(k)
            if v is not None:
                acc[i] += float(v)
                cnt[i] += 1
    return tuple((acc[i] / cnt[i]) if cnt[i] else None for i in range(3))


# ══════════════════════ 五、批判弹幕与批判段（§2.1/§2.4） ══════════════════════

def load_critique_dict(path):
    """读质疑词词典 config/dicts/critique_words.tsv（UTF-8；列 word/weight/source/confirmed）。

    跳过 # 注释与空行；返回 [{word, weight, source, confirmed, line}]（保留文件顺序）。
    格式错误抛 ValueError（词典是契约，不静默吞行）。
    """
    rows = []
    with open(path, encoding='utf-8-sig') as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split('\t')
            if len(parts) != 4:
                raise ValueError('%s:%d 列数应为 4（word/weight/source/confirmed），实际 %d'
                                 % (path, lineno, len(parts)))
            word, weight, source, confirmed = [p.strip() for p in parts]
            if not word:
                raise ValueError('%s:%d 词为空' % (path, lineno))
            try:
                weight = float(weight)
                confirmed = int(confirmed)
            except ValueError:
                raise ValueError('%s:%d weight/confirmed 非法：%r/%r'
                                 % (path, lineno, weight, confirmed))
            if source not in ('seed', 'expand'):
                raise ValueError('%s:%d source 应为 seed|expand，实际 %r' % (path, lineno, source))
            if confirmed not in (0, 1):
                raise ValueError('%s:%d confirmed 应为 0|1，实际 %r' % (path, lineno, confirmed))
            if not (0.0 < weight <= 5.0):
                raise ValueError('%s:%d weight 应在 (0,5]，实际 %r' % (path, lineno, weight))
            rows.append({'word': word, 'weight': weight, 'source': source,
                         'confirmed': confirmed, 'line': lineno})
    return rows


def confirmed_words(rows):
    """confirmed=1 的词 → [(word, weight)]（同词取最大 weight；按长度降序、字典序稳定）。"""
    wmap = {}
    for r in rows:
        if int(r.get('confirmed', 0)) != 1:
            continue
        w = float(r['weight'])
        if r['word'] not in wmap or w > wmap[r['word']]:
            wmap[r['word']] = w
    return sorted(wmap.items(), key=lambda kv: (-len(kv[0]), kv[0]))


def dict_fingerprint(path):
    """词典文件 sha1 前 12 位（报告里做可追溯；dict_version 由人工版本号给出）。"""
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def hit_words(text, words):
    """子串命中：返回 [(word, weight)]（按字典序，便于复现）。words=[(word,weight),...]。"""
    text = text or ''
    return sorted([(w, wt) for w, wt in words if w and w in text])


def is_critique_danmaku(hit, text_norm, sent_score, min_len=CRITIQUE_MIN_LEN):
    """批判弹幕判定：命中 confirmed 质疑词 且 len>=8 且 sent_score<0。"""
    if not hit:
        return False
    if text_norm is None or sent_score is None:
        return False
    if len(text_norm) < int(min_len):
        return False
    return float(sent_score) < 0.0


def critique_weight(text_norm, sent_score, word_weight=1.0):
    """单条批判弹幕权重：w = (1 + 0.05*min(len,40) + 0.2*min(|sent|,5)) × 命中词最大 weight。

    多词命中不累加（取最大 weight，避免「强行+强行煽情」双重计分；Lead 2026-10-19 裁定）。
    """
    ln = len(text_norm or '')
    sent = abs(float(sent_score or 0.0))
    base = 1.0 + 0.05 * min(ln, 40) + 0.2 * min(sent, 5.0)
    return float(base) * float(word_weight)


def find_critique_segments(bucket_stats,
                           cand_min=CRITIQUE_CAND_MIN,
                           merge_gap=CRITIQUE_MERGE_GAP,
                           min_sum=CRITIQUE_SEG_MIN,
                           top_n=CRITIQUE_TOP_N):
    """批判段：候选桶 n_crit>=cand_min → 间隔<=merge_gap 桶合并 → 段 Σn_crit>=min_sum 才保留。

    bucket_stats: [{b, n_crit, score}, ...]（任意顺序；b 为 5s 桶号）
    返回按 critique_score desc 排序（并列 start 升序）的前 top_n 段：
      {start_bucket, end_bucket, n_critique, critique_score}
    seg_id 由调用方按本返回顺序编号（0=分数最高段）。
    """
    cands = []
    for x in bucket_stats:
        if x.get('b') is None:
            continue
        if int(x.get('n_crit', 0)) >= int(cand_min):
            cands.append({'b': int(x['b']),
                          'n_crit': int(x.get('n_crit', 0)),
                          'score': float(x.get('score', 0.0))})
    cands.sort(key=lambda x: x['b'])
    groups = []
    for c in cands:
        if groups and c['b'] - groups[-1][-1]['b'] <= int(merge_gap):
            groups[-1].append(c)
        else:
            groups.append([c])
    segs = []
    for g in groups:
        n_crit = sum(x['n_crit'] for x in g)
        if n_crit < int(min_sum):
            continue
        segs.append({'start_bucket': int(g[0]['b']), 'end_bucket': int(g[-1]['b']),
                     'n_critique': int(n_crit),
                     'critique_score': float(sum(x['score'] for x in g))})
    segs.sort(key=lambda s: (-s['critique_score'], s['start_bucket']))
    return segs[:int(top_n)]


def critique_top_danmaku(rows, top_n=CRITIQUE_TOP_DANMAKU):
    """批判段代表弹幕：按「长度 + 负向强度」排序取前 N。返回 (texts, counts)。

    rows: [{text, sent_score, n}, ...]（同文本多条会合并；n 为条数）
    """
    agg = {}
    for r in rows:
        text = r.get('text')
        if not text:
            continue
        n = int(r.get('n', 1))
        sent = float(r.get('sent_score') or 0.0)
        rec = agg.get(text)
        if rec is None:
            agg[text] = {'n': n, 'sent': sent}
        else:
            rec['n'] += n
            if sent < rec['sent']:
                rec['sent'] = sent
    items = sorted(agg.items(),
                   key=lambda kv: (-(len(kv[0]) + abs(kv[1]['sent'])), -kv[1]['n'], kv[0]))
    top = items[:int(top_n)]
    return [t for t, _ in top], [int(v['n']) for _, v in top]


# ══════════════════════ 六、从众/中立/批判分类与视频画像（§2.5） ══════════════════════

def classify_feedback_text(text_norm, n, is_critique):
    """文本级三分类（权重=该文本条数 n）：critique 优先 → herd（复读/短/语气词）→ neutral。"""
    if is_critique:
        return 'critique'
    text = text_norm or ''
    if int(n) >= 2:
        return 'herd'
    if len(text) <= 4:
        return 'herd'
    if HERD_TONE_RE.search(text):
        return 'herd'
    return 'neutral'


def feedback_ratios(n_herd, n_neutral, n_critique):
    """三分类占比（按条数）：返回 (ratio_herd, ratio_neutral, ratio_critique)；总数为 0 → 全 0。"""
    total = int(n_herd) + int(n_neutral) + int(n_critique)
    if total <= 0:
        return 0.0, 0.0, 0.0
    return (float(n_herd) / total, float(n_neutral) / total, float(n_critique) / total)


def resonance_summary(peaks):
    """视频级共振汇总：{n_peaks, resonance_total(Σpeak_score), max_peak_score}；无峰全 0。"""
    scores = [int(p.get('peak_score', 0)) for p in (peaks or [])]
    return {'n_peaks': len(scores),
            'resonance_total': int(sum(scores)),
            'max_peak_score': int(max(scores)) if scores else 0}


# ══════════════════════ 七、分位/稳健 z（§2.2-9 / §2.5） ══════════════════════

def percentile_rank(value, sorted_values):
    """百分位（0–100，保留 2 位）= 100 × (#基准值 <= value) / N。空基准 → None。"""
    if not sorted_values:
        return None
    n = len(sorted_values)
    return round(100.0 * bisect.bisect_right(sorted_values, float(value)) / n, 2)


def median_mad(values):
    """中位数与 MAD（未乘 1.4826）：返回 (median, mad)；空 → (None, None)。"""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return None, None

    def _median(a):
        m = len(a)
        if m == 0:
            return None
        return a[m // 2] if m % 2 else (a[m // 2 - 1] + a[m // 2]) / 2.0

    med = _median(vals)
    mad = _median(sorted(abs(v - med) for v in vals))
    return med, mad


def robust_z(value, med, mad):
    """稳健 z = (v − 中位) / (1.4826×MAD)；MAD=0 或缺失 → None（不伪造 0）。"""
    if med is None or mad is None or float(mad) <= 0:
        return None
    return (float(value) - float(med)) / (1.4826 * float(mad))


# ══════════════════════ 八、切片/高光选取（§2.6；Lead 2026-10-19 产品政策） ══════════════════════

def _peak_score_of(p):
    return float(p.get('score', p.get('peak_score', 0.0)) or 0.0)


def _rank_peaks(peaks):
    """峰排序：分数 desc → start_sec asc → end_sec asc（确定性）。"""
    return sorted(peaks or [],
                  key=lambda p: (-_peak_score_of(p),
                                 float(p.get('start_sec', 0) or 0),
                                 float(p.get('end_sec', 0) or 0)))


def _pick_peaks(peaks, top_n, prefer_non_opening):
    """优先非片头峰；非片头不足 top_n 才回落片头峰。返回 (picked, n_from_opening)。"""
    ranked = _rank_peaks(peaks)
    if not prefer_non_opening:
        return ranked[:int(top_n)], 0
    non = [p for p in ranked if not p.get('in_opening')]
    opening = [p for p in ranked if p.get('in_opening')]
    picked = non[:int(top_n)]
    need = int(top_n) - len(picked)
    if need > 0:
        picked = picked + opening[:need]
    return picked, len(picked) - min(len(non), int(top_n))


def select_slices(peaks, top_n=3, prefer_non_opening=True):
    """长视频切片建议：top_n 段（优先非片头，不足回落片头并标 opening_only）。

    返回 [{rank,start_sec,end_sec,peak_sec,emotion,label,score,pctl,top_danmaku,opening_only}]
    """
    picked, _ = _pick_peaks(peaks, top_n, prefer_non_opening)
    out = []
    for rank, p in enumerate(picked, 1):
        out.append({'rank': rank,
                    'start_sec': int(p.get('start_sec', 0)),
                    'end_sec': int(p.get('end_sec', 0)),
                    'peak_sec': float(p.get('peak_sec', bucket_midpoint_sec(0))),
                    'emotion': p.get('emotion_type', 'neutral'),
                    'label': p.get('emotion_label', '中性'),
                    'score': p.get('score', p.get('peak_score')),
                    'pctl': p.get('pctl'),
                    'top_danmaku': list(p.get('top_danmaku') or []),
                    'opening_only': bool(p.get('in_opening'))})
    return out


def select_highlights(peaks, top_n=3, prefer_non_opening=True):
    """短视频高光定位：top_n 个时刻（sec=峰桶中点；优先非片头，不足回落片头并标 opening_only）。

    返回 [{rank,sec,start_sec,end_sec,emotion,label,score,pctl,opening_only}]
    """
    picked, _ = _pick_peaks(peaks, top_n, prefer_non_opening)
    out = []
    for rank, p in enumerate(picked, 1):
        out.append({'rank': rank,
                    'sec': float(p.get('peak_sec', bucket_midpoint_sec(0))),
                    'start_sec': int(p.get('start_sec', 0)),
                    'end_sec': int(p.get('end_sec', 0)),
                    'emotion': p.get('emotion_type', 'neutral'),
                    'label': p.get('emotion_label', '中性'),
                    'score': p.get('score', p.get('peak_score')),
                    'pctl': p.get('pctl'),
                    'opening_only': bool(p.get('in_opening'))})
    return out
