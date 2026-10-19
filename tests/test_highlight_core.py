#!/usr/bin/env python3
"""test_highlight_core.py — 精彩时刻检测纯函数单测（PLAN_HIGHLIGHTS §2.2–2.6）

运行：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python -m pytest tests/test_highlight_core.py -q
全部为纯函数级测试（不启动 Spark）：合成数据覆盖峰检测、25% 段定界与合并、情绪规则、
批判判定/分段、从众-中立-批判分类、比例、分位/稳健 z、切片/高光选取，以及边界案例
（无复读 / 无批判 / 超长视频 / 短视频 / 空序列）。
"""
import io
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, '..', 'jobs')
ROOT = os.path.dirname(HERE)
sys.path.insert(0, JOBS)

import highlight_core as H  # noqa: E402


# ═══════════════════════ 0. 纯函数约束 ═══════════════════════

def test_core_has_no_pyspark_or_random():
    """highlight_core 必须可脱离 Spark 单测：不 import pyspark、不用随机数。"""
    src = io.open(os.path.join(JOBS, 'highlight_core.py'), encoding='utf-8').read()
    for bad in (r'^\s*import pyspark', r'^\s*from pyspark', r'^\s*import random', r'np\.random'):
        assert re.search(bad, src, re.M) is None, \
            'highlight_core.py 不应依赖 %r（纯函数、无随机）' % bad


# ═══════════════════════ 1. 平台峰 ═══════════════════════

def test_platform_peaks_basic():
    assert H.platform_peaks([0, 1, 0]) == [1]
    assert H.platform_peaks([0, 2, 2, 0]) == [1]       # 平台取中点
    assert H.platform_peaks([2, 1, 0]) == []           # 端点不算峰
    assert H.platform_peaks([0, 0, 0]) == []
    assert H.platform_peaks([0, 2, 1, 2, 0]) == [1, 3]


def test_peak_bucket_midpoint():
    assert H.bucket_midpoint_sec(0) == 2.5
    assert H.bucket_midpoint_sec(3) == 17.5


def test_detect_peak_buckets_constraints():
    assert H.detect_peak_buckets([0, 0, 5, 0, 0], [0, 0, 1, 0, 0]) == [2]
    # n_rep_texts(b)>=1 且 s(b)>=3：任一不满足都不算峰
    assert H.detect_peak_buckets([0, 0, 5, 0, 0], [0, 0, 0, 0, 0]) == []
    assert H.detect_peak_buckets([0, 0, 2, 0, 0], [0, 0, 1, 0, 0]) == []
    # 平台峰取中点
    assert H.detect_peak_buckets([0, 5, 5, 0], [0, 1, 1, 0]) == [1]
    # 端点 run 不算峰
    assert H.detect_peak_buckets([5, 3, 0], [1, 1, 0]) == []
    with pytest.raises(ValueError):
        H.detect_peak_buckets([1, 2], [1])


# ═══════════════════════ 2. 峰 → 段（25% 定界 + 谷值合并） ═══════════════════════

def test_expand_segments_25pct():
    """以峰为中心：s(b) 跌破 max(1, 0.25×s_peak) 即停（峰=12 → 阈值 3）。"""
    nr = [0, 0, 0, 12, 4, 3, 2, 1, 0, 0]
    segs = H.expand_segments(nr, [3])
    assert [(s['start_bucket'], s['end_bucket']) for s in segs] == [(3, 5)]


def test_build_segments_flat_no_segments():
    assert H.build_segments([0] * 20, []) == []
    assert H.build_segments([], []) == []
    assert H.build_segments([0, 0, 0], [1]) == []           # 峰高 0 → 无段


def test_build_segments_merge_overlapping_peaks():
    """两个峰扩展区间重叠 → 合并为一段，段峰取段内 max。"""
    nr = [0, 0, 10, 8, 6, 0, 0]
    segs = H.build_segments(nr, [2, 4])
    assert len(segs) == 1
    assert segs[0] == {'start_bucket': 2, 'end_bucket': 4,
                       'peak_bucket': 2, 'peak_score': 10}


def test_merge_expanded_segments_valley_ratio():
    """谷值 ÷ 较小峰 > 0.5 → 合并；否则各自成段（直接单测合并分支）。"""
    nr = [0, 10, 8, 7, 6, 7, 7, 7, 8, 6, 0]
    a = {'start_bucket': 1, 'end_bucket': 2, 'peak_bucket': 1, 'peak_score': 10}
    b = {'start_bucket': 8, 'end_bucket': 9, 'peak_bucket': 8, 'peak_score': 8}
    merged = H.merge_expanded_segments(nr, [a, b])
    assert len(merged) == 1 and merged[0]['start_bucket'] == 1 and merged[0]['end_bucket'] == 9
    # 谷值 0（0/8 = 0 < 0.5）→ 两段
    nr2 = [0, 10, 8, 1, 0, 2, 1, 1, 8, 6, 0]
    merged2 = H.merge_expanded_segments(nr2, [a, b])
    assert len(merged2) == 2
    # 区间相邻也合并（start <= prev.end+1）
    c = {'start_bucket': 3, 'end_bucket': 4, 'peak_bucket': 3, 'peak_score': 4}
    merged3 = H.merge_expanded_segments([0] * 8, [a, c])
    assert len(merged3) == 1 and merged3[0]['end_bucket'] == 4


def test_finalize_segments_peak_choice():
    """段内最高峰取首个最大值（并列取靠前），peak_score=段内 max。"""
    nr = [0, 10, 8, 10, 0]
    segs = H.finalize_segments(nr, [{'start_bucket': 0, 'end_bucket': 4,
                                     'peak_bucket': 0, 'peak_score': 0}])
    assert segs[0]['peak_bucket'] == 1 and segs[0]['peak_score'] == 10


def test_segment_top_danmaku_dedup_and_order():
    texts = ['哈', '哈', '笑死', '']
    counts = [5, 3, 2, 0]
    t, c = H.segment_top_danmaku(0, 2, texts, counts)
    assert t == ['哈', '笑死'] and c == [8, 2]
    # 同次数取更长文本
    t2, c2 = H.segment_top_danmaku(0, 1, ['短', '更长文本'], [2, 2])
    assert t2 == ['更长文本', '短'] and c2 == [2, 2]


def test_clip_segment_secs():
    assert H.clip_segment_secs(2, 5, 40) == (10, 30)
    assert H.clip_segment_secs(2, 5, 20) == (10, 20)     # clip 到时长
    assert H.clip_segment_secs(2, 5, None) == (10, 30)
    assert H.clip_segment_secs(5040, 5045, 25205) == (25200, 25205)   # 超长视频封顶


# ═══════════════════════ 3. 情绪类型（规则顺序） ═══════════════════════

@pytest.mark.parametrize('text,pos,neg,expect', [
    ('泪目', None, None, ('moving', '感人')),
    ('绷不住泪了', None, None, ('moving', '感人')),      # 感人优先于好笑（含「绷不住」）
    ('哈哈哈哈', None, None, ('funny', '好笑')),
    ('笑死我了', None, None, ('funny', '好笑')),
    ('卧槽这波太燃了', None, None, ('hype', '燃')),
    ('退钱', None, None, ('angry', '愤怒')),
    ('好看', None, None, ('neutral', '中性')),
    ('好看', None, 0.6, ('angry', '愤怒')),              # 中性默认 + neg>0.5 弱证据
    ('哈哈哈哈', None, 0.9, ('funny', '好笑')),          # 规则命中即停，不被 neg 覆盖
])
def test_classify_emotion(text, pos, neg, expect):
    assert H.classify_emotion(text, pos, neg) == expect


# ═══════════════════════ 4. 批判弹幕判定 / 权重 / 分段 ═══════════════════════

def test_is_critique_danmaku():
    long_text = '逻辑不通剧本太烂了'
    assert H.is_critique_danmaku(True, long_text, -2.0) is True
    assert H.is_critique_danmaku(True, '就这水平', -1.0) is False      # len<8
    assert H.is_critique_danmaku(True, long_text, 0.0) is False        # 非负情感
    assert H.is_critique_danmaku(True, long_text, None) is False
    assert H.is_critique_danmaku(False, long_text, -2.0) is False
    assert H.is_critique_danmaku(True, None, -2.0) is False


def test_critique_weight_formula_and_caps():
    # len=9 → 0.05*9=0.45；|sent|=3 → 0.2*3=0.6；× 词权重 1.2
    assert abs(H.critique_weight('逻辑不通剧本太烂了', -3.0, 1.2) - (1 + 0.45 + 0.6) * 1.2) < 1e-12
    # len 封顶 40、|sent| 封顶 5：1 + 2 + 1 = 4
    assert abs(H.critique_weight('x' * 100, -9.0, 1.0) - 4.0) < 1e-12
    # 负向强度取绝对值（-sent 与 +sent 同权）
    assert abs(H.critique_weight('x' * 10, -2.0, 1.0) - H.critique_weight('x' * 10, 2.0, 1.0)) < 1e-12


def test_find_critique_segments_thresholds():
    # 候选桶 n_crit>=2；单桶 Σ=2 <3 → 丢弃
    st = [{'b': 10, 'n_crit': 2, 'score': 2.0},
          {'b': 11, 'n_crit': 1, 'score': 1.0},        # n_crit=1 不是候选
          {'b': 20, 'n_crit': 2, 'score': 3.0}]
    assert H.find_critique_segments(st) == []
    # 间隔 2 桶合并：10/12 合成一段（Σ=4），单人得 1 段
    st2 = [{'b': 10, 'n_crit': 2, 'score': 2.0},
           {'b': 12, 'n_crit': 2, 'score': 2.0}]
    segs = H.find_critique_segments(st2)
    assert len(segs) == 1 and segs[0]['start_bucket'] == 10 and segs[0]['end_bucket'] == 12
    assert segs[0]['n_critique'] == 4 and abs(segs[0]['critique_score'] - 4.0) < 1e-12
    # 间隔 3 桶不合并，各 Σ=2 <3 → 空
    st3 = [{'b': 10, 'n_crit': 2, 'score': 1.0},
           {'b': 13, 'n_crit': 2, 'score': 1.0}]
    assert H.find_critique_segments(st3) == []
    # 破例参数（n_crit>=1 / Σ>=2）：10/11 合并成 1 段（Σ=2）保留；30/31 同理 → 2 段
    st4 = [{'b': 10, 'n_crit': 1, 'score': 1.0},
           {'b': 11, 'n_crit': 1, 'score': 1.0},
           {'b': 30, 'n_crit': 1, 'score': 1.0},
           {'b': 31, 'n_crit': 1, 'score': 1.0}]
    relaxed = H.find_critique_segments(st4, **H.CRITIQUE_RELAXED)
    assert len(relaxed) == 2 and [s['n_critique'] for s in relaxed] == [2, 2]
    # 严格参数下同样输入全部丢弃（Σ<3）
    assert H.find_critique_segments(st4) == []


def test_find_critique_segments_top3_score_order():
    st = [{'b': 0, 'n_crit': 3, 'score': 1.0},
          {'b': 20, 'n_crit': 3, 'score': 5.0},
          {'b': 40, 'n_crit': 3, 'score': 3.0},
          {'b': 60, 'n_crit': 3, 'score': 4.0}]
    segs = H.find_critique_segments(st)
    assert len(segs) == 3
    assert [s['critique_score'] for s in segs] == [5.0, 4.0, 3.0]   # 分数降序 = seg_id 顺序
    assert [s['start_bucket'] for s in segs] == [20, 60, 40]


def test_critique_top_danmaku_ranking():
    rows = [{'text': '这是很长的批判弹幕', 'sent_score': -2.0, 'n': 1},
            {'text': '太假了', 'sent_score': -3.0, 'n': 4},
            {'text': '浪费时间', 'sent_score': -1.0, 'n': 2}]
    texts, counts = H.critique_top_danmaku(rows)
    # 排序键 = 长度 + |sent|：11 > 6 > 5
    assert texts == ['这是很长的批判弹幕', '太假了', '浪费时间']
    assert counts == [1, 4, 2]
    assert H.critique_top_danmaku([]) == ([], [])


# ═══════════════════════ 5. 从众/中立/批判三分类 ═══════════════════════

def test_classify_feedback_text():
    assert H.classify_feedback_text('跟风复读', 3, False) == 'herd'        # n>=2 复读
    assert H.classify_feedback_text('哈哈哈哈', 1, False) == 'herd'        # 语气词
    assert H.classify_feedback_text('前排', 1, False) == 'herd'
    assert H.classify_feedback_text('短', 1, False) == 'herd'              # len<=4
    assert H.classify_feedback_text('这是一条认真讨论剧情的弹幕', 1, False) == 'neutral'
    assert H.classify_feedback_text('这是一条认真讨论剧情的弹幕', 1, True) == 'critique'  # 批判优先


def test_feedback_ratios():
    assert H.feedback_ratios(5, 3, 2) == (0.5, 0.3, 0.2)
    assert H.feedback_ratios(0, 0, 0) == (0.0, 0.0, 0.0)


def test_resonance_summary():
    assert H.resonance_summary([]) == {'n_peaks': 0, 'resonance_total': 0, 'max_peak_score': 0}
    assert H.resonance_summary([{'peak_score': 3}, {'peak_score': 7}]) == \
        {'n_peaks': 2, 'resonance_total': 10, 'max_peak_score': 7}


# ═══════════════════════ 6. 情感曲线覆盖 ═══════════════════════

def _curve():
    return [{'bucket': b, 'pos_ratio': 0.01 * b, 'neg_ratio': 0.10,
             'neu_ratio': 0.50, 'n': 10} for b in range(100)]


def test_curve_ratios_covering_buckets():
    # duration=1000 → 每桶 10s；段 [0,10) 只覆盖 bucket 0
    p, n, u = H.curve_ratios(_curve(), 0, 10, 1000)
    assert abs(p - 0.0) < 1e-12 and abs(n - 0.10) < 1e-12 and abs(u - 0.50) < 1e-12
    # 段 [5,15) 跨 bucket 0/1 → 均值
    p2, _, _ = H.curve_ratios(_curve(), 5, 15, 1000)
    assert abs(p2 - 0.005) < 1e-12
    # 边界：正好落在桶边界（end=10 不含 bucket 1）
    p3, _, _ = H.curve_ratios(_curve(), 10, 20, 1000)
    assert abs(p3 - 0.01) < 1e-12


def test_curve_ratios_missing_returns_none():
    assert H.curve_ratios([], 0, 10, 1000) == (None, None, None)
    assert H.curve_ratios(_curve(), 0, 10, None) == (None, None, None)
    assert H.curve_ratios(_curve(), 10, 10, 1000) == (None, None, None)
    assert H.curve_ratios(_curve(), 0, 10, 0) == (None, None, None)


# ═══════════════════════ 7. 分位 / 稳健 z ═══════════════════════

def test_percentile_rank():
    vals = [1.0, 2.0, 2.0, 3.0]
    assert H.percentile_rank(2, vals) == 75.0     # 包含并列：3/4
    assert H.percentile_rank(0, vals) == 0.0
    assert H.percentile_rank(4, vals) == 100.0
    assert H.percentile_rank(2.5, vals) == 75.0
    assert H.percentile_rank(1, []) is None


def test_median_mad_robust_z():
    assert H.median_mad([1, 2, 3, 4, 5]) == (3.0, 1.0)
    assert H.median_mad([1, 2, 3, 4]) == (2.5, 1.0)
    assert H.median_mad([]) == (None, None)
    med, mad = H.median_mad([2, 4, 4, 4, 5, 5, 7, 9])   # 经典例：med=4.5? 实为 (4+5)/2=4.5
    assert abs(med - 4.5) < 1e-12
    assert H.robust_z(5, 4.5, mad) is not None
    assert H.robust_z(5, 4.5, 0) is None
    assert H.robust_z(5, None, 1.0) is None
    # z 公式 = (v−med)/(1.4826×MAD)
    assert abs(H.robust_z(5, 3, 1) - 2 / 1.4826) < 1e-12


# ═══════════════════════ 8. 切片 / 高光选取（Lead 2026-10-19 产品政策） ═══════════════════════

def _peak(start, score, opening, peak_sec=None, label='燃'):
    return {'start_sec': start, 'end_sec': start + 10,
            'peak_sec': float(peak_sec if peak_sec is not None else start + 5),
            'score': score, 'pctl': 80.0, 'in_opening': opening,
            'emotion_type': 'hype', 'emotion_label': label,
            'top_danmaku': ['弹幕%d' % start]}


def test_select_slices_prefers_non_opening():
    peaks = [_peak(10, 50, True), _peak(100, 30, False),
             _peak(200, 40, False), _peak(300, 20, False)]
    out = H.select_slices(peaks, 3)
    assert [p['rank'] for p in out] == [1, 2, 3]
    assert [p['start_sec'] for p in out] == [200, 100, 300]     # 分数降序
    assert all(p['opening_only'] is False for p in out)
    assert out[0]['label'] == '燃' and out[0]['top_danmaku'] == ['弹幕200']


def test_select_slices_falls_back_to_opening():
    peaks = [_peak(10, 50, True), _peak(20, 45, True), _peak(100, 30, False)]
    out = H.select_slices(peaks, 3)
    # 非片头只有 1 个 → 补 2 个片头峰，且逐条标 opening_only
    assert [p['start_sec'] for p in out] == [100, 10, 20]
    assert [p['opening_only'] for p in out] == [False, True, True]


def test_select_slices_empty_and_fewer_than_topn():
    assert H.select_slices([], 3) == []
    out = H.select_slices([_peak(100, 30, False)], 3)
    assert len(out) == 1 and out[0]['rank'] == 1


def test_select_highlights_short_video():
    peaks = [_peak(10, 50, True), _peak(100, 30, False), _peak(200, 40, False)]
    out = H.select_highlights(peaks, 3)
    assert [p['start_sec'] for p in out] == [200, 100, 10]
    assert [p['sec'] for p in out] == [205.0, 105.0, 15.0]
    assert [p['opening_only'] for p in out] == [False, False, True]


# ═══════════════════════ 9. 词典解析 ═══════════════════════

def test_load_critique_dict_and_confirmed_words(tmp_path):
    p = tmp_path / 'critique_words.tsv'
    p.write_text('# 注释行\n\n不对\t1.2\tseed\t1\n强行\t1.0\tseed\t1\n待定词\t1.5\texpand\t0\n',
                 encoding='utf-8')
    rows = H.load_critique_dict(str(p))
    assert [r['word'] for r in rows] == ['不对', '强行', '待定词']
    words = H.confirmed_words(rows)
    assert dict(words) == {'不对': 1.2, '强行': 1.0}      # 未确认词不进入检测
    assert dict(H.hit_words('这不对吧', words)) == {'不对': 1.2}
    assert H.hit_words('', words) == []
    assert len(H.dict_fingerprint(str(p))) == 12


def test_confirmed_words_dedup_max_weight():
    rows = [{'word': '强行', 'weight': 1.0, 'source': 'seed', 'confirmed': 1},
            {'word': '强行', 'weight': 1.3, 'source': 'expand', 'confirmed': 1}]
    assert dict(H.confirmed_words(rows)) == {'强行': 1.3}


def test_load_critique_dict_bad_line_raises(tmp_path):
    p = tmp_path / 'bad.tsv'
    p.write_text('只有三列\t1.0\tseed\n', encoding='utf-8')
    with pytest.raises(ValueError):
        H.load_critique_dict(str(p))


def test_repo_critique_dict_contract():
    """仓库内定稿词典的契约检查（文件缺失时 skip，便于分头开发）。"""
    path = os.path.join(ROOT, 'config', 'dicts', 'critique_words.tsv')
    assert os.path.exists(path), 'Lead 定稿词典缺失：config/dicts/critique_words.tsv'
    rows = H.load_critique_dict(path)
    words = H.confirmed_words(rows)
    assert len(rows) >= 40 and len(words) >= 40
    assert all(0.0 < wt <= 5.0 for _, wt in words)
    # Lead 2026-10-19 修订：删掉单字噪声词「假/骗/烂」，改用双字以上词
    for bad in ('假', '骗', '烂'):
        assert bad not in dict(words), '词典不应含单字噪声词 %r' % bad
    assert H.DICT_VERSION.startswith('critique')


# ═══════════════════════ 10. 边界案例 ═══════════════════════

def test_edge_no_repetition():
    """无复读：全 0 序列无峰、无段；无批判桶无段。"""
    nr = [0] * 50
    nrt = [0] * 50
    assert H.detect_peak_buckets(nr, nrt) == []
    assert H.build_segments(nr, []) == []
    assert H.find_critique_segments([]) == []
    assert H.critique_top_danmaku([]) == ([], [])


def test_edge_short_video_opening_peak():
    """短视频（154s → 31 个 5s 桶）：片头第 2 桶爆发的段边界与片头标记。"""
    L = 31
    nr = [0] * L
    nrt = [0] * L
    nr[2], nrt[2] = 6, 1
    peaks = H.detect_peak_buckets(nr, nrt)
    assert peaks == [2]
    seg = H.build_segments(nr, peaks)[0]
    start_sec, end_sec = H.clip_segment_secs(seg['start_bucket'], seg['end_bucket'], 154)
    assert (start_sec, end_sec) == (10, 15)
    assert start_sec < H.OPENING_SEC            # 片头 → in_opening
    assert seg['peak_score'] == 6


def test_edge_long_video_segment_and_clip():
    """超长视频（25205s → 5041 个桶）：远离片头的段 + 末桶秒数 clip。"""
    L = 5041
    nr = [0] * L
    nrt = [0] * L
    for b in (5000, 5001):
        nr[b] = 9
        nrt[b] = 1
    peaks = H.detect_peak_buckets(nr, nrt)
    assert len(peaks) == 1                    # 平台峰取中点
    seg = H.build_segments(nr, peaks)[0]
    assert seg['start_bucket'] == 5000 and seg['end_bucket'] == 5001
    start_sec, end_sec = H.clip_segment_secs(seg['start_bucket'], seg['end_bucket'], 25205)
    assert (start_sec, end_sec) == (25000, 25010)
    assert start_sec >= H.OPENING_SEC
    # 末桶秒数 clip 到视频时长
    assert H.clip_segment_secs(5040, 5040, 25205) == (25200, 25205)


def test_edge_all_same_score_tie_deterministic():
    """并列峰：peak_bucket 取靠前桶、段顺序稳定（重跑一致）。"""
    nr = [0, 0, 7, 0, 7, 0, 0]
    nrt = [0, 0, 1, 0, 1, 0, 0]
    peaks = H.detect_peak_buckets(nr, nrt)
    assert peaks == [2, 4]
    a = H.build_segments(nr, peaks)
    b = H.build_segments(list(nr), list(peaks))
    assert a == b
    assert [s['start_bucket'] for s in a] == [2, 4]     # 各自成段（谷值 0）


def test_edge_critique_noise_low_count_dropped():
    """批判段宁缺毋滥：段 Σn_crit<3 的噪声不产出段。"""
    st = [{'b': 5, 'n_crit': 1, 'score': 0.9}, {'b': 6, 'n_crit': 1, 'score': 0.8}]
    assert H.find_critique_segments(st) == []
    assert H.find_critique_segments(st, **H.CRITIQUE_RELAXED) != []
