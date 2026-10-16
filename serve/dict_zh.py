# -*- coding: utf-8 -*-
'''业务语言词典（33 号 §2 / §7 交付物 1）——前端主视图只允许引用这里的说法

规则（33 号 §4 军规 2/3）：
  - 不说小数，说方位：一律"高于/低于同类 X%"或"同类前/后 X%"，不露 z 分数
  - 每个偏离维必须带方向 + 一句建议；方向只来自控龄关联（32 号 P3）
      * direction='evidence' → 有控龄显著关联，可标 绿(好)/红(要改)
      * direction='neutral'  → 形状维，P3 无独立证据，只标"中性差异"，不装因果
  - advice 一律"对标式"，不做因果承诺（不写"这样一定更火"）

⚠️ 与 33 号 §2 表格的两处对账（以代码口径为准，已在交付说明中报告）：
  1. early_lag_ratio：代码口径 = lag_sec<6h 的弹幕占比（jobs/m2_l1_fingerprint.py:52），
     **高 = 观众来得快（秒回）**。33 号 §2 原表把高低两列写反，2026-10-16 用户裁定「代码对」后已订正规格书（本条注释保留，作为该口径的裁定留痕）。
  2. opening_5pct：代码口径 = offset_pct<0.05 的弹幕占比，即**前 5%**（不是前 10%）。
'''

DICT_VERSION = 'v3-dict-1'

# 主视图禁止出现的术语（34 号零术语测试的可执行清单）
FORBIDDEN_TERMS = ['opening_ratio', 'opening_5pct', 'z-score', 'zscore', 'z 分数', 'z分数',
                   'entropy', 'rhythm_entropy', 'cosine', '余弦', 'burst_cv', 'early_lag_ratio',
                   'mid_ratio', 'ending_5pct', 'dm_per_min', 'wdm_per_min', 'avg_text_len',
                   'peak_cnt', 'main_peak_pct', 'MAD', 'IsolationForest', 'kNN', 'percentile']

# 方向标签
DIR = {
    'good':    {'label': '方向好', 'color': 'green', 'mark': '🟢'},
    'bad':     {'label': '要改', 'color': 'red', 'mark': '🔴'},
    'neutral': {'label': '中性差异', 'color': 'gray', 'mark': '⚪'},
}

# 强度分级（|z| → 人话），阈值对齐 32 号 P4 的 z>3.5 惯例
LEVELS = [(3.5, '明显偏离'), (2.0, '偏离'), (1.0, '略有差异')]


def level_of(abs_z):
    for th, name in LEVELS:
        if abs_z >= th:
            return name
    return '接近同类'


DICT = {
    # ── 量维：P3 控龄显著，有方向 ─────────────────────────────
    'dm_per_min': {
        'label': '互动热度', 'tech': 'dm_per_min', 'unit': '条/分钟',
        'display': True, 'direction': 'evidence', 'metric_kind': 'rate',
        'seg': (0, 100), 'seg_name': '整片',
        'high_text': '比同类热闹', 'low_text': '比同类冷清',
        'compare_low': '整片弹幕比同类冷清（只有同类的 {r}%）',
        'compare_high': '整片弹幕比同类热闹（是同类的 {r}%）',
        'advice_low': '整片互动偏冷。先把最有信息量、最有冲突的片段提前——观众有得聊才会发弹幕',
        'advice_high': '互动热度是你的优势项，保持当前的节奏密度',
        'p3': 'play ↔ dm_per_min 控龄 ρ=+0.398（n=679）；wdm 同向 +0.402',
    },
    'wdm_per_min': {
        'label': '互动热度（加权）', 'tech': 'wdm_per_min', 'unit': '条/分钟',
        'display': False, 'direction': 'evidence', 'metric_kind': 'rate',
        'seg': (0, 100), 'seg_name': '整片',
        'high_text': '有效弹幕更多', 'low_text': '有效弹幕更少',
        'compare_low': '有效弹幕量只有同类的 {r}%', 'compare_high': '有效弹幕量是同类的 {r}%',
        'advice_low': '参考同分区里互动更热的视频，看它们的开场和中段怎么铺',
        'advice_high': '有效互动高于同类，保持',
        'p3': 'play ↔ wdm_per_min 控龄 ρ=+0.402（n=679）',
    },
    'avg_text_len': {
        'label': '弹幕内容量', 'tech': 'avg_text_len', 'unit': '字/条',
        'display': True, 'direction': 'evidence', 'metric_kind': 'len',
        'seg': (0, 100), 'seg_name': '整片',
        'high_text': '观众愿意打字', 'low_text': '观众只刷短弹幕',
        'compare_low': '平均每条弹幕只有同类的 {r}%（观众懒得打字）',
        'compare_high': '平均每条弹幕是同类的 {r}%（观众愿意打字）',
        'advice_low': '弹幕偏短。给出可讨论的点（提问、留悬念、抛争议）比单纯增加密度更有效',
        'advice_high': '观众愿意展开说，说明内容有讨论度，保持',
        'p3': 'span_days ↔ avg_text_len 控龄 ρ=+0.303（n=135，小样本）；hk_halflife +0.254（n=166）',
    },

    # ── 形状维：P3 无独立证据 → 中性差异 ──────────────────────
    'opening_5pct': {
        'label': '开场吸引力', 'tech': 'opening_5pct', 'unit': '占比',
        'display': True, 'direction': 'neutral', 'metric_kind': 'share',
        'seg': (0, 5), 'seg_name': '前 5%',
        'high_text': '开头就抓住了人', 'low_text': '开场没抓住人——前几秒可能劝退',
        'compare_low': '前 5% 的弹幕只有同类的 {r}%',
        'compare_high': '前 5% 的弹幕是同类的 {r}%',
        'advice_low': '把最有钩子的画面/结论提到前 5%（开头几秒）——同类里开场强的视频都是这么开的',
        'advice_high': '开场抓住人了，这是你的强项',
        'p3': None,
    },
    'early_lag_ratio': {
        'label': '观众来得快不快', 'tech': 'early_lag_ratio', 'unit': '占比',
        'display': True, 'direction': 'neutral', 'metric_kind': 'share',
        'seg': (0, 20), 'seg_name': '开播 6 小时内',
        'high_text': '观众来得快（发布 6 小时内就涌进来）', 'low_text': '观众来得慢（热度是慢慢攒起来的）',
        'compare_low': '你的观众来得比同类慢（同类后 {tail}%）',
        'compare_high': '你的观众来得比同类快（同类前 {tail}%）',
        'advice_low': '观众来得慢。前段要给一个"必须现在看"的理由；也可参考同分区里来得快的视频怎么开场',
        'advice_high': '观众来得快，说明开场有即时吸引力',
        'p3': None,
    },
    'main_peak_pct': {
        'label': '爆点位置', 'tech': 'main_peak_pct', 'unit': '进度%',
        'display': True, 'direction': 'neutral', 'metric_kind': 'pct',
        'seg': (0, 100), 'seg_name': '整片',
        'high_text': '爆点偏后——前面铺垫太长', 'low_text': '爆点偏前——后面可能乏力',
        'compare_low': '你的最大爆点在第 {v}% 就出现了（同类中位在第 {m}%）',
        'compare_high': '你的最大爆点拖到第 {v}% 才出现（同类中位在第 {m}%）',
        'advice_low': '爆点过后还有 {tail}% 没接住。爆点之后要再给一个次高点，别让后半段空转',
        'advice_high': '前面铺垫偏长。把最强的爆点往前提，同类里爆点靠前的视频通常前段就给了信息',
        'p3': None,
    },
    'rhythm_entropy': {
        'label': '有没有高潮', 'tech': 'rhythm_entropy', 'unit': '0–1',
        'display': True, 'direction': 'neutral', 'metric_kind': 'index',
        'seg': (0, 100), 'seg_name': '整片',
        'high_text': '弹幕均匀铺满，没有明显爆点', 'low_text': '弹幕集中在几个点，有明确高潮',
        'compare_low': '弹幕集中在少数几个点（同类里算最集中的 {tail}%）',
        'compare_high': '弹幕铺得很平，没有明确爆点（同类里算最平的 {tail}%）',
        'advice_low': '高潮很集中，注意别只在这一点有反应——中段可以再埋一个小高潮',
        'advice_high': '全片太平。挑一个最强的点做"可预期的高潮"，让观众知道该在哪一刻刷',
        'p3': None,
    },
    'burst_cv': {
        'label': '节奏起伏', 'tech': 'burst_cv', 'unit': '变异系数',
        'display': True, 'direction': 'neutral', 'metric_kind': 'cv',
        'seg': (0, 100), 'seg_name': '整片',
        'high_text': '大起大落，爆点和冷场都明显', 'low_text': '四平八稳，缺少情绪起伏',
        'compare_low': '你的起伏只有同类的 {r}%——全片比较平',
        'compare_high': '你的起伏是同类的 {r}%——大起大落比同类明显',
        'advice_low': '节奏太平均。留出"安静—爆发"的落差，落差本身比密度更抓人',
        'advice_high': '落差明显。注意冷场段别太长，长冷场会掉观众',
        'p3': None,
    },
    'peak_cnt': {
        'label': '几个爆点', 'tech': 'peak_cnt', 'unit': '个',
        'display': True, 'direction': 'neutral', 'metric_kind': 'count',
        'seg': (0, 100), 'seg_name': '整片',
        'high_text': '多点开花', 'low_text': '只有一个爆点',
        'compare_low': '全片只检测到 {v} 个爆点（同类中位 {m} 个）',
        'compare_high': '全片检测到 {v} 个爆点，比同类多（同类中位 {m} 个）',
        'advice_low': '只有一个爆点，后半段容易掉。按同类多点开花的做法，每隔一段给一个小高潮',
        'advice_high': '爆点多，分段节奏好，保持',
        'p3': None,
    },
    'ending_5pct': {
        'label': '收尾承接力', 'tech': 'ending_5pct', 'unit': '占比',
        'display': True, 'direction': 'neutral', 'metric_kind': 'share',
        'seg': (95, 100), 'seg_name': '最后 5%',
        'high_text': '结尾还有人在刷', 'low_text': '结尾掉得快',
        'compare_low': '最后 5% 的弹幕只有同类的 {r}%',
        'compare_high': '最后 5% 的弹幕是同类的 {r}%',
        'advice_low': '结尾掉得快。收尾给一个回扣或彩蛋，把最后一波弹幕留住',
        'advice_high': '收尾有人接，说明结尾设计有效',
        'p3': None,
    },
    'mid_ratio': {
        'label': '中段留存', 'tech': 'mid_ratio', 'unit': '占比',
        'display': True, 'direction': 'neutral', 'metric_kind': 'share',
        'seg': (5, 95), 'seg_name': '中段 5–95%',
        'high_text': '中段撑住了', 'low_text': '中段掉链子',
        'compare_low': '中段弹幕只有同类的 {r}%——中间掉得比较快',
        'compare_high': '中段弹幕是同类的 {r}%——中段撑得住',
        'advice_low': '中段掉得快。中段是流失最狠的位置，考虑缩短铺垫或插入一个小转折',
        'advice_high': '中段留得住人，结构没问题',
        'p3': None,
    },
}

# 诚实边界：所有行动建议的统一后缀（33 号 §4 军规 3：不做因果承诺）
ACTION_CAVEAT = '（这是同类的做法参照，不是"改了就会火"的承诺）'
CURVE_CAVEAT = '形状差异与后续热度没有稳定关联（32 号 P6：量>形状），所以这类差异按"风格"看，不按"对错"看。'


def label_of(dim):
    return DICT.get(dim, {}).get('label', dim)


def display_dims():
    return [k for k, v in DICT.items() if v.get('display')]



# 关切极：这一维往哪一边偏才是"值得改"的（None = 纯方位差异，只报事实不报对错）
CONCERN = {
    'dm_per_min': 'low', 'wdm_per_min': 'low', 'avg_text_len': 'low',
    'opening_5pct': 'low', 'early_lag_ratio': 'low', 'ending_5pct': 'low',
    'mid_ratio': 'low', 'rhythm_entropy': 'high', 'burst_cv': 'low',
    'peak_cnt': 'low', 'main_peak_pct': 'both',   # both = 两个方向都值得注意（见词典 §2）
}


def human_pos(pct):
    """百分位 → 人话方位（33 号军规 2：不说小数说方位）"""
    if pct is None:
        return '同类里位置未知'
    ahead = round(100 - pct)
    if ahead <= 0:
        return '同类最高'
    if ahead <= 50:
        return '同类前 %d%%' % ahead
    if pct <= 1:
        return '同类最低'
    return '同类后 %d%%' % round(pct)


def pos_of(pct):
    """带谓语的方位短语"""
    h = human_pos(pct)
    return ('排' + h) if h.startswith('同类') and '未知' not in h else h


# 方位词：默认 偏低/偏高；进度型维度用 偏前/偏后
POLE_WORD = {'main_peak_pct': ('偏前', '偏后')}


def pole_word(dim, z):
    lo, hi = POLE_WORD.get(dim, ('偏低', '偏高'))
    return hi if z > 0 else lo


def sim_label(p):
    if p is None:
        return '节奏相似'
    if p >= 80:
        return '节奏非常像'
    if p >= 65:
        return '节奏比较像'
    if p >= 50:
        return '节奏有点像'
    return '节奏略有相似'


# 分区"性格"短词（用于"这个分区的观众习惯"一句话；主视图用）
TRAIT = {
    'opening_5pct':    ('开场就炸', '开场慢热'),
    'mid_ratio':       ('中段稳', '中段掉'),
    'ending_5pct':     ('收尾有人接', '收尾掉得快'),
    'rhythm_entropy':  ('弹幕铺得平', '高潮很集中'),
    'burst_cv':        ('大起大落', '四平八稳'),
    'peak_cnt':        ('多点开花', '爆点少'),
    'early_lag_ratio': ('观众来得快', '观众来得慢'),
    'dm_per_min':      ('热闹', '冷清'),
    'avg_text_len':    ('观众爱打字', '只刷短弹幕'),
    'main_peak_pct':   ('爆点靠后', '爆点靠前'),
}


def trait_of(dim, z):
    # TRAIT 的顺序是 (高, 低)——别解包反了（曾因此把"热闹"的分区判成"冷清"）
    hi, lo = TRAIT.get(dim, (dim, dim))
    return hi if z > 0 else lo

