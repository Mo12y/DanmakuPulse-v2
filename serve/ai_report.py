# -*- coding: utf-8 -*-
'''AI 报告解读 —— 把诊断能力产品化到**可交付形态**（受众：UP 主）。

定位（答辩口径，照实说，不要改说法）：
  诊断卡给人**看**，报告给人**转述**。这就是这一模块存在的理由。
  ⚠️ 答辩别说"别人都有所以我们也做"（会被追问创新点在哪），也别说"全新的创新点"
  （各行各业都有 ≠ 新）。
  它**不是**"去 AI 味"的对象 —— 去 AI 味只指设计层（措辞 / 视觉），不指"不许有 AI 功能"。

⚠️ 付费闸（铁律·付费服务先设闸；2026-09-22 曾一次失控约 ¥30，3 小时 56 分、1024+ 次调用）
  三条**同时**生效，不可放松：
    ① 单日 <= 200 次   ② 累计 <= ¥10   ③ 单次 max_tokens <= 600
  并且：同 (bvid, 快照) **只解读一次**（缓存，缓存命中不计费）；
        超限**自动停**（把开关置回关闭并记日志）；**默认关闭**，管理员在后台开；
        密钥走 env（compose/.env.llm，**.gitignore 已忽略、不进 git**）。
  成本量级（估计值，以实测为准）：一次约 2000 输入 + 400 输出 token ≈ ¥0.003。
  ⚠️ 监控盯的是"累计花费"而不是"进度"——超阈值自动停。

提示词硬约束（写死在 SYSTEM 里，不可放松）：
  ① 只能基于给出的检测结果解读，**不添加数据之外的因果推断或数字**；
  ② 观察点用"可参考"，**禁用**"一定会 / 导致 / 因为…所以建议"；
  ③ 不判定情绪、不评价内容质量（数据里也没有这些字段）。
  输出后再过一遍禁用词校验：**命中就拒绝展示并记日志**（不许静默换成别的文案）。

数据源（2026-10-20 切换）：输入 = serve/highlights.py 的 hl.card() 载荷（共振峰 / 质疑段 /
从众-质疑占比 / 分区对标 / 切片建议 / 诚实边界），**不再使用旧诊断卡**（verdict / position /
deviations / benchmarks）。缓存键带 PROMPT_VERSION，旧口径报告不会被命中。
'''
import json
import os
import re

import psycopg

try:
    import requests
except ImportError:            # 容器 .py 依赖见 serve/requirements.txt（已含 requests）
    requests = None

# 提示词/数据源版本：写进缓存键，换口径就换版本，旧缓存自然失效（不改表结构）
# v3（2026-10-20 用户拍板）：v2 真实冒烟发现模型把字段名 pctl 与统计数字念进正文；
# 加一条"只许引时间点、禁字段名/统计数字"的约束。
# v4（2026-10-22 用户拍板）：时间一律写 mm:ss（如 22:30–22:40），不再写「第 1350–1360 秒」；
# 与前端 rangeText 同口径。换提示词必须换版本，否则旧缓存会被命中。
PROMPT_VERSION = 'hl-ai-v4'

# ── 三条闸（改这里就是改闸，别在别处再定义一份）──
# ── 三条闸的**默认值**（用户 2026-10-16 拍板：100 次/日、¥10 上限、单次 600 token）──
# 实际生效值存在 ai_config 表里，**管理员可在「后台管理 · 采集控制台」调**（见 _limits()）。
DAY_LIMIT = 100                     # ① 单日调用上限默认值
BUDGET_CNY = 10.0                   # ② 累计花费上限默认值（元）
MAX_TOKENS = 600                    # ③ 单次输出 token 上限默认值
# 硬天花板：管理员在界面上只能**收紧**，不能越过 —— 否则「硬上限」就是摆设。
# 要突破只能改这里 + 重新部署，改动留在 git 里可追溯。
DAY_LIMIT_MAX = 200
BUDGET_CNY_MAX = 10.0
MAX_TOKENS_MAX = 600
DEFAULT_ENABLED = False             # 默认关闭：管理员显式开启后才可用
EST_COST_CNY = 0.003                # 单次成本量级（估计值；有 usage 时按实际 token 算）

# 禁用词：断言"因果 / 必然"的措辞。命中即拒绝展示并记日志（规格要求，不许改成别的文案）。
BANNED_WORDS = ('一定会', '必然会', '肯定会', '一定能', '必然导致', '必然', '导致', '造成',
                '所以建议', '因此建议', '这就说明', '证明了',
                # 2026-10-20：只加**具体断言词**。像「内容质量 / 情绪类型 / 情绪判断」这类
                # 边界名词**不加**——模型复述"不判定情绪 / 不评价内容质量"时会被误杀；
                # 边界由 SYSTEM 约束，校验只抓具体断言。
                '感人', '好笑', '愤怒', '燃', '节奏健康', '没有明显短板',
                '内容最好', '名场面', '爆款', '一定会火')
BANNED_RE = re.compile(r'因为[^。！？\n]{0,40}?(所以|因此|故)\s*建议')

DDL = '''
CREATE TABLE IF NOT EXISTS ai_config (
  id         int PRIMARY KEY,
  enabled    boolean NOT NULL DEFAULT false,
  day_limit  int NOT NULL DEFAULT 100,
  budget_cny numeric(10,2) NOT NULL DEFAULT 10.00,
  max_tokens int NOT NULL DEFAULT 600,
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT ai_config_single CHECK (id = 1)
);
INSERT INTO ai_config (id, enabled) VALUES (1, false) ON CONFLICT (id) DO NOTHING;
-- 老库补列（表已存在时 CREATE IF NOT EXISTS 不会加列）
ALTER TABLE ai_config ADD COLUMN IF NOT EXISTS day_limit  int NOT NULL DEFAULT 100;
ALTER TABLE ai_config ADD COLUMN IF NOT EXISTS budget_cny numeric(10,2) NOT NULL DEFAULT 10.00;
ALTER TABLE ai_config ADD COLUMN IF NOT EXISTS max_tokens int NOT NULL DEFAULT 600;
CREATE TABLE IF NOT EXISTS ai_call (
  id         bigserial PRIMARY KEY,
  ts         timestamptz NOT NULL DEFAULT now(),
  bvid       text,
  cache_key  text,
  ok         boolean NOT NULL,
  reason     text,
  cost_cny   double precision NOT NULL DEFAULT 0,
  cached     boolean NOT NULL DEFAULT false
);
CREATE TABLE IF NOT EXISTS ai_report (
  bvid        text NOT NULL,
  snap_id     text NOT NULL,
  batch_date  text,
  created_at  timestamptz NOT NULL DEFAULT now(),
  model       text,
  report      text NOT NULL,
  suggestions text,
  PRIMARY KEY (bvid, snap_id)
);
'''

SYSTEM = '''你在为一个「B站视频弹幕精彩时刻检测」工具写一段简短说明。
你只会收到一份结构化检测结果（JSON）。硬约束，不可违反：
1. 只能使用给定数据，不得编造数字、因果或视频内容（没有画面、没有内容标签）。
2. 不判定情绪：禁止出现「感人 / 好笑 / 愤怒 / 情绪类型 / 这一秒是什么情绪」等标签；数据里也没有这些字段。
3. 不评价内容质量：禁止「节奏健康 / 没有明显短板 / 内容最好 / 名场面 / 爆款 / 一定会火」等判断。
4. 不给因果处方：禁止「一定会 / 导致 / 造成 / 因为…所以建议」。
5. 【报告解读】2–3 句，只讲这支视频的共振峰/质疑段分布有什么值得注意；不要逐项复述页面已有的数字；可引用关键时间点。
6. 【可参考的观察点】2–3 条，每条一句话：说清「可参考看 22:30–22:40（共振/质疑），那里观众在刷什么、在同类里处于什么位置」；时间一律写成 mm:ss（如 03:05–03:15；不足 1 分钟可写「第 45–50 秒」），**不要写「第 1350–1360 秒」这种需要心算的秒数**；不要建议改内容、不要承诺效果；长视频可点明「可作为切片候选」，但必须补一句「这只是共振最强处，不代表内容更好」。
7. 数据里没有的写「暂无」，不许编。
8. 如果原始弹幕文本里出现被禁词，不要直接引用原文，改用「（原文含敏感词，略）」或概括。
9. 可以引用时间点，但必须是 mm:ss（如 22:30–22:40）；禁止出现任何字段名（pctl、score、n_repeat 等）与统计数字（百分位、样本数、复读条数、得分）；相对强弱用自然语言说（如「共振最强的一段」），不要念原始数值。
输出格式（严格照此，两种标题必须原样出现）：
【报告解读】
2–3 句。
【可参考的观察点】
1. 可参考…
2. 可参考…
'''


def _dsn():
    return os.environ['ADS_PG_DSN']


def _q(sql, params=(), fetch=True):
    with psycopg.connect(_dsn()) as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        if not fetch:
            conn.commit()
            return None
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _q1(sql, params=()):
    r = _q(sql, params)
    return r[0] if r else None


def ensure_tables():
    '''幂等建表；库没就绪只 warning，不阻塞服务启动（同 auth.ensure_tables 的做法）'''
    try:
        with psycopg.connect(_dsn()) as conn, conn.cursor() as cur:
            cur.execute(DDL)
            conn.commit()
        return True
    except Exception as e:
        print('[ai] AI 解读表初始化失败（不阻塞启动）：%s' % e, flush=True)
        return False


# ── 配置与用量 ──────────────────────────────────────────────

def _env():
    key = (os.environ.get('LLM_API_KEY') or '').strip()
    base = (os.environ.get('LLM_BASE_URL') or 'https://api.deepseek.com').rstrip('/')
    model = (os.environ.get('LLM_MODEL') or 'deepseek-chat').strip()
    return key, base, model


def configured():
    return bool(_env()[0])


def is_enabled():
    r = _q1('SELECT enabled FROM ai_config WHERE id = 1')
    return bool(r and r['enabled'])


def set_enabled(flag):
    _q('UPDATE ai_config SET enabled = %s, updated_at = now() WHERE id = 1',
       (bool(flag),), fetch=False)
    return is_enabled()


def set_enabled_off(reason):
    '''超限自动停：把开关置回关闭，并把原因记进日志'''
    _q('UPDATE ai_config SET enabled = false, updated_at = now() WHERE id = 1', fetch=False)
    _q('INSERT INTO ai_call (ok, reason, cost_cny, cached) VALUES (false, %s, 0, false)',
       ('自动停：' + reason,), fetch=False)


def _limits():
    """读管理员设定的三个闸；读不到就退回默认，并且**一律夹到硬天花板内**。

    为什么要夹：界面上的输入框是给管理员调松紧用的，不是用来突破硬上限的 ——
    硬上限一旦能在界面上改，它就不是硬上限（项目曾因无闸失控 ¥30）。
    """
    d, b, m = DAY_LIMIT, BUDGET_CNY, MAX_TOKENS
    r = _q1('SELECT day_limit, budget_cny, max_tokens FROM ai_config WHERE id = 1')
    if r:
        if r.get('day_limit'):
            d = int(r['day_limit'])
        if r.get('budget_cny') is not None:
            b = float(r['budget_cny'])
        if r.get('max_tokens'):
            m = int(r['max_tokens'])
    return (max(1, min(d, DAY_LIMIT_MAX)),
            max(0.5, min(b, BUDGET_CNY_MAX)),
            max(100, min(m, MAX_TOKENS_MAX)))


def set_limits(day_limit=None, budget_cny=None, max_tokens=None):
    """管理员调整闸值。越界直接拒绝（不静默夹值），并把改动记进 ai_call 日志。"""
    cur = _limits()
    d = cur[0] if day_limit is None else int(day_limit)
    b = cur[1] if budget_cny is None else float(budget_cny)
    m = cur[2] if max_tokens is None else int(max_tokens)
    if not (1 <= d <= DAY_LIMIT_MAX):
        raise ValueError('单日次数需在 1..%d 之间' % DAY_LIMIT_MAX)
    if not (0.5 <= b <= BUDGET_CNY_MAX):
        raise ValueError('金额上限需在 0.5..%.0f 元之间' % BUDGET_CNY_MAX)
    if not (100 <= m <= MAX_TOKENS_MAX):
        raise ValueError('单次 token 需在 100..%d 之间' % MAX_TOKENS_MAX)
    _q('UPDATE ai_config SET day_limit = %s, budget_cny = %s, max_tokens = %s, updated_at = now() '
       'WHERE id = 1', (d, b, m), fetch=False)
    _q('INSERT INTO ai_call (ok, reason, cost_cny, cached) VALUES (false, %s, 0, false)',
       ('管理员调整闸：%d 次 / ¥%.2f / %d token' % (d, b, m),), fetch=False)
    return _limits()


def _used_today():
    r = _q1("SELECT COUNT(*) AS n FROM ai_call WHERE ok AND NOT cached AND ts::date = CURRENT_DATE")
    return int(r['n']) if r else 0


def _spent():
    r = _q1('SELECT COALESCE(SUM(cost_cny), 0) AS s FROM ai_call')
    return round(float(r['s']), 4) if r else 0.0


def usage():
    key, base, model = _env()
    return {
        'enabled': is_enabled(),
        'configured': bool(key),
        'model': model if key else None,
        'used_today': _used_today(),
        'day_limit': _limits()[0],
        'spent_cny': _spent(),
        'budget_cny': _limits()[1],
        'max_tokens': _limits()[2],
        # 硬天花板：界面上的输入框不能越过它
        'day_limit_max': DAY_LIMIT_MAX,
        'budget_cny_max': BUDGET_CNY_MAX,
        'max_tokens_max': MAX_TOKENS_MAX,
        'default_enabled': DEFAULT_ENABLED,
        'key_source': 'compose/.env.llm（.gitignore 已忽略，不进 git）',
    }


def limits_tripped(autostop=True):
    '''返回被触发的闸的原因；没触发返回 None。触发时**自动停**。'''
    n, s = _used_today(), _spent()
    day_limit, budget = _limits()[0], _limits()[1]
    reason = None
    if n >= day_limit:
        reason = '单日调用已达上限 %d 次' % day_limit
    elif s >= budget:
        reason = '累计花费已达上限 ¥%.2f（当前 ¥%.4f）' % (budget, s)
    if reason and autostop and is_enabled():
        set_enabled_off(reason)
    return reason


# ── 输出校验（禁用词）──────────────────────────────────────

def check_output(text):
    '''返回命中的禁用词/禁用构造；空列表 = 通过。命中即拒绝展示（规格要求，不许改文案）。'''
    hits = [w for w in BANNED_WORDS if w in (text or '')]
    if BANNED_RE.search(text or ''):
        hits.append('"因为…所以建议"构造')
    return hits


# ── 提示词与调用 ────────────────────────────────────────────

def _pairs(list_, n=3):
    """top_danmaku 统一成 [{text, n}]；字符串形态也兼容（不喂整段原文以外的结构）"""
    out = []
    for it in (list_ or [])[:n]:
        if isinstance(it, dict):
            out.append({'text': it.get('text'), 'n': it.get('n')})
        else:
            out.append({'text': str(it), 'n': None})
    return out


def highlights_payload(d):
    """把 hl.card()['data'] 里**给人看的那部分**结构化出来当提示词输入。

    刻意不喂：旧诊断卡的 verdict / position / deviations / benchmarks（已换数据源）、
    曲线 101 个点（太长）、任何情绪字段（数据源里也没有）、tech 层参数。
    ⚠️ highlights 自己的 `benchmark`（单数，分区共振对标）**必须保留** —— 别和旧卡
    的 `benchmarks`（复数）搞混，删错就丢了"在同类里处于什么位置"。
    """
    d = d or {}
    # 峰：按复读强度取前 5（页面全量 41 个峰时提示词会被数字淹没）
    peaks = sorted((d.get('peaks') or []),
                   key=lambda p: (-(p.get('score') or 0), p.get('start_sec') or 0))[:5]
    crit = d.get('critique') or {}
    segs = (crit.get('segments') or [])[:3]
    fb = d.get('feedback') or {}
    b = d.get('benchmark') or {}
    picks = (d.get('slice_suggestions') or d.get('highlight_positions') or [])[:3]
    return {
        'bvid': d.get('bvid'),
        'title': d.get('title'),
        'tname': d.get('tname'),
        'duration': d.get('duration'),
        'is_long': d.get('is_long'),
        'peaks': [{'start_sec': p.get('start_sec'), 'end_sec': p.get('end_sec'),
                   'score': p.get('score'), 'n_repeat': p.get('n_repeat'),
                   'n_rep_texts': p.get('n_rep_texts'),
                   'top_danmaku': _pairs(p.get('top_danmaku')),
                   'pctl': p.get('pctl'), 'in_opening': bool(p.get('in_opening'))}
                  for p in peaks],
        'critique': {'detected': crit.get('detected'),
                     'n_segments': crit.get('n_segments'),
                     'segments': [{'start_sec': s.get('start_sec'),
                                   'end_sec': s.get('end_sec'),
                                   'score': s.get('score'), 'n': s.get('n'),
                                   'top_danmaku': _pairs(s.get('top_danmaku'))}
                                  for s in segs]},
        'feedback': {'herd': fb.get('herd'), 'neutral': fb.get('neutral'),
                     'critique': fb.get('critique'),
                     'n_danmaku': fb.get('n_danmaku'), 'n_herd': fb.get('n_herd'),
                     'n_neutral': fb.get('n_neutral'), 'n_critique': fb.get('n_critique')},
        'benchmark': {'tname': b.get('tname'), 'partition_n': b.get('partition_n'),
                      'partition_ok': b.get('partition_ok'),
                      'resonance_pctl': b.get('resonance_pctl'),
                      'med': b.get('partition_med')},
        'picks': [{'start_sec': p.get('start_sec'), 'end_sec': p.get('end_sec'),
                   'pctl': p.get('pctl'), 'opening_only': bool(p.get('opening_only')),
                   'top_danmaku': _pairs(p.get('top_danmaku'))}
                  for p in picks],
        'caveats': d.get('caveats'),
    }


def make_snap_id(batch_date, engine_version, dict_version):
    """缓存键：版本 + 批次 + 引擎 + 词典。

    ⚠️ **不要**用 meta.generated_at —— hl.card() 里它是每次请求 datetime.now() 生成的，
    拿它当键会让缓存永不命中（每次都当成新快照重新计费）。
    """
    return '%s|%s|%s|%s' % (PROMPT_VERSION, batch_date, engine_version, dict_version)


def build_messages(payload):
    return [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': '诊断数据（JSON，只能用这里的数字）：\n' +
         json.dumps(payload, ensure_ascii=False, indent=1)},
    ]


def _call_llm(messages):
    '''OpenAI 兼容协议的一次调用。max_tokens 硬钉在 MAX_TOKENS。'''
    key, base, model = _env()
    if requests is None:
        raise RuntimeError('容器里没有 requests（见 serve/requirements.txt）')
    r = requests.post(
        base + '/chat/completions',
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'},
        json={'model': model, 'messages': messages, 'max_tokens': _limits()[2], 'temperature': 0.3},
        timeout=45)
    if r.status_code != 200:
        raise RuntimeError('上游返回 %s：%s' % (r.status_code, (r.text or '')[:200]))
    j = r.json()
    text = ((j.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
    return text.strip(), (j.get('usage') or {}), model


def _cost_cny(usage_info):
    '''按实际 token 算，价格可配（env）；拿不到 usage 就退回单次估计值。
    监控口径 = **累计花费**，不是进度。'''
    try:
        pin = float(os.environ.get('LLM_PRICE_IN_PER_M') or 1.0)
        pout = float(os.environ.get('LLM_PRICE_OUT_PER_M') or 2.0)
    except ValueError:
        pin, pout = 1.0, 2.0
    pi = usage_info.get('prompt_tokens')
    po = usage_info.get('completion_tokens')
    if pi is None or po is None:
        return EST_COST_CNY
    return round(pi / 1e6 * pin + po / 1e6 * pout, 6)


def _split(text):
    '''按提示词约定的两个标题切出「报告解读」与「可参考的观察点」'''
    t = (text or '').strip()
    m = re.search(r'【可参考的观察点】|可参考的观察点\s*[:：]', t)
    if not m:
        return re.sub(r'^【报告解读】\s*', '', t).strip(), []
    rep = re.sub(r'^【报告解读】\s*', '', t[:m.start()].strip()).strip()
    tail = t[m.end():].strip()
    items = [re.sub(r'^\s*(?:[-*•]|\d+[.、)])\s*', '', x).strip()
             for x in re.split(r'\n+', tail) if x.strip()]
    return rep, items[:3]


def generate(bvid, snap_id, batch_date, hl_data):
    '''返回 (payload, status)。status ∈ {ok, cached, disabled, unconfigured, limited, blocked}。

    hl_data = highlights.card(bvid)['data']（不是旧诊断卡）。
    '''
    # ① 未配置密钥：明确说"未配置"，不是报错
    if not configured():
        raise PermissionError('AI 解读未配置：需要在 compose/.env.llm 里放 LLM_API_KEY（不进 git）。')
    # ② 默认关闭：管理员开了才跑
    if not is_enabled():
        raise PermissionError('AI 解读默认关闭。管理员可在「后台管理 · 采集控制台」开启。')
    # ③ 缓存：同 (bvid, 快照) 只解读一次 —— 命中不重复计费
    hit = _q1('SELECT * FROM ai_report WHERE bvid = %s AND snap_id = %s', (bvid, snap_id))
    if hit:
        _q('INSERT INTO ai_call (bvid, cache_key, ok, reason, cost_cny, cached) '
           'VALUES (%s,%s,true,%s,0,true)', (bvid, snap_id, '缓存命中'), fetch=False)
        return ({'report': hit['report'],
                 'suggestions': [x for x in (hit['suggestions'] or '').split('\n') if x.strip()],
                 'model': hit['model'],
                 'created_at': hit['created_at'].strftime('%Y-%m-%d %H:%M')},
                'cached')
    # ④ 三条闸（超限自动停）
    reason = limits_tripped()
    if reason:
        raise OverflowError('已自动停：' + reason + '（本次不调用，不产生费用）')

    messages = build_messages(highlights_payload(hl_data))
    try:
        text, ui, model = _call_llm(messages)
    except Exception as e:
        _q('INSERT INTO ai_call (bvid, cache_key, ok, reason, cost_cny, cached) '
           'VALUES (%s,%s,false,%s,0,false)', (bvid, snap_id, '上游失败：' + str(e)[:200]),
           fetch=False)
        raise RuntimeError('模型调用失败：%s' % e)

    cost = _cost_cny(ui)
    # ⑤ 禁用词校验：命中 -> 拒绝展示 + 记日志（不计费入缓存，但**记录真实花费**）
    hits = check_output(text)
    if hits:
        _q('INSERT INTO ai_call (bvid, cache_key, ok, reason, cost_cny, cached) '
           'VALUES (%s,%s,false,%s,%s,false)',
           (bvid, snap_id, '命禁用词拒绝展示：' + '、'.join(hits), cost), fetch=False)
        # 报错文案不得包含费用金额（费用只写 ai_call 日志，展示层不露）
        raise ValueError('模型输出命中禁用词（%s），已拒绝展示并记入日志。' % '、'.join(hits))

    report, suggestions = _split(text)
    if not report:
        _q('INSERT INTO ai_call (bvid, cache_key, ok, reason, cost_cny, cached) '
           'VALUES (%s,%s,false,%s,%s,false)', (bvid, snap_id, '输出为空', cost), fetch=False)
        raise RuntimeError('模型返回空内容。')

    _q('INSERT INTO ai_report (bvid, snap_id, batch_date, model, report, suggestions) '
       'VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (bvid, snap_id) DO NOTHING',
       (bvid, snap_id, batch_date, model, report, '\n'.join(suggestions)), fetch=False)
    _q('INSERT INTO ai_call (bvid, cache_key, ok, reason, cost_cny, cached) '
       'VALUES (%s,%s,true,%s,%s,false)', (bvid, snap_id, '生成成功', cost), fetch=False)
    return ({'report': report, 'suggestions': suggestions, 'model': model,
             'created_at': None, 'cost_cny': cost,
             'prompt_tokens': ui.get('prompt_tokens'), 'completion_tokens': ui.get('completion_tokens')},
            'ok')
