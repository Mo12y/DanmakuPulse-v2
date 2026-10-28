#!/usr/bin/env python3
"""
m7_gate_eval.py — 「早期弹幕 + 元数据 → 互动质量」门控判定（2026-10-22）

输入：data-out/m7_gate_features_csv/*.csv（由 jobs/m7_gate_features.py 产出）
输出：data-out/ml_gate_interaction_quality_report.json
      data-out/ml_gate_interaction_quality_report.md
      data-out/m7_gate_roc_pr.png

判据（**写死，不达标就是不过**）：
  测试集 AUC ≥ 0.65，且比「只用元数据」基线高 ≥ 0.05
  （标签按分区内排名分位切；切分按批次时间，不随机）

口径要点：
  · 三个标签都算（互动率 ir / 三连率 tri / 投币率 coinr），**按分区内变异系数预注册**选主标签，
    其余两个只报 AUC 作参照，不做事后换标签。
  · 分区内分位参考分布 = **训练期（早期批次）**，测试期视频按同一参考分布定位；
    训练期行用留一法（避免自己算自己）。分区内训练样本 < MIN_REF 的分区不参与（无法排序）。
  · 特征里**没有** play/like/coin/favorite/share/danmaku/reply（标签分子分母与同期结果变量）。

用法：cd ~/projects/danmaku-v2 && ~/danmaku-venv/bin/python jobs/m7_gate_eval.py
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

OUT = 'data-out'
TRAIN_MAX_BATCH = 20260917        # 训练 = 20260902/0907/0914/0917
TEST_BATCH = 20261013             # 主测试 = 最新批次
ROBUST_BATCHES = (20261012, 20261013)
MIN_REF = 8                       # 分区内训练样本下限（低于此无法做分区内排序）
HIGH_PCT, LOW_PCT = 0.70, 0.30    # 主标签：分区内 top30% vs bottom30%
SEED = 42

EARLY_COLS = ['early_n_log', 'early_users_log', 'early_texts_log', 'early_repeat_rate',
              'early_peak_cnt', 'early_entropy', 'early_short_ratio', 'early_len_mean',
              'early_users_per_msg', 'early_per_min_log']
META_COLS = ['title_len', 'title_has_digit', 'title_has_q', 'title_has_excl',
             'duration_log', 'pub_hour', 'pub_wday']


def load():
    path = os.path.join(OUT, 'm7_gate_features.csv')
    if not os.path.exists(path):
        raise SystemExit('找不到特征 CSV：先跑 jobs/m7_gate_features.py')
    df = pd.read_csv(path)
    df['batch_date'] = df['batch_date'].astype(int)
    for c in ('play', 'like', 'coin', 'favorite', 'share'):
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    df = df[df['play'] > 0].copy()
    df['ir'] = (df['like'] + df['coin'] + df['favorite'] + df['share']) / df['play']
    df['tri'] = (df['like'] + df['coin'] + df['favorite']) / df['play']
    df['coinr'] = df['coin'] / df['play']

    df['early_n_log'] = np.log1p(df['early_n'].fillna(0))
    df['early_users_log'] = np.log1p(df['early_users'].fillna(0))
    df['early_texts_log'] = np.log1p(df['early_texts'].fillna(0))
    df['early_per_min_log'] = np.log1p(df['early_per_min'].fillna(0))
    df['duration_log'] = np.log1p(df['duration'].fillna(0))
    df['pub_hour'] = df['pub_hour'].fillna(-1)
    df['pub_wday'] = df['pub_wday'].fillna(-1)
    df['tname'] = df['tname'].fillna('未知')
    return df


def dispersion(df, col):
    """分区内离散度：只统计训练期 n>=MIN_REF 的分区；返回加权平均 CV（sd/mean）。"""
    ref = df[df['batch_date'] <= TRAIN_MAX_BATCH]
    rows = []
    for t, g in ref.groupby('tname'):
        if len(g) < MIN_REF:
            continue
        mu, sd = g[col].mean(), g[col].std(ddof=1)
        if mu and mu > 0:
            rows.append((len(g), sd / mu))
    if not rows:
        return None
    w = np.array([r[0] for r in rows], dtype=float)
    v = np.array([r[1] for r in rows], dtype=float)
    return float((w * v).sum() / w.sum())


def add_pctl(df, col):
    """分区内分位（参考=训练期；训练行留一法）。返回 Series（0–1），无法定位为 NaN。"""
    ref = df[df['batch_date'] <= TRAIN_MAX_BATCH]
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for t, g in ref.groupby('tname'):
        vals = np.sort(g[col].dropna().values)
        if len(vals) < MIN_REF:
            continue
        idx = df.index[df['tname'] == t]
        for i in idx:
            x = df.at[i, col]
            if pd.isna(x):
                continue
            lo = int(np.searchsorted(vals, x, side='left'))
            hi = int(np.searchsorted(vals, x, side='right'))
            if df.at[i, 'batch_date'] <= TRAIN_MAX_BATCH:
                # 留一：把自己从参考分布里去掉
                lo = max(0, lo - 1)
                hi = max(1, hi - 1)
                n = len(vals) - 1
            else:
                n = len(vals)
            if n <= 0:
                continue
            out.at[i] = (lo + hi) / 2.0 / n
    return out


def build_xy(df, label_col, feature_cols):
    sub = df.dropna(subset=['label']).copy()
    X = sub[feature_cols].astype(float).values
    y = sub['label'].astype(int).values
    return sub, X, y


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    d = {'n': int(len(y)), 'n_pos': int(y.sum()), 'n_neg': int(len(y) - y.sum())}
    if len(set(y)) < 2:
        d.update({'auc': None, 'pr_auc': None, 'f1': None, 'acc': None})
        return d
    d.update({'auc': float(roc_auc_score(y, p)),
              'pr_auc': float(average_precision_score(y, p)),
              'f1': float(f1_score(y, pred, zero_division=0)),
              'acc': float(accuracy_score(y, pred))})
    return d


def make_models():
    lr = lambda: Pipeline([('sc', StandardScaler()),
                           ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))])
    return {
        'majority': DummyClassifier(strategy='most_frequent'),
        'lr_meta': lr(),
        'lr_full': lr(),
        'gbt_full': GradientBoostingClassifier(n_estimators=200, max_depth=2,
                                               learning_rate=0.05, random_state=SEED),
    }


def cv_auc_train(tr, cols, seed=SEED):
    """训练期 5 折 CV AUC：**只用于选标签**，不看测试集（避免用测试集挑标签）。"""
    from sklearn.model_selection import StratifiedKFold
    y = tr['label'].values.astype(int)
    if len(set(y)) < 2 or len(y) < 60:
        return None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for trn, val in skf.split(tr[cols].values, y):
        mdl = Pipeline([('sc', StandardScaler()),
                        ('lr', LogisticRegression(max_iter=2000, C=1.0, random_state=seed))])
        mdl.fit(tr[cols].values[trn], y[trn])
        if len(set(y[val])) < 2:
            continue
        aucs.append(roc_auc_score(y[val], mdl.predict_proba(tr[cols].values[val])[:, 1]))
    return float(np.mean(aucs)) if aucs else None


def bootstrap_auc(y, p, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[b])) < 2:
            continue
        vals.append(roc_auc_score(y[b], p[b]))
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def run_label(df, label_col, metric_col):
    """对一个标签跑 4 个模型；返回 dict（含测试集指标、切片、重要性）。"""
    d = df.copy()
    d['label'] = np.nan
    pctl = add_pctl(d, metric_col)
    d['pctl'] = pctl
    d.loc[pctl >= HIGH_PCT, 'label'] = 1
    d.loc[pctl <= LOW_PCT, 'label'] = 0
    feat_meta = META_COLS + [c for c in d.columns if c.startswith('tn_')]
    feat_full = feat_meta + EARLY_COLS

    tr = d[(d['batch_date'] <= TRAIN_MAX_BATCH) & d['label'].notna()]
    te = d[(d['batch_date'] == TEST_BATCH) & d['label'].notna()]
    te2 = d[d['batch_date'].isin(ROBUST_BATCHES) & d['label'].notna()]
    if len(tr) < 30 or len(te) < 20:
        return {'error': '样本不足', 'n_train': int(len(tr)), 'n_test': int(len(te))}

    models = make_models()
    res = {'label': label_col, 'n_train': int(len(tr)), 'n_test': int(len(te)),
           'cv_auc_train': cv_auc_train(tr, feat_full),
           'n_test_robust': int(len(te2)),
           'pos_rate_train': float(tr['label'].mean()), 'models': {}}
    probs = {}
    for name, mdl in models.items():
        if name == 'majority':
            mdl.fit(tr[feat_meta].values, tr['label'].values)
            p = np.zeros(len(te))
        elif name == 'lr_meta':
            mdl.fit(tr[feat_meta].values, tr['label'].values)
            p = mdl.predict_proba(te[feat_meta].values)[:, 1]
        else:
            mdl.fit(tr[feat_full].values, tr['label'].values)
            p = mdl.predict_proba(te[feat_full].values)[:, 1]
        res['models'][name] = metrics(te['label'].values.astype(int), p)
        probs[name] = p

    # 稳健性：最后两批
    for name in ('lr_meta', 'lr_full'):
        mdl = models[name]
        cols = feat_meta if name == 'lr_meta' else feat_full
        p2 = mdl.predict_proba(te2[cols].values)[:, 1]
        res['models'][name]['robust_last2'] = metrics(te2['label'].values.astype(int), p2)

    # 特征重要性
    lr_full = models['lr_full'].named_steps['lr']
    coef = pd.Series(lr_full.coef_[0], index=feat_full).abs().sort_values(ascending=False)
    gbt = models['gbt_full'].feature_importances_
    imp = pd.Series(gbt, index=feat_full).sort_values(ascending=False)
    res['importance'] = {
        'lr_abs_coef_top10': {k: round(float(v), 4) for k, v in coef.head(10).items()},
        'gbt_top10': {k: round(float(v), 4) for k, v in imp.head(10).items()},
    }

    # 切片：分区 / 时长（测试集，n>=15 才报）
    res['slices'] = {'tname': {}, 'duration_q': {}}
    te = te.copy()
    te['p_full'] = probs['lr_full']
    te['p_meta'] = probs['lr_meta']
    for t, g in te.groupby('tname'):
        if len(g) >= 15 and g['label'].nunique() == 2:
            res['slices']['tname'][t] = {
                'n': int(len(g)),
                'auc_full': round(float(roc_auc_score(g['label'], g['p_full'])), 4),
                'auc_meta': round(float(roc_auc_score(g['label'], g['p_meta'])), 4),
            }
    try:
        te['dq'] = pd.qcut(te['duration'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        for q, g in te.groupby('dq', observed=True):
            if len(g) >= 15 and g['label'].nunique() == 2:
                res['slices']['duration_q'][str(q)] = {
                    'n': int(len(g)),
                    'auc_full': round(float(roc_auc_score(g['label'], g['p_full'])), 4),
                }
    except ValueError:
        pass

    res['_probs'] = {'y': te['label'].values.astype(int), 'p_full': probs['lr_full'],
                     'p_meta': probs['lr_meta'], 'p_gbt': probs['gbt_full']}
    return res


def plot_roc_pr(res, path):
    y = res['_probs']['y']
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), dpi=150)
    for key, lab in (('p_meta', 'metadata only'), ('p_full', 'metadata + early danmaku'),
                     ('p_gbt', 'GBT (metadata + early)')):
        p = res['_probs'][key]
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label='%s AUC=%.3f' % (lab, roc_auc_score(y, p)))
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, label='%s AP=%.3f' % (lab, average_precision_score(y, p)))
    axes[0].plot([0, 1], [0, 1], '--', color='grey', linewidth=0.8)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR'); axes[0].set_title('ROC (test)')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision'); axes[1].set_title('PR (test)')
    for a in axes:
        a.legend(fontsize=7); a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    df = load()
    # 分区 one-hot（只保留训练期出现 >=5 次的分区，其余归 other）
    ref = df[df['batch_date'] <= TRAIN_MAX_BATCH]
    keep = [t for t, n in ref['tname'].value_counts().items() if n >= 5][:20]
    for t in keep:
        df['tn_' + str(t)] = (df['tname'] == t).astype(float)
    df['tn_other'] = (~df['tname'].isin(keep)).astype(float)

    disp = {c: dispersion(df, c) for c in ('ir', 'tri', 'coinr')}
    results = {}
    for col in ('ir', 'tri', 'coinr'):
        results[col] = run_label(df, col, col)
        print(col, 'done')

    # 选标签（**只看训练期，测试集不参与**）：
    #   主标签默认 = ir（任务书写死）；备选要替换，必须同时满足
    #     ① 训练期 5 折 CV AUC >= 0.60（"可分"的最低证据）
    #     ② 比 ir 的 CV AUC 高 >= 0.05
    #   为什么不用"方差最大"：投币率分母小（中位 0.0068），分区内 CV 1.19 全靠噪声撑起来，
    #   元数据基线直接 0.736 —— 方差大 ≠ 可分（2026-10-22 首跑踩过，如实记录）。
    cv = {k: v.get('cv_auc_train') for k, v in results.items()}
    primary = 'ir'
    alts = [k for k in ('tri', 'coinr')
            if cv.get(k) is not None and cv.get(k) >= 0.60
            and cv.get('ir') is not None and cv[k] - cv['ir'] >= 0.05]
    if alts:
        primary = max(alts, key=lambda k: cv[k])
    print('分区内 CV：', {k: (round(v, 4) if v else None) for k, v in disp.items()})
    print('训练期 5 折 CV AUC：', {k: (round(v, 4) if v else None) for k, v in cv.items()},
          '→ 主标签', primary)

    main_res = results[primary]
    if 'models' not in main_res:
        # 样本不足：不硬跑，如实写"未过门控（样本不足）"
        main_res = {'error': main_res.get('error'), 'n_train': main_res.get('n_train'),
                    'n_test': main_res.get('n_test'), 'models': {}, 'importance': {},
                    'slices': {'tname': {}, 'duration_q': {}}, '_probs': {'y': []}}
        results[primary] = main_res
    keep_keys = {k: v for k, v in main_res.items() if not k.startswith('_')}
    plot_roc_pr(main_res, os.path.join(OUT, 'm7_gate_roc_pr.png'))

    m = main_res['models']
    auc_full = m['lr_full']['auc']
    auc_meta = m['lr_meta']['auc']
    y_te = main_res['_probs']['y']
    ci_full = bootstrap_auc(y_te, main_res['_probs']['p_full']) if len(y_te) else (None, None)
    ci_meta = bootstrap_auc(y_te, main_res['_probs']['p_meta']) if len(y_te) else (None, None)
    delta_ci = (None, None)
    if len(y_te):
        rng = np.random.default_rng(SEED)
        ds = []
        idx = np.arange(len(y_te))
        pf, pm = main_res['_probs']['p_full'], main_res['_probs']['p_meta']
        for _ in range(1000):
            b = rng.choice(idx, size=len(idx), replace=True)
            if len(set(y_te[b])) < 2:
                continue
            ds.append(roc_auc_score(y_te[b], pf[b]) - roc_auc_score(y_te[b], pm[b]))
        if ds:
            delta_ci = (float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)))
    gate = {
        'auc_full': auc_full,
        'auc_full_ci95': ci_full,
        'auc_meta': auc_meta,
        'auc_meta_ci95': ci_meta,
        'delta': (None if (auc_full is None or auc_meta is None) else round(auc_full - auc_meta, 4)),
        'pass_auc': bool(auc_full is not None and auc_full >= 0.65),
        'pass_delta': bool(auc_full is not None and auc_meta is not None and auc_full - auc_meta >= 0.05),
    }
    gate['delta_ci95'] = delta_ci
    gate['passed_written'] = bool(gate['pass_auc'] and gate['pass_delta'])
    # 稳定性守卫（2026-10-22 新增，**加严不加宽**）：
    #   ① 训练期 CV AUC >= 0.55（同期内至少学到一点）
    #   ② ΔAUC 的 95% CI 下界 > 0（增量不是小样本噪声）
    #   两条任一不过 → 不采信字面通过，按"未过门控"收口（不美化回测）。
    cv_primary = main_res.get('cv_auc_train')
    gate['cv_auc_train'] = cv_primary
    gate['stable_cv'] = bool(cv_primary is not None and cv_primary >= 0.55)
    gate['stable_delta'] = bool(delta_ci[0] is not None and delta_ci[0] > 0)
    gate['passed_stable'] = bool(gate['passed_written'] and gate['stable_cv'] and gate['stable_delta'])
    # 对外只留一个"过/没过"：字面判据单独记在 passed_written，避免 JSON 里两个 passed 打架
    gate['passed'] = gate['passed_stable']
    gate['verdict'] = '通过' if gate['passed_stable'] else '未过门控'
    gate['guard_note'] = ('稳定性守卫为 2026-10-22 追加（训练期 CV AUC>=0.55、ΔAUC 95%CI 下界>0）：'
                          '只按任务书两条门槛，本数据会在 n=83 上"字面通过"，但同期内学不到信号、'
                          'Δ 的置信区间跨 0，故不采信。')

    # 标签分布（分区内）：写进报告，答辩时能直接回答"高低是怎么切的"
    # run_label 内部 copy，不改 df；这里按主标签重算一次分位用于分布统计
    tmp = df.copy()
    tmp['pctl'] = add_pctl(tmp, primary)
    tmp['lab'] = np.where(tmp['pctl'] >= HIGH_PCT, 1, np.where(tmp['pctl'] <= LOW_PCT, 0, np.nan))
    dist_rows = []
    for t, g in tmp[tmp['lab'].notna()].groupby('tname'):
        dist_rows.append({'tname': t, 'n': int(len(g)), 'pos_rate': round(float(g['lab'].mean()), 4)})
    dist_rows.sort(key=lambda x: -x['n'])

    # 批次方向稳定性：单特征在各批次内预测标签的 AUC（<0.5 = 方向翻转）
    dir_rows = []
    for b, g in tmp[tmp['lab'].notna()].groupby('batch_date'):
        if g['lab'].nunique() < 2:
            continue
        dir_rows.append({
            'batch': int(b), 'n': int(len(g)),
            'auc_early_n': round(float(roc_auc_score(g['lab'], g['early_n_log'])), 3),
            'auc_duration': round(float(roc_auc_score(g['lab'], g['duration_log'])), 3),
        })
    lab_means = (tmp[tmp['lab'].notna()].groupby('lab')['early_n']
                 .agg(['mean', 'median', 'count']).round(1).to_dict('index'))

    report = {
        'job': 'm7_gate_eval',
        'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'task': '早期弹幕 + 元数据 → 互动质量（分区内相对水平）',
        'label_definition': {
            'metric_primary': primary,
            'selection_rule': '主标签默认 ir；备选替换需训练期 CV AUC>=0.60 且比 ir 高 >=0.05；测试集不参与',
            'selection_note': '曾按"分区内方差最大"选过投币率（CV 1.19）：它分母小、噪声大，'
                              '元数据基线已 0.736，属"方差大但不可分"；三个标签训练期 CV AUC 均 <=0.56，'
                              '无一满足替换条件 → 保持主标签 ir。',
            'cv_auc_train': {k: v.get('cv_auc_train') for k, v in results.items()},
            'cut': '分区内排名分位 top30%% vs bottom30%%（参考分布=训练期 %d 及以前）' % TRAIN_MAX_BATCH,
            'median_split_secondary': '同分位 >=0.5 为高（未作为判定口径）',
            'min_partition_ref': MIN_REF,
        },
        'dispersion_within_partition_cv': {k: (round(v, 4) if v else None) for k, v in disp.items()},
        'split': {'train_batches': '<= %d' % TRAIN_MAX_BATCH,
                  'test_batch': TEST_BATCH, 'robust_test': list(ROBUST_BATCHES),
                  'rule': '按批次时间切，禁止随机切'},
        'n_rows': int(len(df)),
        'batch_direction': {'rows': dir_rows,
                            'note': 'AUC<0.5 表示该批次内特征方向与标签相反；跨批大幅摆动 = 关系不稳定',
                            'early_n_by_label': lab_means},
        'label_stats': {
            'n_labeled_train': main_res.get('n_train'),
            'n_labeled_test': main_res.get('n_test'),
            'pos_rate_train': main_res.get('pos_rate_train'),
            'by_partition': dist_rows[:15],
        },
        'gate': gate,
        'primary_result': keep_keys,
        'all_labels': {k: {
            'n_train': v.get('n_train'), 'n_test': v.get('n_test'),
            'cv_auc_train': v.get('cv_auc_train'),
            'auc_full': (v.get('models') or {}).get('lr_full', {}).get('auc'),
            'auc_meta': (v.get('models') or {}).get('lr_meta', {}).get('auc'),
            'delta': (None if (v.get('models') or {}).get('lr_full', {}).get('auc') is None
                      else round((v['models']['lr_full']['auc'] or 0)
                                 - (v['models']['lr_meta']['auc'] or 0), 4)),
        } for k, v in results.items() if 'models' in v},
        'features_used': {'early': EARLY_COLS, 'metadata': META_COLS + ['tname_onehot']},
        'forbidden_features': ['play', 'like', 'coin', 'favorite', 'share',
                               'danmaku_reported', 'danmaku_got', 'reply'],
        'missing_sources': ['简介长度', 'UP 主粉丝数', '早期评论数'],
        'artifacts': {'roc_pr_png': os.path.join(OUT, 'm7_gate_roc_pr.png')},
        'caveats': [
            '样本量小（测试批次 100–200 行），AUC 置信区间宽，门控只作方向判定。',
            '分区内分位用训练期做参考分布；测试期市场漂移会直接压低 AUC。',
            '没有人工标注；标签是市场结果的相对分位，不是"内容好坏"。',
        ],
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'ml_gate_interaction_quality_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 可读 md
    def f(x, nd=3):
        return '—' if x is None else ('%.*f' % (nd, x))

    dci = gate.get('delta_ci95') or (None, None)
    lines = ['# 互动质量预测 · ML 门控报告（%s）' % report['generated_at'], '',
             '## 结论', '',
             '**%s**（主标签 %s）' % ('通过' if gate['passed_stable'] else '未过门控', primary), '',
             '| 判据 | 值 | 结论 |', '|---|---|---|',
             '| 测试集 AUC ≥ 0.65 | %s | %s |' % (f(auc_full), '过' if gate['pass_auc'] else '不过'),
             '| 比元数据基线高 ≥ 0.05 | Δ=%s | %s |' % (f(gate['delta']),
                                                     '过' if gate['pass_delta'] else '不过'),
             '| 训练期 5 折 CV AUC ≥ 0.55 | %s | %s |' % (f(gate.get('cv_auc_train')),
                                                       '过' if gate.get('stable_cv') else '不过'),
             '| ΔAUC 的 95%% CI 下界 > 0 | [%s, %s] | %s |'
             % (f(dci[0]), f(dci[1]), '过' if gate.get('stable_delta') else '不过'),
             '',
             '> 前两条是任务书写死的门槛，后两条是 2026-10-22 加的**稳定性守卫**（加严不加宽）：',
             '> 只看前两条会在 n=83 的测试集上"字面通过"，但训练期 CV 与 Δ 的置信区间都不支持。',
             '', '## 标签', '',
             '切法：分区内排名分位（参考=训练期），分位 ≥%.2f 记 1、≤%.2f 记 0（各约 %.0f%%），中间丢弃；'
             % (HIGH_PCT, LOW_PCT, (1 - HIGH_PCT) * 100),
             '训练 %d 行（正例 %.1f%%）/ 测试 %d 行。' % (
                 main_res.get('n_train') or 0, 100 * (main_res.get('pos_rate_train') or 0),
                 main_res.get('n_test') or 0),
             '',
             '| 分区 | 标注行数 | 正例占比 |', '|---|---|---|']
    for r in dist_rows[:8]:
        lines.append('| %s | %d | %.1f%% |' % (r['tname'], r['n'], 100 * r['pos_rate']))
    lines += ['', '### 批次方向稳定性（单特征 AUC）', '',
              '| 批次 | n | early_n | duration |', '|---|---|---|---|']
    for r in dir_rows:
        lines.append('| %d | %d | %.3f | %.3f |' % (r['batch'], r['n'], r['auc_early_n'], r['auc_duration']))
    _dir_vals = [v for r in dir_rows for v in (r['auc_early_n'], r['auc_duration'])]
    lines += ['', '> 单特征 AUC 在 %.2f–%.2f 之间来回翻：同一特征在不同批次方向相反，说明不是稳定关系。'
              % (min(_dir_vals), max(_dir_vals)),
              '> 按标签看 early_n 均值：低组 %s / 高组 %s —— 几乎相同。'
              % (lab_means.get(0, {}).get('mean'), lab_means.get(1, {}).get('mean')),
              '', '| 指标 | 分区内加权 CV | 训练期 CV AUC | 测试 AUC(全) | 测试 AUC(元数据) | Δ |',
             '|---|---|---|---|---|---|']
    for k in ('ir', 'tri', 'coinr'):
        v = results.get(k) or {}
        mm = v.get('models') or {}
        lines.append('| %s | %s | %s | %s | %s | %s |' % (
            k, f(disp.get(k), 4), f(v.get('cv_auc_train')),
            f((mm.get('lr_full') or {}).get('auc')), f((mm.get('lr_meta') or {}).get('auc')),
            f(None if not mm.get('lr_full') or not mm.get('lr_meta')
              else (mm['lr_full']['auc'] - mm['lr_meta']['auc']))))
    lines += ['', '主标签 = **%s**（默认 ir；备选替换需训练期 CV AUC≥0.60 且比 ir 高 ≥0.05；测试集不参与选）' % primary,
              '', '## 模型对比（测试批次 %d，主标签 %s）' % (TEST_BATCH, primary), '',
              '| 模型 | AUC | PR-AUC | F1@0.5 | n |', '|---|---|---|---|---|']
    for name in ('majority', 'lr_meta', 'lr_full', 'gbt_full'):
        r = m.get(name) or {}
        lines.append('| %s | %s | %s | %s | %s |' % (name, f(r.get('auc')), f(r.get('pr_auc')),
                                                     f(r.get('f1')), r.get('n')))
    lines += ['', '## 特征重要性 Top10', '']
    for k, v in main_res['importance']['gbt_top10'].items():
        lines.append('- %s：%.4f' % (k, v))
    lines += ['', '## 误差切片（测试集）', '']
    for t, r in main_res['slices']['tname'].items():
        lines.append('- 分区 %s（n=%d）：AUC 全模型 %s / 元数据 %s' %
                     (t, r['n'], f(r['auc_full']), f(r['auc_meta'])))
    for q, r in main_res['slices']['duration_q'].items():
        lines.append('- 时长 %s（n=%d）：AUC 全模型 %s' % (q, r['n'], f(r['auc_full'])))
    lines += ['', '## 口径与边界', '',
              '- 特征只用**播放进度前 10%** 的弹幕 + 元数据；不含 play/like/coin/fav/share/总弹幕/reply。',
              '- 训练/测试按批次时间切（训练 ≤%d，测试 %d），无随机切分。' % (TRAIN_MAX_BATCH, TEST_BATCH),
              '- 缺字段：简介长度、UP 主粉丝数、早期评论数（仓库无源表）。',
              '- 样本量小（测试 %d 行），AUC 区间宽；本报告只回答"这条线值不值得继续"，不回答"能预测爆款"。'
              % (main_res.get('n_test') or 0)]
    with open(os.path.join(OUT, 'ml_gate_interaction_quality_report.md'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))
    print('\nJSON →', os.path.join(OUT, 'ml_gate_interaction_quality_report.json'))


if __name__ == '__main__':
    main()
