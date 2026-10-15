# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

D = pd.read_csv('/home/molly/projects/danmaku-v2/data-out/exp_data.csv')
CURVE = ['c%d' % i for i in range(101)]
SHAPE = ['opening_5pct','ending_5pct','mid_ratio','rhythm_entropy','burst_cv','early_lag_ratio']
VOL   = ['dm_per_min','wdm_per_min','n_danmaku','avg_text_len']
print('总行 %d | tname %d | up_mid %d | n_days %d' % (len(D), D.tname.notna().sum(), D.up_mid.notna().sum(), D.n_days_obs.notna().sum()))

def cv(make_clf, X, y, groups, metric='f1', n=5):
    out = []
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        m = make_clf(); m.fit(X[tr], y[tr])
        if metric == 'f1': out.append(f1_score(y[te], m.predict(X[te]), average='macro', zero_division=0))
        else: out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(out)), float(np.std(out))

RF = lambda: Pipeline([('s', StandardScaler()), ('m', RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1))])
RFP = lambda k: Pipeline([('s', StandardScaler()), ('p', PCA(n_components=k, random_state=0)), ('m', RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1))])
LR = lambda: Pipeline([('s', StandardScaler()), ('m', LogisticRegression(max_iter=3000, C=0.5))])

lc = D.groupby('tname')['bvid'].nunique(); keep = sorted(lc[lc >= 20].index)
A = D[D.tname.isin(keep)].dropna(subset=['up_mid']).reset_index(drop=True)
ya = A.tname.map({t: i for i, t in enumerate(keep)}).values
print()
print('=== A. 节奏 -> 分区（K=%d n=%d chance=%.4f）===' % (len(keep), len(A), 1.0/len(keep)))
for tag, cols, mk in [('7维签名', SHAPE, RF), ('101点曲线', CURVE, RF), ('101点+PCA10', CURVE, lambda: RFP(10)), ('101点+PCA20', CURVE, lambda: RFP(20))]:
    X = A[cols].values
    for gname in ['bvid', 'up_mid']:
        m, s = cv(mk, X, ya, A[gname].values)
        print('   %-14s by %-7s macroF1 = %.3f +- %.3f' % (tag, gname, m, s))

B = D.dropna(subset=['n_days_obs','up_mid']).copy()
B['y'] = (B.n_days_obs >= B.n_days_obs.median()).astype(int)
print()
print('=== B/C. 节奏 -> 持续性（n=%d 正类 %.2f）===' % (len(B), B.y.mean()))
for tag, cols in [('仅量(4维)', VOL), ('仅形状(6维)', SHAPE), ('量+形状', VOL+SHAPE)]:
    X = B[cols].values; y = B.y.values
    for gname in ['bvid', 'up_mid']:
        a, _ = cv(LR, X, y, B[gname].values, metric='auc')
        f, _ = cv(RF, X, y, B[gname].values, metric='auc')
        print('   %-14s by %-7s AUROC  LR=%.3f  RF=%.3f' % (tag, gname, a, f))