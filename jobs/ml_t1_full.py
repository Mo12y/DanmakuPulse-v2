# -*- coding: utf-8 -*-
'''T1 完整版：macro-F1 + accuracy + 逐类 + 全 16 维对照 + UP主混杂'''
import numpy as np, json
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
spark = SparkSession.builder.appName('t1full').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
sig = spark.read.option('header', True).option('inferSchema', True).csv('/opt/work/data-out/rhythm_signature')
dim = spark.read.parquet('/warehouse/dim_video').select('bvid','tname','up_mid')
df = sig.drop('tname').join(dim, 'bvid', 'left')
cnt = df.groupBy('tname').agg(F.countDistinct('bvid').alias('nv')).filter(F.col('tname').isNotNull())
keep = sorted([r['tname'] for r in cnt.filter(F.col('nv') >= 20).collect()])
df = df.filter(F.col('tname').isin(keep))
K = len(keep)
lab = F.create_map(*[x for t in keep for x in (F.lit(t), F.lit(float(keep.index(t))))])
df = df.withColumn('label', lab[F.col('tname')])
print('K=%d  |  chance=%.4f' % (K, 1.0/K))
cc = df.groupBy('tname').count().orderBy(F.desc('count')).collect()
print('类别分布:', ', '.join('%s:%d' % (r['tname'], r['count']) for r in cc))
maj = max(r['count'] for r in cc) / sum(r['count'] for r in cc)
print('多数类基线 accuracy = %.3f' % maj)

SHAPE = ['opening_5pct','ending_5pct','mid_ratio','rhythm_entropy','burst_cv','early_lag_ratio','main_peak_pct']
FULL16 = SHAPE + ['wdm_per_min','avg_text_len','peak_cnt']
SETS = [('形状7维', SHAPE), ('形状+峰数', SHAPE+['peak_cnt']), ('强度', ['dm_per_min','wdm_per_min','avg_text_len','n_danmaku']), ('形状+强度+情绪', SHAPE+['dm_per_min','wdm_per_min','avg_text_len','n_danmaku','sent_early','sent_mid','sent_late'])]
def metrics(pred):
    from pyspark.mllib.evaluation import MulticlassMetrics
    rdd = pred.select('prediction','label').rdd.map(lambda r: (float(r[0]), float(r[1])))
    cm = MulticlassMetrics(rdd).confusionMatrix().toArray()
    f1s = []
    for i in range(len(cm)):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(f1s)), float(np.trace(cm))/max(cm.sum(),1), [round(x,3) for x in f1s]
LR = lambda: LogisticRegression(maxIter=200, regParam=0.3, featuresCol='features', family='multinomial')
RF = lambda: RandomForestClassifier(numTrees=200, maxDepth=6, seed=42, featuresCol='features')
print()
print('%-16s %-3s %6s %-16s %-16s %s' % ('特征集','模型','n','macroF1(bvid)','macroF1(upmid)','acc(upmid)'))
for name, cols in SETS:
    use = [c for c in cols if c in df.columns]
    d = df.dropna(subset=use+['label','up_mid'])
    d = d.withColumn('fb', F.pmod(F.hash('bvid'), F.lit(5))).withColumn('fu', F.pmod(F.hash('up_mid'), F.lit(5)))
    n = d.count()
    if n < 100: print('%-16s 样本不足 %d' % (name, n)); continue
    for mn, mk in [('LR', LR), ('RF', RF)]:
        res = {}
        for tag, fc in [('bvid','fb'), ('upmid','fu')]:
            va = VectorAssembler(inputCols=use, outputCol='fv', handleInvalid='skip')
            sc = StandardScaler(inputCol='fv', outputCol='features', withMean=True, withStd=True)
            f1s, accs = [], []
            for f in range(5):
                tr = d.filter(F.col(fc) != f); te = d.filter(F.col(fc) == f)
                try:
                    m = Pipeline(stages=[va, sc, mk()]).fit(tr)
                    a, b, _ = metrics(m.transform(te)); f1s.append(a); accs.append(b)
                except Exception as e: print('   %s %s fold%d %s' % (name, mn, f, str(e)[:50]))
            res[tag] = (np.mean(f1s) if f1s else float('nan'), np.std(f1s) if f1s else 0, np.mean(accs) if accs else float('nan'))
        print('%-16s %-3s %6d %.3f±%.3f      %.3f±%.3f      %.3f'
              % (name, mn, n, res['bvid'][0], res['bvid'][1], res['upmid'][0], res['upmid'][1], res['upmid'][2]))
spark.stop()