# -*- coding: utf-8 -*-
'''T1：节奏 -> 内容分区（判别力）+ UP主混杂检验 + 特征集消融
修正：LogisticRegression/RF 的 featuresCol 必须显式设为 scaler 输出列
新增：GroupKFold by up_mid —— 检验模型学的是「内容类型」还是「UP主的观众」
'''
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

spark = SparkSession.builder.appName('t1').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
sig = spark.read.option('header', True).option('inferSchema', True).csv('/opt/work/data-out/rhythm_signature')
dim = spark.read.parquet('/warehouse/dim_video').select('bvid','tname','up_mid')
df = sig.drop('tname').join(dim, 'bvid', 'left')
print('签名表行数', df.count())
cnt = df.groupBy('tname').agg(F.countDistinct('bvid').alias('nv')).filter(F.col('tname').isNotNull())
keep = sorted([r['tname'] for r in cnt.filter(F.col('nv') >= 20).collect()])
df = df.filter(F.col('tname').isin(keep))
K = len(keep)
lab = F.create_map(*[x for t in keep for x in (F.lit(t), F.lit(float(keep.index(t))))])
df = df.withColumn('label', lab[F.col('tname')])
nv = df.select('up_mid').distinct().count()
print('K=%d 类 | 行 %d | unique bvid %d | unique up_mid %d' % (K, df.count(), df.select('bvid').distinct().count(), nv))

SHAPE = ['opening_5pct','ending_5pct','mid_ratio','rhythm_entropy','burst_cv','early_lag_ratio','main_peak_pct']
INTEN = ['dm_per_min','wdm_per_min','avg_text_len','n_danmaku']
SENT  = ['sent_early','sent_mid','sent_late']
SETS = [('仅形状', SHAPE), ('仅强度', INTEN), ('仅情绪', SENT), ('形状+情绪', SHAPE+SENT), ('全部', SHAPE+INTEN+SENT)]

def macro_f1(pred):
    from pyspark.mllib.evaluation import MulticlassMetrics
    rdd = pred.select('prediction','label').rdd.map(lambda r: (float(r[0]), float(r[1])))
    cm = MulticlassMetrics(rdd).confusionMatrix().toArray()
    f1s = []
    for i in range(len(cm)):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(f1s)), float(np.trace(cm))/max(cm.sum(),1)

def run(d, use, group_col, mk, tag):
    va = VectorAssembler(inputCols=use, outputCol='fv', handleInvalid='skip')
    sc = StandardScaler(inputCol='fv', outputCol='features', withMean=True, withStd=True)
    f1s, accs = [], []
    for f in range(5):
        tr = d.filter(F.col('fold') != f); te = d.filter(F.col('fold') == f)
        try:
            m = Pipeline(stages=[va, sc, mk()]).fit(tr)
            f1, acc = macro_f1(m.transform(te)); f1s.append(f1); accs.append(acc)
        except Exception as e:
            print('     fold%d 失败 %s' % (f, str(e)[:70])); return None
    return np.mean(f1s), np.std(f1s), np.mean(accs)

LR = lambda: LogisticRegression(maxIter=200, regParam=0.3, featuresCol='features', family='multinomial')
RF = lambda: RandomForestClassifier(numTrees=200, maxDepth=6, seed=42, featuresCol='features')

print()
print('%-12s %-4s %7s %-22s %s' % ('特征集','模型','n','macroF1 (by bvid)','macroF1 (by up_mid, 防UP主混杂)'))
print('-'*100)
for name, cols in SETS:
    use = [c for c in cols if c in df.columns]
    d = df.dropna(subset=use+['label','up_mid']).withColumn('fold_b', F.pmod(F.hash('bvid'), F.lit(5))).withColumn('fold_u', F.pmod(F.hash('up_mid'), F.lit(5)))
    d.cache(); n = d.count()
    for mn, mk in [('LR', LR), ('RF', RF)]:
        r1 = run(d.withColumn('fold', F.col('fold_b')), use, 'bvid', mk, mn)
        r2 = run(d.withColumn('fold', F.col('fold_u')), use, 'up_mid', mk, mn)
        s1 = '%.3f±%.3f' % (r1[0], r1[1]) if r1 else 'FAIL'
        s2 = '%.3f±%.3f' % (r2[0], r2[1]) if r2 else 'FAIL'
        print('%-12s %-4s %7d %-22s %s' % (name, mn, n, s1, s2))
    d.unpersist()
print()
print('chance(1/K) = %.4f' % (1.0/K))
spark.stop()