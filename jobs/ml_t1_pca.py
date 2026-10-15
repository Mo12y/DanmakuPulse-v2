# -*- coding: utf-8 -*-
'''对照实验：用 101 点密度曲线本身（PCA）替代 7 维签名 —— 排除「签名丢信息」的可能'''
import numpy as np
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
spark = SparkSession.builder.appName('t1pca').config('spark.sql.shuffle.partitions','8').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
d1 = spark.read.parquet('/warehouse/dws_density_1pct').select('bvid','batch_date','bucket1pct','wcnt')
# 每视频 101 维曲线（缺失桶补 0）
rows = d1.collect()
from collections import defaultdict
g = defaultdict(dict)
for r in rows: g[(r['bvid'], r['batch_date'])][int(r['bucket1pct'])] = float(r['wcnt'] or 0.0)
dim = spark.read.parquet('/warehouse/dim_video').select('bvid','tname','up_mid')
tmap = {r['bvid']: (r['tname'], r['up_mid']) for r in dim.collect()}
data = []
for (bv, bd), d in g.items():
    if bv not in tmap: continue
    t, u = tmap[bv]
    if not t: continue
    vec = [d.get(i, 0.0) for i in range(101)]
    mx = max(vec) or 1.0
    data.append((bv, bd, u, t, [v/mx for v in vec]))   # 归一化到峰值=1（形状）
import pyspark.sql.types as T
schema = T.StructType([T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.StringType()),
                       T.StructField('up_mid', T.StringType()), T.StructField('tname', T.StringType()),
                       T.StructField('curve', T.ArrayType(T.DoubleType()))])
df = spark.createDataFrame(data, schema)
cnt = df.groupBy('tname').agg(F.countDistinct('bvid').alias('nv'))
keep = sorted([r['tname'] for r in cnt.filter(F.col('nv') >= 20).collect()])
df = df.filter(F.col('tname').isin(keep))
K = len(keep)
lab = F.create_map(*[x for t in keep for x in (F.lit(t), F.lit(float(keep.index(t))))])
df = df.withColumn('label', lab[F.col('tname')])
print('K=%d chance=%.4f | 行 %d' % (K, 1.0/K, df.count()))

def metrics(pred):
    from pyspark.mllib.evaluation import MulticlassMetrics
    rdd = pred.select('prediction','label').rdd.map(lambda r: (float(r[0]), float(r[1])))
    cm = MulticlassMetrics(rdd).confusionMatrix().toArray()
    f1s = []
    for i in range(len(cm)):
        tp = cm[i,i]; fp = cm[:,i].sum()-tp; fn = cm[i,:].sum()-tp
        p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p*r/(p+r) if p+r else 0.0)
    return float(np.mean(f1s)), float(np.trace(cm))/max(cm.sum(),1)

for ncomp in [10, 20]:
    for tag, fc, gcol in [('bvid','fb','bvid'), ('upmid','fu','up_mid')]:
        d = df.withColumn('fb', F.pmod(F.hash('bvid'), F.lit(5))).withColumn('fu', F.pmod(F.hash('up_mid'), F.lit(5)))
        f1s, accs = [], []
        for f in range(5):
            tr = d.filter(F.col(fc) != f); te = d.filter(F.col(fc) == f)
            try:
                va = VectorAssembler(inputCols=['curve'], outputCol='fv')
                sc = StandardScaler(inputCol='fv', outputCol='sc', withMean=True, withStd=True)
                pca = PCA(k=ncomp, inputCol='sc', outputCol='features')
                rf = RandomForestClassifier(numTrees=100, maxDepth=6, seed=42, featuresCol='features')
                m = Pipeline(stages=[va, sc, pca, rf]).fit(tr)
                a, b = metrics(m.transform(te)); f1s.append(a); accs.append(b)
            except Exception as e:
                print('   %s fold%d ERR %s' % (tag, f, str(e)[:70]))
        if f1s:
            print('101点曲线+PCA(%d) 分组=%-5s  macroF1=%.3f+-%.3f  acc=%.3f'
                  % (ncomp, tag, np.mean(f1s), np.std(f1s), np.mean(accs)))
spark.stop()