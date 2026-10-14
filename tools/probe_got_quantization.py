from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = (SparkSession.builder.appName('probe_got_quant')
         .config('spark.sql.shuffle.partitions', '8').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

snap = spark.read.parquet('/warehouse/dwd_video_metric_snapshot')
d = snap.filter(F.col('batch_date').isin('20261012', '20261013'))
print('daily snapshot rows:', d.count())

print('--- danmaku_got 取值分布（top12） ---')
for r in (d.groupBy('danmaku_got').count().orderBy(F.desc('count')).limit(12).collect()):
    g = r['danmaku_got']
    print(f"got={g:>6} n={r['count']:>3}  %600={g % 600 if g else 0}")

print('--- 饱和判定（got>=1200 且 reported > 1.5*got）---')
sat = d.filter((F.col('danmaku_got') >= 1200) & (F.col('danmaku_reported') > F.col('danmaku_got') * 1.5))
print('saturated rows:', sat.count(), '/', d.count())

print('--- BV1QmpE6aEA7 跨批（机制实锤样例）---')
for r in (snap.filter(F.col('bvid') == 'BV1QmpE6aEA7')
          .select('batch_date', 'danmaku_reported', 'danmaku_got', 'play').collect()):
    print(dict(r.asDict()))

print('--- 量子化覆盖：全库 got 值落在 600 整数倍的比例（饱和段 got>=1200）---')
f = d.filter((F.col('danmaku_got') >= 1200))
tot = f.count()
onq = f.filter((F.col('danmaku_got') % 600) == 0).count()
print(f'{onq}/{tot} = {onq / max(tot, 1):.3f}')
spark.stop()
