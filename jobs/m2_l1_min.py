#!/usr/bin/env python3
"""
m2_l1_min.py — L1 最小版：dwd → 101 点密度曲线 + 8 维节奏指标
Spark SQL 原生实现（零 Python UDF，10 §2.4 原生化清单），对拍基准 = 旧仓库
analysis/danmaku_metrics.py（NumPy SSOT）+ tests/goldens/*.csv。

算法等价性要点（与 danmaku_metrics.density_curve / extract_8_indicators 逐条对应）：

 1. dur_eff = duration>0 ? duration : 600.0（旧 UDF build_curve 回退，density.py:59）
    progress = clip(time_offset / max(dur_eff,1.0), 0, 1)
 2. 101 桶直方图（np.histogram(bins=101, range=(0,1)) 精确语义：bin k ⟺ e_k ≤ x < e_{k+1}，
    e_k = k*(1.0/101)——numpy linspace 用乘法造边，非除法！x==1.0 强制入桶 100）：
    SQL 实现 = floor(x*101) 起步 + 两次边界修正（与边值比较，±1 内必收敛）
 3. max 归一 → 高斯 σ=1.0 → 再 max 归一。高斯实现 = **窗口函数 + 反射扩展行**（非数组高阶函数）：
    - scipy gaussian_filter1d 默认 mode='reflect' 为 **edge-repeat**（2026-10-12 scipy 实测：
      x[-1]=x[0], x[-2]=x[1]；x[n]=x[n-1]）→ 映射：p<0 → -p-1；p>100 → 201-p
    - 9-tap 核（truncate=4.0）：w[t]=exp(-0.5t²)/Σexp(-0.5t²)，t∈[-4,4]，逐 tap lag/lead 加权和
    - ⚠️ 不用 transform/element_at 数组链：投影塌缩内联 + 解释执行路径（ArrayTransform 每次取元素
      触发 SQLConf 查询）会把复杂度放大到单 task 满转 16 分钟（2026-10-12 实测线程栈定位）
    - 零计数桶稠密化（crossJoin 0..100）：直方图语义 = 全 101 桶（含 0），稀疏行会漏掉反射源
 4. 8 维（front_window='adaptive' 旧口径，B2 切新口径前不得改）：
    - mu/sd = 全 101 点均值/总体标准差(ddof=0)
    - 峰 = scipy find_peaks 精确语义：**平台顶也算峰**——等值 run 左缘严格升、右缘严格降
      → 峰位置 = 平台中点 (L+R)//2；高度过滤 c ≥ μ+2σ；端点（触 0/100 桶）不算峰；
      distance=max(2,101//50)=2 对中点仍 no-op（峰间必有谷 ⇒ 中点距 ≥2，含平台情形）
    - main_peak_pct = (最高峰|无峰取全局argmax，并列取最小下标)/101*100
    - peak_fwhm = 含全局 argmax 的 ≥max/2 连续段跨度：(右侧首个<half 或 100)−(左侧首个<half 或 0)
    - cold_pct = mean(c < μ*0.5)*100
    - front_pct = min(10, max(3, int(60/max(duration_raw,1)*100)))（原始 duration，非 600 回退值）
      front_idx = max(1, int(101*front_pct/100))；front_ratio = mean(c[:idx])/μ（μ≤0→1.0）
    - mid_std = c[20:80].std()（60 点总体 std）；end_rebound = mean(c[90:])/μ（11 点，μ≤0→1.0）
    - cv = sd/μ（μ≤0→0.0）
 浮点纪律：除法/乘法次序逐表达式复刻 Python 左结合次序；求和顺序差异（Spark 窗口 vs NumPy
 成对求和）按容差 1e-9 吸收；计数类（peak_count/peak_fwhm/danmaku_count）为精确整数。

输出：{wh}/dws_video_curve_min、{wh}/dws_video_metrics_min（batch_date 分区，幂等覆盖）
"""
import argparse
import json
import math
import os

from pyspark.sql import SparkSession
from pyspark.sql import Window
import pyspark.sql.functions as F

N_POINTS = 101
SMOOTH_SIGMA = 1.0
KERNEL_RADIUS = 4  # int(truncate=4.0 * sigma=1.0 + 0.5)
ENGINE_VERSION = 'b0-min-v1'


def gaussian_kernel(sigma=SMOOTH_SIGMA, radius=KERNEL_RADIUS):
    """与 scipy.ndimage._gaussian_kernel1d 相同公式：exp(-0.5*t²/σ²) 归一化。
    math.exp 与 np.exp 可能差 1 ulp → 曲线差 ~1e-16，在 1e-9 容差内。"""
    raw = [math.exp(-0.5 * t * t / (sigma * sigma)) for t in range(-radius, radius + 1)]
    s = sum(raw)
    return [w / s for w in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root
    out_dir = args.out_dir or ('/opt/work/data-out' if not args.local
                               else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data-out'))
    os.makedirs(out_dir, exist_ok=True)

    b = (SparkSession.builder.appName('m2_l1_min')
         .config('spark.sql.shuffle.partitions', '8')
         .config('spark.sql.sources.partitionOverwriteMode', 'dynamic'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # ── 1) dwd → 桶计数（直方图 SQL 语义见文件头注 2）──
    src = (spark.read.parquet(f'{wh}/dwd_danmaku_clean_min')
           .filter(F.col('quality_flag') == 'ok'))

    meta = (src.groupBy('bvid', 'batch_date')
            .agg(F.max('duration').alias('duration_raw'),
                 F.max('title').alias('title'),
                 F.max('tname').alias('tname'),
                 F.max('play').alias('play'),
                 F.count('*').alias('danmaku_count')))

    pts = (src.select('bvid', 'batch_date', 'time_offset')
           .join(meta.select('bvid', 'batch_date', 'duration_raw'), ['bvid', 'batch_date'])
           .withColumn('dur_eff',
                       F.when(F.col('duration_raw') > 0,
                              F.col('duration_raw').cast('double')).otherwise(F.lit(600.0)))
           .withColumn('progress',
                       F.least(F.greatest(
                           F.col('time_offset') / F.greatest(F.col('dur_eff'), F.lit(1.0)),
                           F.lit(0.0)), F.lit(1.0)))
           # 直方图分桶：floor 起步 + 边界修正（e_k = k*(1.0/101)）
           .withColumn('b0', F.floor(F.col('progress') * F.lit(101.0)).cast('int'))
           .withColumn('b1', F.when((F.col('b0') >= 1) &
                                    (F.col('progress') < F.col('b0') * F.lit(1.0 / 101.0)),
                                    F.col('b0') - 1).otherwise(F.col('b0')))
           .withColumn('b2', F.when(F.col('progress') >=
                                    (F.col('b1') + 1) * F.lit(1.0 / 101.0),
                                    F.col('b1') + 1).otherwise(F.col('b1')))
           .withColumn('bucket', F.least(F.greatest(F.col('b2'), F.lit(0)), F.lit(100))))

    cnt = (pts.groupBy('bvid', 'batch_date', 'bucket').agg(F.count('*').alias('cnt')))

    # ── 2) 稠密化（全 101 桶，零计数补 0）+ 第一次 max 归一 ──
    wv = Window.partitionBy('bvid', 'batch_date')
    buckets = spark.range(0, N_POINTS).select(F.col('id').alias('bucket'))
    dense = (meta.select('bvid', 'batch_date').distinct().crossJoin(buckets)
             .join(cnt, ['bvid', 'batch_date', 'bucket'], 'left')
             .withColumn('cnt', F.coalesce(F.col('cnt'), F.lit(0)).cast('int')))
    dense = (dense.withColumn('m1', F.max('cnt').over(wv))
             .withColumn('c1', F.when(F.col('m1') > 0,
                                      F.col('cnt') / F.col('m1')).otherwise(F.col('cnt').cast('double'))))

    # ── 3) 反射扩展行（edge-repeat，scipy 实测：p<0 → -p-1；p>100 → 201-p）──
    ext_pairs = [(p, -p - 1) for p in range(-KERNEL_RADIUS, 0)] + \
                [(p, 201 - p) for p in range(N_POINTS, N_POINTS + KERNEL_RADIUS)]
    ext_map = spark.createDataFrame(ext_pairs, ['p', 'src_bucket']).cache()
    ext = (dense.select('bvid', 'batch_date', F.col('bucket').alias('src_bucket'), 'c1')
           .join(ext_map, 'src_bucket')
           .select('bvid', 'batch_date', F.col('p'), 'c1'))
    real = dense.select('bvid', 'batch_date', F.col('bucket').alias('p'), 'c1')
    rows = real.unionByName(ext)

    # ── 4) 9-tap 高斯（lag/lead 窗口加权和，tap 顺序 t=-4..+4 与 scipy 累加次序一致）──
    wp = Window.partitionBy('bvid', 'batch_date').orderBy('p')
    K = gaussian_kernel()
    smooth_expr = None
    for j, wt in enumerate(K):
        t = j - KERNEL_RADIUS
        if t < 0:
            col = F.lag('c1', -t).over(wp)
        elif t == 0:
            col = F.col('c1')
        else:
            col = F.lead('c1', t).over(wp)
        term = F.lit(float(wt)) * col
        smooth_expr = term if smooth_expr is None else smooth_expr + term
    rows = rows.withColumn('smooth', smooth_expr)

    # ── 5) 真实桶 → 第二次 max 归一 → 收集 101 数组（聚合边界，安全）──
    real_smooth = (rows.filter((F.col('p') >= 0) & (F.col('p') <= 100))
                   .withColumn('m2', F.max('smooth').over(wv))
                   .withColumn('cv_val', F.when(F.col('m2') > 0,
                                                F.col('smooth') / F.col('m2')).otherwise(F.col('smooth'))))
    curves = (real_smooth.groupBy('bvid', 'batch_date')
              .agg(F.map_from_entries(F.collect_list(
                  F.struct(F.col('p').cast('int'), F.col('cv_val')))).alias('cmap'))
              .withColumn('curve',
                          F.expr('transform(sequence(0,100), b -> coalesce(element_at(cmap, b), 0.0D))')))

    cv = (curves.join(meta, ['bvid', 'batch_date'])
          # front 窗口：原始 duration（非 600 回退），口径=旧 extract_8_indicators('adaptive')
          .withColumn('front_pct', F.least(F.lit(10), F.greatest(F.lit(3),
                  F.floor(F.lit(60.0) / F.greatest(F.coalesce(F.col('duration_raw'), F.lit(1)), F.lit(1))
                          * F.lit(100.0)).cast('int'))))
          .withColumn('front_idx', F.greatest(F.lit(1),
                  F.floor(F.col('front_pct') * F.lit(101) / F.lit(100.0)).cast('int'))))

    n_videos = cv.count()
    cv.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_video_curve_min')

    # ── 6) 8 维指标：posexplode → 窗口函数族（行级窗口，无数组链内联风险）──
    pt = cv.select('bvid', 'batch_date', 'duration_raw', 'danmaku_count', 'title', 'tname', 'play',
                   'front_idx', F.posexplode('curve').alias('bucket', 'c'))

    w = Window.partitionBy('bvid', 'batch_date')
    wo = w.orderBy('bucket')

    pt = (pt
          # μ/σ/cmax/argmax（全局 argmax 并列取最小下标 = 最大值平台左缘，np.argmax 语义）
          .withColumn('mu', F.avg('c').over(w))
          .withColumn('cmax', F.max('c').over(w))
          .withColumn('argmax_idx', F.min(F.when(F.col('c') == F.col('cmax'),
                                                 F.col('bucket'))).over(w))
          .withColumn('half', F.col('cmax') * F.lit(0.5))
          .withColumn('sd', F.sqrt(F.sum(F.pow(F.col('c') - F.col('mu'), F.lit(2.0))).over(w)
                                   / F.lit(101.0)))
          # 相邻值 + 等值 run 编号（find_peaks 平台语义的基础）
          .withColumn('prev_v', F.lag('c').over(wo))
          .withColumn('next_v', F.lead('c').over(wo))
          .withColumn('run_id', F.sum(F.when(F.col('prev_v').isNull() |
                                             (F.col('c') != F.col('prev_v')),
                                             1).otherwise(0)).over(wo))
          # FWHM：含 argmax 的 ≥half 连续段（左右首个 <half，无则 0/100）
          .withColumn('l_false_w', F.max(F.when((F.col('bucket') < F.col('argmax_idx')) &
                                                 (F.col('c') < F.col('half')),
                                                 F.col('bucket'))).over(w))
          .withColumn('r_false_w', F.min(F.when((F.col('bucket') > F.col('argmax_idx')) &
                                                 (F.col('c') < F.col('half')),
                                                 F.col('bucket'))).over(w))
          .withColumn('fwhm_w', F.coalesce(F.col('r_false_w'), F.lit(100))
                      - F.coalesce(F.col('l_false_w'), F.lit(0)))
          # cold / front / mid / end
          .withColumn('cold_k_w', F.sum(F.when(F.col('c') < F.col('mu') * F.lit(0.5), 1)
                                        .otherwise(0)).over(w))
          .withColumn('front_sum_w', F.sum(F.when(F.col('bucket') < F.col('front_idx'),
                                                  F.col('c'))).over(w))
          .withColumn('mid_mu_w', F.avg(F.when((F.col('bucket') >= 20) & (F.col('bucket') <= 79),
                                               F.col('c'))).over(w))
          .withColumn('mid_ssq_w', F.sum(F.pow(F.when((F.col('bucket') >= 20) &
                                                      (F.col('bucket') <= 79), F.col('c'))
                                               - F.col('mid_mu_w'), F.lit(2.0))).over(w))
          .withColumn('mid_std_w', F.sqrt(F.col('mid_ssq_w') / F.lit(60.0)))
          .withColumn('end_sum_w', F.sum(F.when(F.col('bucket') >= 90, F.col('c'))).over(w)))

    # ── 6b) 峰检测：find_peaks 精确语义（scipy _local_maxima_1d）──
    # 平台顶也算峰：run 内值恒定，左缘严格升 (c>prev)、右缘严格降 (c>next) → 峰；
    # 报告位置 = 平台中点 (L+R)//2（2026-10-12 对拍实测：BV15Xe768Enh@20260917
    # 的 2 长平台峰被"严格局部极大"漏检后定位）。边界 run（触 0/100 桶）因 prev/next
    # 为 null 自动排除 = scipy "端点不算峰"。distance=2 对中点仍 no-op（峰间必有谷，
    # 中点距 ≥2，含平台情形）。
    runs = (pt.groupBy('bvid', 'batch_date', 'run_id')
            .agg(F.min('bucket').alias('L'), F.max('bucket').alias('R'),
                 F.max('c').alias('val'),
                 F.max(F.when(F.col('prev_v').isNotNull() & (F.col('c') > F.col('prev_v')),
                              1)).alias('rise'),
                 F.max(F.when(F.col('next_v').isNotNull() & (F.col('c') > F.col('next_v')),
                              1)).alias('fall'),
                 F.max('mu').alias('mu'), F.max('sd').alias('sd'))
            .withColumn('midpoint', F.floor((F.col('L') + F.col('R')) / F.lit(2.0)).cast('int'))
            .withColumn('is_peak_run',
                        (F.col('rise') == 1) & (F.col('fall') == 1) &
                        (F.col('val') >= F.col('mu') + F.lit(2.0) * F.col('sd'))))

    w_pk = Window.partitionBy('bvid', 'batch_date')
    peaks_v = (runs.filter('is_peak_run')
               .withColumn('pk_cmax', F.max('val').over(w_pk))
               .withColumn('main_mid', F.min(F.when(F.col('val') == F.col('pk_cmax'),
                                                    F.col('midpoint'))).over(w_pk))
               .groupBy('bvid', 'batch_date')
               .agg(F.count('*').alias('peak_count'),
                    F.max('main_mid').alias('main_peak_mid')))

    m = (pt.groupBy('bvid', 'batch_date')
         .agg(F.max('title').alias('title'), F.max('tname').alias('tname'),
              F.max('play').alias('play'), F.max('duration_raw').alias('duration'),
              F.max('danmaku_count').alias('danmaku_count'),
              F.max('argmax_idx').alias('argmax_idx'),
              F.max('fwhm_w').alias('peak_fwhm'),
              F.max('cold_k_w').alias('cold_k'),
              F.max('front_sum_w').alias('front_sum'), F.max('front_idx').alias('front_idx'),
              F.max('end_sum_w').alias('end_sum'),
              F.max('mu').alias('mu'), F.max('sd').alias('sd'),
              F.max('mid_std_w').alias('mid_std'))
         .join(peaks_v, ['bvid', 'batch_date'], 'left')
         .withColumn('peak_count', F.coalesce(F.col('peak_count'), F.lit(0)).cast('int'))
         # main_peak：有峰=最高平台峰中点（并列取最小中点）；无峰=全局 argmax（np.argmax 语义）
         .withColumn('main_idx', F.when(F.col('peak_count') > 0,
                                        F.col('main_peak_mid')).otherwise(F.col('argmax_idx')))
         .withColumn('main_peak_pct', (F.col('main_idx').cast('double') / F.lit(101.0)) * F.lit(100.0))
         .withColumn('cold_pct', (F.col('cold_k').cast('double') / F.lit(101.0)) * F.lit(100.0))
         .withColumn('front_ratio', F.when(F.col('mu') > 0,
                                           (F.col('front_sum') / F.col('front_idx').cast('double'))
                                           / F.col('mu')).otherwise(F.lit(1.0)))
         .withColumn('end_rebound', F.when(F.col('mu') > 0,
                                           (F.col('end_sum') / F.lit(11.0)) / F.col('mu'))
                                       .otherwise(F.lit(1.0)))
         .withColumn('cv', F.when(F.col('mu') > 0, F.col('sd') / F.col('mu')).otherwise(F.lit(0.0)))
         .withColumn('engine_version', F.lit(ENGINE_VERSION))
         .drop('main_idx', 'cold_k', 'front_sum', 'front_idx', 'end_sum', 'mu', 'sd',
               'main_peak_mid', 'argmax_idx'))

    n_metrics = m.count()
    m.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_video_metrics_min')

    # ── 7) 运行报告 ──
    report = {'job': 'm2_l1_min', 'engine_version': ENGINE_VERSION,
              'n_videos': n_videos, 'n_metrics': n_metrics,
              'anchors_expect': {'n_videos': 509, 'sum_danmaku_count': 672439}}
    sum_dm = m.agg(F.sum('danmaku_count')).collect()[0][0]
    report['sum_danmaku_count'] = sum_dm
    report['anchors_match'] = {'n_videos': n_videos == 509, 'sum_danmaku_count': sum_dm == 672439}
    with open(os.path.join(out_dir, 'l1_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m2_l1_min 完成：{n_videos} 视频批次曲线+8维 '
          f'(锚点匹配: {report["anchors_match"]})')
    spark.stop()


if __name__ == '__main__':
    main()
