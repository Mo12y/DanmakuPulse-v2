#!/usr/bin/env python3
"""
m2_l1_density.py — L1 密度引擎（08 §3.1）

输入：dwd_danmaku_clean（刷量排除主口径：is_spam == false，10 §3）
输出：
  {wh}/dws_density_5s   每 (bvid, batch_date, bucket5s) 一行：cnt / wcnt
  {wh}/dws_density_1pct 每 (bvid, batch_date, bucket1pct) 一行：cnt / wcnt
口径（08 §3.1 / §0.1 幂等；L2 输入契约 2026-10-13 定死）：
  - 5s 桶：bucket5s = floor(time_offset / bucket_sec)；cnt=COUNT(*)，wcnt=SUM(attr_weight)
  - 1% 桶：bucket1pct = floor(offset_pct*100)，clamp 到 [0,99]（pct≥1.0 时 floor 得 100，并入 99；
    桶域固定 100 桶——注意 ≠ L1-min 金标曲线的 101 点口径，两者是不同产物）
  - **稀疏落盘（非空桶才写行）**：缺失行 ⇔ cnt=0 / wcnt=0。消费方重建等长稠密序列的契约：
      5s：域 = [0, 各视频 max(bucket5s)]（数据决定；域内缺行补 0）——L2 四检测器输入（08 §4.1，用 wcnt）
      1pct：域 = [0,99] 固定（缺行补 0）
    证据：tools/probe_l2_density_contract.py（2026-10-13：双向 anti-join 0 丢失/0 多余；
      1pct 45,074 = 50,900 − 内部空桶 5,044 − 尾部空桶 782；两表同策略，差异仅因分桶粒度）
  - 检测下游只读 5s 原始口径（禁平滑）；展示曲线由 API 层现场 rolling mean 3（不在本作业做）
  - 全量覆盖写（每批次 D+1 顺序由 m0 → m2_l1_density），重跑幂等
用法：
  集群：spark-submit ... /opt/work/jobs/m2_l1_density.py
  本地：~/danmaku-venv/bin/python jobs/m2_l1_density.py --local --wh-root file:data-out/dev-wh
"""
import argparse

import pyspark.sql.functions as F
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b1-v1'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--bucket-sec', type=int, default=5)
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    b = (SparkSession.builder.appName('m2_l1_density')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    d = (spark.read.parquet(f'{wh}/dwd_danmaku_clean')
         .filter(~F.col('is_spam'))
         .select('bvid', 'batch_date', 'time_offset', 'offset_pct', 'attr_weight'))

    # ── 5s 桶（原始口径，检测下游唯一输入）──
    d5 = (d.filter(F.col('time_offset').isNotNull())
          .withColumn('bucket5s',
                      F.floor(F.col('time_offset') / F.lit(args.bucket_sec)).cast('long'))
          .groupBy('bvid', 'batch_date', 'bucket5s')
          .agg(F.count('*').alias('cnt'),
               F.sum('attr_weight').alias('wcnt'))
          .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    d5.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_density_5s')

    # ── 1% 桶（offset_pct ×100，clamp 99）──
    d1 = (d.filter(F.col('offset_pct').isNotNull())
          .withColumn('bucket1pct',
                      F.least(F.floor(F.col('offset_pct') * 100), F.lit(99)).cast('int'))
          .groupBy('bvid', 'batch_date', 'bucket1pct')
          .agg(F.count('*').alias('cnt'),
               F.sum('attr_weight').alias('wcnt'))
          .withColumn('engine_version', F.lit(ENGINE_VERSION)))
    d1.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_density_1pct')

    n5 = d5.count()
    n1 = d1.count()
    nvb = d.select('bvid', 'batch_date').distinct().count()
    print(f'✅ m2_l1_density：5s={n5:,} 行 ｜ 1pct={n1:,} 行 ｜ video-batches={nvb}')
    spark.stop()


if __name__ == '__main__':
    main()
