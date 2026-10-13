#!/usr/bin/env python3
"""
m2_l2_sync.py — L2 同步性计算（08 §4.3）+ 羊群权重（08 §4.4）

输入：dws_rhythm_events（事件）+ dwd_danmaku_clean（非 spam；窗口弹幕与置换抽样）
      + dws_video_rhythm_features（全集口径）
输出：① 回写 dws_rhythm_events（追加 sync_score / sync_p / h_pos / h_time / n_window 列）
      ② {wh}/dws_sync_features —— 每 (bvid, batch_date) 一行（视频级聚合 + 羊群权重）
      ③ 报告 data-out/m2_l2_sync_report.json

口径（08 §4.3，本实现冻结详见 2026-10-13 实现决策）：
  - 峰窗 W = 事件 [start−peak_win, end+peak_win] 桶内弹幕（非 spam；与 density 同全集）
  - H_pos = 窗口弹幕在视频 5s 桶网格 [0..B-1] 上分布的归一化熵（除以 ln B；越小越聚集）
  - H_time = 窗口弹幕 send_date 分布的归一化熵（除以 ln D，D=视频自身日跨度；越大越跨天）
  - sync_score = (1 − H_pos)·H_time ∈ [0,1]
  - 置换检验：从该视频全体弹幕随机抽同数量 N（无放回）重算 → sync_rand；perm_n 次；
    sync_p = P(sync_rand ≥ sync_obs)；种子 = crc32(bvid|batch_date) 固定 → 重跑幂等（08 §0.1）
  - 边缘：N<2 的事件 → sync_score=0、p=1.0（仍输出，保证「p 值覆盖所有事件」DoD）
  - 羊群权重（08 §4.4）：同 (bvid,batch_date,text) 第 j 次出现（按 send_time 排序）权重 j/k_t；
    herd_mean = 视频内权重均值（全集：剔除 is_ad / is_meaningless / is_subtitle_pool；spam 已排除）
  - 视频级（08 §4.3「视频级取事件中位数」）：n_events / n_sig05 / sync_med / sync_p_med
实现：逐视频含 1000 次置换 → 数据并行（RDD cogroup + NumPy；种子固定；不引 pandas）。
"""

import argparse
import json
import os
import zlib

import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession

ENGINE_VERSION = 'b2-v1'
PERM_CHUNK = 256


# ══════════════════ 纯函数（可单测） ══════════════════

def entropy_norm(counts, norm_dim):
    """归一化 Shannon 熵：counts → 占比 → −Σp·ln p / ln(norm_dim)。"""
    n = float(np.sum(counts))
    if n <= 0 or norm_dim <= 1:
        return 0.0
    p = counts[counts > 0] / n
    return float(-np.sum(p * np.log(p)) / np.log(norm_dim))


def sync_metrics(buckets_c, dates_c, w_lo, w_hi, B, D, rng, perm_n, chunk=PERM_CHUNK):
    """观测 + 置换（无放回抽样）。返回 (sync, p, h_pos, h_time, N)。"""
    M = len(buckets_c)
    in_w = (buckets_c >= w_lo) & (buckets_c <= w_hi)
    idx = np.nonzero(in_w)[0]
    N = len(idx)
    if N < 2:
        return 0.0, 1.0, 0.0, 0.0, N

    def _calc(sel):  # sel: 选中评论的下标数组
        hp = entropy_norm(np.bincount(buckets_c[sel], minlength=B), B)
        ht = entropy_norm(np.bincount(dates_c[sel], minlength=D), D)
        return (1.0 - hp) * ht

    h_pos_obs = entropy_norm(np.bincount(buckets_c[idx], minlength=B), B)
    h_time_obs = entropy_norm(np.bincount(dates_c[idx], minlength=D), D)
    sync_obs = (1.0 - h_pos_obs) * h_time_obs

    ge = 0
    done = 0
    # 每次 chunk 的抽样数自适应（keys 矩阵内存上界 ~2M 元素）
    chunk = max(1, min(chunk, 2_000_000 // max(M, 1)))
    while done < perm_n:
        m = min(chunk, perm_n - done)
        # 均匀无放回抽样：随机 keys + argpartition（legacy RandomState.choice 的多维 size 在
        # replace=False 下不可用——2026-10-13 实测 ValueError；此处为等价实现，种子固定仍幂等）
        if N < M:
            draw = np.argpartition(rng.random_sample((m, M)), N, axis=1)[:, :N]
        else:
            draw = np.tile(np.arange(M, dtype=np.int64), (m, 1))
        # 行内直方图：线性索引 + 单次 bincount（快于逐行）
        rows = np.repeat(np.arange(m), N)
        flat_b = rows * B + buckets_c[draw].ravel()
        hist_b = np.bincount(flat_b, minlength=m * B).reshape(m, B)
        flat_d = rows * D + dates_c[draw].ravel()
        hist_d = np.bincount(flat_d, minlength=m * D).reshape(m, D)
        with np.errstate(divide='ignore', invalid='ignore'):
            pb = hist_b / N
            hb = -np.where(pb > 0, pb * np.log(pb), 0.0).sum(axis=1) / np.log(B)
            pd = hist_d / N
            hd = -np.where(pd > 0, pd * np.log(pd), 0.0).sum(axis=1) / np.log(D)
        sync_rand = (1.0 - hb) * hd
        ge += int(np.sum(sync_rand >= sync_obs))
        done += m
    return sync_obs, ge / perm_n, h_pos_obs, h_time_obs, N


# ══════════════════ Spark 作业 ══════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--config', default='/opt/work/config/params.yaml')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    ap.add_argument('--local', action='store_true')
    args = ap.parse_args()
    wh = args.wh_root

    import yaml
    with open(args.config, encoding='utf-8') as f:
        params = yaml.safe_load(f)
    peak_win = int(params['sync']['peak_win'])
    perm_n = int(params['sync']['perm_n'])
    bucket_sec = int(params['detect']['bucket_sec'])

    b = (SparkSession.builder.appName('m2_l2_sync')
         .config('spark.sql.shuffle.partitions', '8'))
    if args.local:
        b = b.master('local[2]').config('spark.driver.memory', '2g')
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    # ⚠️ 自读自写保护：本作业回写 dws_rhythm_events（overwrite）——必须先物化 ev 再写，
    #    否则惰性扫描在 overwrite 删除旧文件后触发 FileNotFound（2026-10-13 实测踩坑）
    ev = spark.read.parquet(f'{wh}/dws_rhythm_events').cache()
    ev.count()
    dwd = (spark.read.parquet(f'{wh}/dwd_danmaku_clean')
           .filter(~F.col('is_spam'))
           .filter(F.col('time_offset').isNotNull())
           .select('bvid', 'batch_date', 'time_offset', 'send_date',
                   F.floor(F.col('time_offset') / bucket_sec).cast('int').alias('bucket')))

    # ── 逐视频：事件 × 弹幕 → 观测 + 置换 ──
    # ⚠️ cogroup 自身完成分组——两个输入必须是**原始 (k,v) RDD**；若先 groupByKey 再 cogroup
    #    会双重分组（组值变为 ResultIterable 的 ResultIterable），实测踩坑（2026-10-13，见 CHANGELOG）
    ev_r = (ev.select('bvid', 'batch_date', 'event_id', 'start_bucket', 'end_bucket')
            .rdd.map(lambda r: ((r['bvid'], r['batch_date']),
                                (int(r['event_id']), int(r['start_bucket']), int(r['end_bucket'])))))
    cm_r = dwd.rdd.map(lambda r: ((r['bvid'], r['batch_date']), (int(r['bucket']), r['send_date'])))

    def per_video(key_ev_cm):
        (bvid, batch_date), (evs, cms) = key_ev_cm
        evs = sorted(evs)
        cms = list(cms)
        if not evs or not cms:
            return []
        buckets_c = np.array([c[0] for c in cms], dtype=np.int64)
        # 日期 → 视频内 id（D = 视频日跨度）
        dates = sorted({c[1] for c in cms})
        dmap = {d: i for i, d in enumerate(dates)}
        dates_c = np.array([dmap[c[1]] for c in cms], dtype=np.int64)
        B = int(buckets_c.max()) + 1
        D = len(dates)
        rng = np.random.RandomState(zlib.crc32(f'{bvid}|{batch_date}'.encode()) % (2 ** 31 - 1))
        out = []
        for eid, s, e in evs:
            sync, p, hp, ht, nw = sync_metrics(buckets_c, dates_c,
                                               s - peak_win, e + peak_win, B, D, rng, perm_n)
            out.append((bvid, batch_date, eid, float(sync), float(p),
                        float(hp), float(ht), int(nw)))
        return out

    sync_schema = T.StructType([
        T.StructField('bvid', T.StringType()), T.StructField('batch_date', T.StringType()),
        T.StructField('event_id', T.IntegerType()),
        T.StructField('sync_score', T.DoubleType()), T.StructField('sync_p', T.DoubleType()),
        T.StructField('h_pos', T.DoubleType()), T.StructField('h_time', T.DoubleType()),
        T.StructField('n_window', T.IntegerType()),
    ])
    sync_ev = spark.createDataFrame(ev_r.cogroup(cm_r).flatMap(per_video), schema=sync_schema)
    sync_ev.cache()
    n_sync = sync_ev.count()
    n_sig = sync_ev.filter(F.col('sync_p') < 0.05).count()

    # ── 回写事件表（保留原列 + 同步列）──
    ev2 = (ev.join(sync_ev, ['bvid', 'batch_date', 'event_id'], 'left')
           .select('bvid', 'batch_date', 'event_id', 'start_bucket', 'end_bucket',
                   'start_sec', 'end_sec', 'score_fused',
                   'score_d1', 'score_d2', 'score_d3', 'score_d4', 'n_methods',
                   'sync_score', 'sync_p', 'h_pos', 'h_time', 'n_window',
                   'calibrated_on', 'engine_version'))
    ev2.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_rhythm_events')
    print(f'↩️ dws_rhythm_events 已回写同步列：{n_sync} 事件（p<0.05：{n_sig}）')

    # ── 羊群权重（08 §4.4）──
    from pyspark.sql import Window
    w_univ = (spark.read.parquet(f'{wh}/dwd_danmaku_clean')
              .filter(~F.col('is_spam') & ~F.col('is_ad') & ~F.col('is_meaningless')
                      & ~F.col('is_subtitle_pool'))
              .select('bvid', 'batch_date', 'text', 'send_time'))
    kt = w_univ.groupBy('bvid', 'batch_date', 'text').agg(F.count('*').alias('k_t'))
    wran = Window.partitionBy('bvid', 'batch_date', 'text').orderBy('send_time')
    herd = (w_univ.withColumn('j', F.row_number().over(wran))
            .join(kt, ['bvid', 'batch_date', 'text'])
            .withColumn('herd_w', F.col('j') / F.col('k_t'))
            .groupBy('bvid', 'batch_date')
            .agg(F.avg('herd_w').alias('herd_mean')))

    # ── 视频级聚合（含无事件视频 → n_events=0）──
    agg = (sync_ev.groupBy('bvid', 'batch_date')
           .agg(F.count('*').alias('n_events'),
                F.sum(F.when(F.col('sync_p') < 0.05, 1).otherwise(0)).alias('n_sig05'),
                F.expr('percentile(sync_score, 0.5)').alias('sync_med'),
                F.expr('percentile(sync_p, 0.5)').alias('sync_p_med')))
    base = (spark.read.parquet(f'{wh}/dws_video_rhythm_features')
            .select('bvid', 'batch_date').distinct())
    out = (base.join(agg, ['bvid', 'batch_date'], 'left')
           .join(herd, ['bvid', 'batch_date'], 'left')
           .withColumn('n_events', F.coalesce(F.col('n_events'), F.lit(0)))
           .withColumn('engine_version', F.lit(ENGINE_VERSION))
           .select('bvid', 'batch_date', 'n_events', 'n_sig05',
                   'sync_med', 'sync_p_med', 'herd_mean', 'engine_version'))
    out.write.mode('overwrite').partitionBy('batch_date').parquet(f'{wh}/dws_sync_features')
    n_v = out.count()
    n_herd = out.filter(F.col('herd_mean').isNotNull()).count()

    report = {'job': 'm2_l2_sync', 'engine_version': ENGINE_VERSION,
              'n_events': n_sync, 'n_events_sig_p05': n_sig, 'perm_n': perm_n,
              'n_video_batches': n_v, 'n_herd_mean': n_herd,
              'sync_coverage': f'{n_sync}/{ev.count()}'}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'm2_l2_sync_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'✅ m2_l2_sync：同步性覆盖 {n_sync} 事件（p<0.05 {n_sig}）｜视频级 {n_v} 行（herd {n_herd}）')
    spark.stop()


if __name__ == '__main__':
    main()