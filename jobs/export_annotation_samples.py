#!/usr/bin/env python3
"""
export_annotation_samples.py — B1.4 样本抽取（集群模式；driver 直写 out_dir）

输出（out_dir/annotation/，默认 /opt/work/data-out/annotation → 宿主机 data-out/annotation）：
  highlight_sample.csv    30 视频：分区分散（每 tname 弹幕量 top2 + 打散），含 video_url
  curve_all.csv           30 视频的 101 点密度曲线（辅助标注，来自 dws_video_curve_min）
  sent_sample.csv         1000 条弹幕：分区配额（每分区 ≤25）+ 文本去重 + 去 ultra_short，**不含机器标注**
  progress.csv            进度表模板（人工更新）
  k8 结果另行由 tools/probe_k8.sql 生成

纪律：抽样过程**不 join 任何情感/机器输出**（防标注锚定偏差）。
"""
import argparse
import csv
import os

import pyspark.sql.functions as F
from pyspark.sql import SparkSession, Window


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wh-root', default='/warehouse')
    ap.add_argument('--out-dir', default='/opt/work/data-out')
    args = ap.parse_args()
    out = os.path.join(args.out_dir, 'annotation')
    os.makedirs(out, exist_ok=True)

    spark = (SparkSession.builder.appName('export_annotation_samples')
             .config('spark.sql.shuffle.partitions', '8').getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')

    dwd = spark.read.parquet(f'{args.wh_root}/dwd_danmaku_clean')

    # ── A) 30 视频（分区分散）──
    v = (dwd.groupBy('bvid')
         .agg(F.max('title').alias('title'), F.max('tname').alias('tname'),
              F.max('duration').alias('duration'), F.max('batch_date').alias('batch_date'),
              F.count('*').alias('danmaku_count')))
    w = Window.partitionBy('tname').orderBy(F.col('danmaku_count').desc())
    cand = v.withColumn('rk', F.row_number().over(w)).filter('rk <= 2')
    sel = cand.orderBy(F.rand(7)).limit(30).orderBy('tname', F.desc('danmaku_count'))
    rows = sel.collect()
    write_csv(os.path.join(out, 'highlight_sample.csv'),
              ['bvid', 'title', 'tname', 'duration', 'batch_date', 'danmaku_count', 'video_url'],
              [[r['bvid'], r['title'], r['tname'], r['duration'], r['batch_date'],
                r['danmaku_count'], f'https://www.bilibili.com/video/{r["bvid"]}'] for r in rows])
    bvids = [r['bvid'] for r in rows]
    print(f'[A] highlight_sample.csv: {len(rows)} 视频（{len(set(r["tname"] for r in rows))} 个分区）')

    # ── A2) 曲线辅助 ──
    curve = spark.read.parquet(f'{args.wh_root}/dws_video_curve_min')
    cv = (curve.filter(F.col('bvid').isin(bvids))
          .select(F.posexplode('curve').alias('bucket', 'c'), 'bvid', 'batch_date'))
    cv_rows = cv.select('bvid', 'batch_date', 'bucket', 'c').collect()
    # 每 bvid 取一次（多批视频取第一行的 batch）
    seen = {}
    out_rows = []
    for r in sorted(cv_rows, key=lambda x: (x['bvid'], x['batch_date'], x['bucket'])):
        key = (r['bvid'], r['batch_date'])
        out_rows.append([r['bvid'], r['bucket'], float(r['c'])])
    write_csv(os.path.join(out, 'curve_all.csv'), ['bvid', 'bucket', 'c'], out_rows)
    print(f'[A2] curve_all.csv: {len(out_rows)} 点')

    # ── B) 1000 条情感样本（分区配额 + 文本去重）──
    t = (dwd.filter(~F.col('is_ultra_short'))
         .select('bvid', 'batch_date', 'text', 'tname'))
    td = t.groupBy('text').agg(F.first('bvid').alias('bvid'),
                               F.first('batch_date').alias('batch_date'),
                               F.first('tname').alias('tname'))
    w2 = Window.partitionBy('tname').orderBy(F.rand(11))
    q = td.withColumn('rk2', F.row_number().over(w2)).filter('rk2 <= 25')
    sel2 = q.orderBy(F.rand(13)).limit(1000)
    s_rows = sel2.collect()
    write_csv(os.path.join(out, 'sent_sample.csv'),
              ['sample_id', 'bvid', 'batch_date', 'text', 'label', 'unsure'],
              [[i + 1, r['bvid'], r['batch_date'], r['text'], '', ''] for i, r in enumerate(s_rows)])
    print(f'[B] sent_sample.csv: {len(s_rows)} 条（{len(set(r["tname"] for r in s_rows))} 个分区）')

    # ── 进度表模板 ──
    write_csv(os.path.join(out, 'progress.csv'),
              ['task', 'total', 'done', 'owner', 'updated', 'note'],
              [['A 高能标注', 30, 0, '', '', '待人工启动'],
               ['B 情感标注(人1)', len(s_rows), 0, '', '', '双人独立'],
               ['B 情感标注(人2)', len(s_rows), 0, '', '', '双人独立'],
               ['B 仲裁质检', '', 0, '', '', 'κ 复核']])
    print(f'✅ 样本导出完成 → {out}/')
    spark.stop()


if __name__ == '__main__':
    main()
