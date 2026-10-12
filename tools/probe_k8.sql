-- K≥8 视频占比探针（B1.4 / 10 §2.2 训练池可行性）
-- K = 首采批次日与发布日的间隔天数（天粒度近似；0914 批跨 2.5 天窗口的偏差记入报告）
SELECT COUNT(*) AS n_videos,
       SUM(CASE WHEN k_days >= 8 THEN 1 ELSE 0 END) AS n_k8,
       ROUND(SUM(CASE WHEN k_days >= 8 THEN 1 ELSE 0 END) / COUNT(*), 4) AS k8_ratio,
       MIN(k_days) AS k_min, MAX(k_days) AS k_max,
       ROUND(AVG(k_days), 2) AS k_mean,
       ROUND(PERCENTILE(k_days, 0.5), 1) AS k_median
FROM (
  SELECT bvid,
         DATEDIFF(TO_DATE(MIN(batch_date), 'yyyyMMdd'),
                  FROM_UNIXTIME(MAX(pubdate), 'yyyy-MM-dd')) AS k_days
  FROM dwd_danmaku_clean
  GROUP BY bvid
) t;
