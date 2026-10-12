# danmaku-v2 — 基于 Spark 的 B 站热榜视频弹幕互动节奏分析系统（新系统）

> 旧仓库 `~/projects/danmaku` 已冻结为预研仓库（只读，`collect/bili.py` 以副本形式继续服务采集）。
> 设计基准：旧仓库 `docs/05-新系统重构/07_最终设计方案.md`（v1.4）+ `08_模块详细设计.md`。

## B0 地基周（当前批次）

```bash
make setup        # 一次性：pg jdbc jar + echarts（国内镜像）
make seed-data    # 一次性：历史 4 批三件套复制 + goldens 生成
make goldens      #   （对拍基准：旧 NumPy SSOT，只读旧仓库）
make stage        # cli_collected → data/raw-local 暂存
make up           # 9 服务底座（compose.b0.yml；DS/监控 B1 叠加 obs.yml）
make smoke        # B0.1 验收：HDFS/PG 读写冒烟 + Spark UI alive
make ingest       # raw-local → HDFS /raw/batch_date=*/
make etl          # m0_min → m2_l1_min → ads_min → register_hms
make parity       # B0.3 对拍验收（曲线+8维 vs goldens；池级锚点 508/669,995/509/672,439）
make replay       # 幂等验收：重跑 ETL + 规范导出容差比对
make cron-install # 日频采集 crontab（B0 期宿主 crontab；B1 起 DolphinScheduler 接管）
```

## 目录

```
collect/    bili.py（零改动副本）+ bili_daily.sh / daily_cron.sh（新 wrapper，不跑旧 ETL）
compose/    docker-compose.b0.yml（9 服务）/ hadoop.env / assets/（ingest、nginx、spark-conf）
jobs/       spark-submit 入口：m0_min / m2_l1_min / ads_min / parity_check / register_hms
tools/      export_goldens（对拍基准）/ stage_raw_local（批次映射唯一事实源）/ compare_canonical
serve/      FastAPI 只读 API + 静态验收页（nginx 反代）
config/     params.yaml（可调参数）/ batch_lineage.json（历史批次血缘与已知问题）
tests/      goldens/（对拍基准 CSV，由 make goldens 生成）
```

## 对拍与幂等（B0 DoD）

- **对拍**：新 Spark SQL 实现 vs 旧 NumPy SSOT（`analysis/danmaku_metrics.py`）——
  曲线 101 点 ≤1e-9；peak_count/peak_fwhm/danmaku_count 精确一致；池级锚点逐数对账。
- **幂等**：全部输出分区覆盖写 + ads 全量重建；重跑一致性 = 整数精确 + 浮点 ≤1e-9
  （不做 md5：Spark 聚合浮点位级受 shuffle 顺序影响，属预期行为）。
