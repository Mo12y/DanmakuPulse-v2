# danmaku-spark — apache/spark:3.5.4-java17 + python3（spark-py 仓库已被镜像源拒绝，官方非py镜像自建派生）
# B0：仅需标准库；B1 起加 python3-yaml（m0_full/m1_reconcile 读 config/params.yaml——配置中心 08 §0.3）
FROM apache/spark:3.5.4-java17
USER root
# apt 换清华源（jammy 为 sources.list；24.04 为 ubuntu.sources，两种都处理）
RUN sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g; s|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list 2>/dev/null || true \
 && sed -i 's|archive.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g; s|security.ubuntu.com|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-yaml \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3 /usr/bin/python
ENV PYSPARK_PYTHON=/usr/bin/python3 \
    PYSPARK_DRIVER_PYTHON=/usr/bin/python3
# 归属 /opt/spark/work-dir（官方镜像 WORKDIR，root 下无碍）
WORKDIR /opt/spark/work-dir
