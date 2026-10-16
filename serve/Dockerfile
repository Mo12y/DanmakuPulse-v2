FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# 国内 pip 镜像（tuna）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
# 整目录拷 .py：新增模块不必再改这里
# （2026-10-16 实测：新增 live_diag.py 漏登记 -> 容器 ModuleNotFoundError 反复重启）
COPY *.py ./
COPY v3data/ v3data/
COPY static/ static/
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
