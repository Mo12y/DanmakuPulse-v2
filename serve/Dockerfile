FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# 国内 pip 镜像（tuna）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
COPY app.py dict_zh.py diagnosis.py ./
COPY v3data/ v3data/
COPY static/ static/
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
