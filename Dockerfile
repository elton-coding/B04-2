# B04 目标本体表征与多源数据融合 —— 选手方案镜像
FROM python:3.11-slim

WORKDIR /app/program

COPY requirements.txt /app/program/requirements.txt
RUN pip install --no-cache-dir -r /app/program/requirements.txt

COPY . /app/program/solution/
RUN chmod +x /app/program/solution/run.sh

WORKDIR /app/program/solution
ENV DATA_ROOT=/input
ENV OUTPUT_DIR=/output
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

ENTRYPOINT ["/app/program/solution/run.sh"]
CMD ["--task", "all"]
