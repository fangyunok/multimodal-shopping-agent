FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RETRIEVER_BACKEND=baseline \
    CATALOG_PATH=/app/data/products.jsonl

# 检索副本默认不带 faiss / torch：CPU 副本越轻，扩容越便宜。
# 需要 ANN 后端时用 --build-arg INSTALL_EXTRAS=ann 构建。
ARG INSTALL_EXTRAS=""

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/products.jsonl ./data/products.jsonl

RUN python -m pip install --upgrade pip && \
    if [ -n "$INSTALL_EXTRAS" ]; then python -m pip install ".[$INSTALL_EXTRAS]"; else python -m pip install .; fi

USER app
EXPOSE 8000

# 容器级健康检查用存活探针：它只回答"进程还在不在"。
# 下游依赖（Redis / 索引）挂了应该由编排层的 readinessProbe 打 /readyz 拦流量，
# 而不是让 Docker 反复重启一个本来没坏的进程。
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uvicorn", "shopping_agent.app:app", "--host", "0.0.0.0", "--port", "8000"]
