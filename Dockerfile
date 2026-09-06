FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RETRIEVER_BACKEND=baseline \
    CATALOG_PATH=/app/data/products.jsonl

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/products.jsonl ./data/products.jsonl

RUN python -m pip install --upgrade pip && \
    python -m pip install .

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "shopping_agent.app:app", "--host", "0.0.0.0", "--port", "8000"]

