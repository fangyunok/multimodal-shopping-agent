"""GPU 编码服务：把模型前向从检索进程里剥离出去。

检索副本因此可以完全不带 torch —— 它们是 CPU 侧的、可以廉价地起十几个；
编码器则集中在一台（或少数几台）GPU 机器上，按批次吃满算力。

启动：

    # 有 GPU 时
    pip install -e ".[clip,service]"
    ENCODER_DEVICE=cuda:0 python scripts/serve_encoder.py --host 0.0.0.0 --port 8100

    # 直接当 uvicorn 应用跑也可以
    uvicorn scripts.serve_encoder:app --port 8100

接口：

    POST /embed/text   {"texts": ["..."], ...}                 -> {"vectors": [[...]], "cached": n, "batches": n}
    POST /embed/image  {"images_base64": ["..."]}              -> {"vectors": [[...]]}
    GET  /healthz                                              -> 编码器与批处理统计
    GET  /metrics                                              -> 缓存命中率、平均批次大小等

为什么图片走 base64 而不是路径
-----------------------------
服务端不应该能读客户端文件系统。允许传路径，等于把"任意文件读取"直接暴露成一个接口——
这个项目在图片上传接口上已经踩过同一个坑（禁止传入本机路径），这里保持一致。
"""

from __future__ import annotations

import base64
import binascii
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from shopping_agent.encoder_service import BatchQueueFull, BatcherStopped, EmbeddingCache, CachedEncoder  # noqa: E402

MAX_ITEMS_PER_REQUEST = 512
MAX_IMAGE_BYTES = 5 * 1024 * 1024


class TextRequest(BaseModel):
    model_config = {"extra": "forbid"}
    texts: list[str] = Field(min_length=1, max_length=MAX_ITEMS_PER_REQUEST)


class ImageRequest(BaseModel):
    model_config = {"extra": "forbid"}
    images_base64: list[str] = Field(min_length=1, max_length=MAX_ITEMS_PER_REQUEST)


class EmbeddingResponse(BaseModel):
    vectors: list[list[float]]
    dimension: int
    items: int
    requested: int
    cache_hit_ratio: float
    avg_batch_size: float


_state: dict[str, object] = {}


def build_encoder() -> CachedEncoder:
    """按需构建真实编码器。测试环境可以用 ``set_encoder`` 注入假实现。"""
    from shopping_agent.encoders import ChineseClipEncoder

    model_name = os.getenv("CLIP_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
    device = os.getenv("ENCODER_DEVICE", "cuda:0" if _cuda_available() else "cpu")
    batch_size = int(os.getenv("ENCODER_MODEL_BATCH", "64"))
    inner = ChineseClipEncoder(model_name, device, batch_size)
    cache = EmbeddingCache(
        capacity=int(os.getenv("EMBEDDING_CACHE_SIZE", "4096")),
        redis_client=_redis_client(),
    )
    return CachedEncoder(
        inner,
        cache=cache,
        max_batch=int(os.getenv("ENCODER_MAX_BATCH", "256")),
        max_wait_ms=float(os.getenv("ENCODER_MAX_WAIT_MS", "8")),
        max_pending_items=int(os.getenv("ENCODER_MAX_PENDING_ITEMS", "1024")),
        queue_timeout_seconds=float(os.getenv("ENCODER_QUEUE_TIMEOUT_SECONDS", "5")),
    )


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _redis_client():
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis  # noqa: PLC0415

        return redis.Redis.from_url(url, decode_responses=False)
    except ImportError:  # pragma: no cover - 取决于环境
        return None


def set_encoder(encoder: CachedEncoder) -> None:
    """注入编码器（测试或自定义模型用）。"""
    _state["encoder"] = encoder


def get_encoder() -> CachedEncoder:
    if "encoder" not in _state:
        _state["encoder"] = build_encoder()
    return _state["encoder"]  # type: ignore[return-value]


app = FastAPI(title="Multimodal Encoder Service", version="0.1.0")


def _response(vectors: np.ndarray, requested: int) -> EmbeddingResponse:
    encoder = get_encoder()
    stats = encoder.stats()
    return EmbeddingResponse(
        vectors=[[round(float(value), 6) for value in row] for row in vectors],
        dimension=int(vectors.shape[1]) if vectors.size else 0,
        items=int(vectors.shape[0]),
        requested=requested,
        cache_hit_ratio=float(stats["cache"]["hit_rate"]),
        avg_batch_size=float(stats["text_batcher"]["avg_batch_size"]),
    )


@app.post("/embed/text", response_model=EmbeddingResponse)
def embed_texts(request: TextRequest) -> EmbeddingResponse:
    try:
        vectors = get_encoder().encode_texts(request.texts)
    except (BatchQueueFull, BatcherStopped) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
    return _response(vectors, len(request.texts))


@app.post("/embed/image", response_model=EmbeddingResponse)
def embed_images(request: ImageRequest) -> EmbeddingResponse:
    paths: list[Path] = []
    try:
        for index, payload in enumerate(request.images_base64):
            try:
                raw = base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError) as error:
                raise HTTPException(status_code=422, detail=f"第 {index} 张图片不是合法 base64") from error
            if len(raw) > MAX_IMAGE_BYTES:
                raise HTTPException(status_code=413, detail=f"第 {index} 张图片超过 5 MB")
            with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as temporary:
                temporary.write(raw)
                paths.append(Path(temporary.name))
        try:
            vectors = get_encoder().encode_images(paths)
        except (BatchQueueFull, BatcherStopped) as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except TimeoutError as error:
            raise HTTPException(status_code=504, detail=str(error)) from error
        return _response(vectors, len(paths))
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    encoder = get_encoder()
    return {"status": "ok", "encoder": type(encoder.encoder).__name__, **encoder.stats()}


@app.get("/metrics")
def metrics() -> dict[str, object]:
    return get_encoder().stats()


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="启动 GPU 编码服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

