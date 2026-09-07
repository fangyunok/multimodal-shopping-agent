from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from .agent import ShoppingAgent
from .fusion_retrieval import ScoreFusionRetriever
from .models import AgentRequest, AgentResponse, SearchHit, SearchRequest
from .retrieval import HybridRetriever
from .semantic_retrieval import SemanticRetriever
from .tools import ShoppingTools, ToolDefinition

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"


@lru_cache
def get_tools() -> ShoppingTools:
    catalog = Path(os.getenv("CATALOG_PATH", ROOT / "data" / "products.jsonl"))
    backend = os.getenv("RETRIEVER_BACKEND", "baseline").lower()
    if backend in {"clip", "fusion"}:
        from .encoders import ChineseClipEncoder

        model_name = os.getenv("CLIP_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
        model_path = os.getenv("CLIP_MODEL_PATH", model_name)
        device = os.getenv("MODEL_DEVICE", "cpu")
        encoder = ChineseClipEncoder(model_path, device)
        index_dir = os.getenv("INDEX_DIR")
        retriever = (
            SemanticRetriever.from_index(catalog, index_dir, encoder, model_name)
            if index_dir
            else SemanticRetriever.from_jsonl(catalog, encoder)
        )
        if backend == "fusion":
            retriever = ScoreFusionRetriever(
                HybridRetriever.from_jsonl(catalog), retriever, float(os.getenv("LEXICAL_WEIGHT", "0.65"))
            )
        return ShoppingTools(retriever)
    if backend != "baseline":
        raise ValueError(f"不支持的 RETRIEVER_BACKEND: {backend}")
    return ShoppingTools(HybridRetriever.from_jsonl(catalog))


app = FastAPI(title="Multimodal Shopping Agent", version="0.1.0")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "retriever_backend": os.getenv("RETRIEVER_BACKEND", "baseline").lower()}


@app.get("/tools", response_model=list[ToolDefinition])
def tool_definitions() -> list[ToolDefinition]:
    return get_tools().definitions()


@app.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest) -> list[SearchHit]:
    return get_tools().search_products(request)


@app.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest) -> AgentResponse:
    if request.image_path:
        raise HTTPException(status_code=400, detail="API 不接受本地图片路径，请使用 /agent/image 上传图片")
    return ShoppingAgent(get_tools()).run(request)


@app.post("/agent/image", response_model=AgentResponse)
async def agent_with_image(
    image: UploadFile = File(...),
    query: str = Form(default=""),
    max_price: float | None = Form(default=None),
    category: str | None = Form(default=None),
    top_k: int = Form(default=5, ge=1, le=20),
) -> AgentResponse:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="仅支持 JPEG、PNG 和 WebP 图片")
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 5 MB")
    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[image.content_type]
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(content)
            temp_path = Path(temp.name)
        try:
            with Image.open(temp_path) as uploaded:
                uploaded.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise HTTPException(status_code=422, detail="上传内容不是有效图片") from error
        request = AgentRequest(
            query=query,
            image_path=str(temp_path),
            max_price=max_price,
            category=category,
            top_k=top_k,
        )
        return ShoppingAgent(get_tools()).run(request)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
