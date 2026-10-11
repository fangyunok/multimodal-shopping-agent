from __future__ import annotations

import os
import asyncio
import json
import threading
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from .web_presenter import cards_from_trace, public_trace
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from .agent import ShoppingAgent
from .auth import auth_config, identify, session_key
from .inventory import InventoryUnavailable, inventory_from_env
from .llm_reasoner import LLMReasoner
from .fusion_retrieval import ScoreFusionRetriever
from .llm_planner import LLMPlanner
from .models import AgentRequest, AgentResponse, SearchHit, SearchRequest
from .planner import RulePlanner
from .react import ReActAgent, RuleReasoner
from .retrieval import HybridRetriever
from .semantic_retrieval import SemanticRetriever
from .session import ContextBudget
from .session_store import SessionConflict, SessionHandle
from .tools import ShoppingTools, ToolDefinition

ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def build_encoder():
    """按配置构建编码器：远端 HTTP 编码服务，或本进程内的本地模型。

    设了 ``ENCODER_BASE_URL`` 就走远端（``scripts/serve_encoder.py``）。
    这样检索副本完全不带 torch：CPU 侧可以廉价地起十几个副本，
    编码集中在一台 GPU 机器上按批次吃满算力——这是阶段三的拆分点。
    两条路径实现的是同一个 ``MultimodalEncoder`` 协议，检索层无感。
    """
    base_url = os.getenv("ENCODER_BASE_URL")
    if base_url:
        from .encoder_service import RemoteEncoder

        return RemoteEncoder(base_url)
    from .encoders import ChineseClipEncoder

    model_name = os.getenv("CLIP_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
    return ChineseClipEncoder(os.getenv("CLIP_MODEL_PATH", model_name), os.getenv("MODEL_DEVICE", "cpu"))


def ann_runtime_config():
    """ANN 的**运行期旋钮**：efSearch / nprobe 这类改了就生效、不需要重建索引的参数。

    这里刻意读不到 HNSW 的 ``m`` / ``ef_construction``，也读不到 IVF 的 ``nlist``：
    那些是构建期参数，改了必须重建。把它们暴露成环境变量只会让人以为改了就生效。
    """
    from .ann_index import AnnIndexConfig

    return AnnIndexConfig(
        kind=os.getenv("ANN_KIND", "hnsw"),  # type: ignore[arg-type]
        dimension=int(os.getenv("ANN_DIMENSION", "512")),
        ef_search=int(os.getenv("ANN_EF_SEARCH", "64")),
        nprobe=int(os.getenv("ANN_NPROBE", "32")),
    )


def ann_runtime_kwargs() -> dict[str, int]:
    """重排候选数（0 表示关闭）。

    只在这里读一次：让"没配重排"和"显式关掉重排"走同一条路径，
    避免出现"配了 0 却仍在重排"这种要读两处代码才能发现的偏差。
    """
    candidates = int(os.getenv("ANN_RERANK_CANDIDATES", "0"))
    return {"rerank_candidates": candidates} if candidates else {}


@lru_cache
def get_tools() -> ShoppingTools:
    catalog = Path(os.getenv("CATALOG_PATH", ROOT / "data" / "products.jsonl"))
    backend = os.getenv("RETRIEVER_BACKEND", "baseline").lower()
    if backend in {"ann", "partitioned"}:
        model_name = os.getenv("CLIP_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
        index_dir = os.getenv("INDEX_DIR")
        if not index_dir:
            raise ValueError(f"RETRIEVER_BACKEND={backend} 需要设置 INDEX_DIR")
        encoder = build_encoder()
        config = ann_runtime_config()
        if backend == "partitioned":
            from .partitioned import PartitionedRetriever

            retriever = PartitionedRetriever.from_index(
                catalog, index_dir, encoder, config, **ann_runtime_kwargs()
            )
        else:
            from .faiss_retrieval import FaissRetriever

            retriever = FaissRetriever.from_index(
                catalog, index_dir, encoder, model_name, config, **ann_runtime_kwargs()
            )
        return ShoppingTools(retriever, inventory_from_env(retriever.products))
    if backend in {"clip", "fusion"}:
        model_name = os.getenv("CLIP_MODEL", "OFA-Sys/chinese-clip-vit-base-patch16")
        encoder = build_encoder()
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
        return ShoppingTools(retriever, inventory_from_env(retriever.products))
    if backend != "baseline":
        raise ValueError(f"不支持的 RETRIEVER_BACKEND: {backend}")
    retriever = HybridRetriever.from_jsonl(catalog)
    return ShoppingTools(retriever, inventory_from_env(retriever.products))


def get_reasoner():
    # 每个 HTTP 请求独立，模型故障状态不会跨会话污染。
    backend = os.getenv("REASONER_BACKEND", "rule").lower()
    if backend == "rule":
        return RuleReasoner()
    if backend != "llm" or not os.getenv("LLM_BASE_URL") or not os.getenv("LLM_MODEL"):
        raise ValueError("REASONER_BACKEND 必须为 rule 或已配置地址和模型的 llm")
    return LLMReasoner(
        os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"], os.getenv("LLM_API_KEY", ""),
        timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "10")),
        max_input_tokens=int(os.getenv("LLM_MAX_INPUT_TOKENS", "8192")),
    )


@lru_cache
def get_session_store():
    """会话存储。默认进程内实现，设 SESSION_BACKEND=redis 后变成多副本共享。

    注意这里也必须 `lru_cache`：每次请求新建一个 Redis 连接池会在高并发下把
    文件描述符耗光，而连接池本来就是为复用而存在的。
    """
    from .session_store import create_session_store

    return create_session_store()


@lru_cache
def get_planner():
    tools = get_tools()
    categories = [product.category for product in tools.retriever.products]
    backend = os.getenv("PLANNER_BACKEND", "rule").lower()
    if backend == "rule":
        return RulePlanner(categories)
    if backend == "llm":
        endpoint = os.getenv("LLM_BASE_URL")
        model = os.getenv("LLM_MODEL")
        if not endpoint or not model:
            raise ValueError("LLM Planner 需要设置 LLM_BASE_URL 和 LLM_MODEL")
        return LLMPlanner(categories, endpoint, model, os.getenv("LLM_API_KEY", ""))
    raise ValueError(f"不支持的 PLANNER_BACKEND: {backend}")


app = FastAPI(title="Multimodal Shopping Agent", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024


@app.middleware("http")
async def authenticate(request: Request, call_next):
    if request.url.path in {"/", "/healthz", "/readyz"} or request.url.path.startswith("/static/"):
        return await call_next(request)
    try:
        mode, keys = auth_config()
    except ValueError:
        return JSONResponse(status_code=503, content={"detail": "服务鉴权配置无效"})
    principal = "demo" if mode == "demo" else identify(request.headers.get("Authorization", ""), keys)
    if principal is None:
        return JSONResponse(status_code=401, content={"detail": "需要有效的访问令牌"},
                            headers={"WWW-Authenticate": "Bearer"})
    request.state.principal = principal
    return await call_next(request)


@app.exception_handler(InventoryUnavailable)
async def inventory_unavailable(request: Request, error: InventoryUnavailable):
    return JSONResponse(status_code=503, content={"detail": "库存服务暂时不可用，当前库存未知"})


@app.get("/", include_in_schema=False)
def demo_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "ui" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "retriever_backend": os.getenv("RETRIEVER_BACKEND", "baseline").lower(),
        "planner_backend": os.getenv("PLANNER_BACKEND", "rule").lower(),
    }


@app.get("/healthz")
def liveness() -> dict[str, str]:
    """存活探针：进程还在就返回 200。

    刻意不检查检索器与存储——它们挂了应该由就绪探针拦住流量，
    而不是让编排系统把进程杀掉重启（重启并不能修好一个挂掉的 Redis）。
    """
    return {"status": "alive"}


@app.get("/readyz")
def readiness(response: Response) -> dict[str, object]:
    """就绪探针：确认检索器已加载、索引版本可读、会话存储可达。

    多副本部署时这三件事任何一件不成立，这个副本都不该接流量。
    """
    checks: dict[str, object] = {}
    healthy = True
    try:
        mode, _ = auth_config()
        checks["auth"] = {"mode": mode, "ok": True}
        checks["reasoner"] = {"backend": get_reasoner().backend, "configured": True}
    except ValueError:
        healthy = False
        checks["configuration"] = {"ok": False}
    try:
        tools = get_tools()
        retriever = tools.retriever
        checks["retriever"] = {"products": len(retriever.products), "ok": True}
        checks["inventory"] = {"source": tools.inventory.source, "live_check": False}
        # 带 vectors.npy 的索引才有精确兜底能力，这是极端过滤下的正确性保障。
        describe = getattr(retriever, "describe", None)
        if callable(describe):
            checks["index"] = describe()
    except Exception as error:  # noqa: BLE001 - 就绪探针必须把异常转成状态而不是抛穿
        healthy = False
        checks["retriever"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    try:
        store = get_session_store()
        reachable = store.ping()
        healthy = healthy and reachable
        checks["session_store"] = {
            "backend": type(store).__name__,
            "reachable": reachable,
            "ttl_seconds": int(os.getenv("SESSION_TTL_SECONDS", "3600")),
        }
    except Exception as error:  # noqa: BLE001
        healthy = False
        checks["session_store"] = {"reachable": False, "error": f"{type(error).__name__}: {error}"}
    response.status_code = 200 if healthy else 503
    return {"status": "ready" if healthy else "degraded", "checks": checks}


@app.get("/index")
def index_status() -> dict[str, object]:
    """索引版本状态：当前服务的是哪一个版本、有哪些版本可回滚。"""
    root = os.getenv("INDEX_ROOT")
    if not root:
        return {"versions": None, "note": "未设置 INDEX_ROOT，当前未使用版本化索引"}
    from .incremental import IndexVersionManager

    return IndexVersionManager(root).status()


@app.get("/tools", response_model=list[ToolDefinition])
def tool_definitions() -> list[ToolDefinition]:
    return get_tools().definitions()


@app.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest) -> list[SearchHit]:
    if request.image_path:
        raise HTTPException(status_code=400, detail="API 不接受本地图片路径，请使用 /agent/image 上传图片")
    return get_tools().search_products(request)


@app.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest) -> AgentResponse:
    if request.image_path:
        raise HTTPException(status_code=400, detail="API 不接受本地图片路径，请使用 /agent/image 上传图片")
    return ShoppingAgent(get_tools(), get_planner()).run(request)


class ChatRequest(BaseModel):
    model_config = {"extra": "forbid"}
    session_id: str = Field(min_length=1, max_length=128, description="调用方自带的会话标识")
    query: str = Field(min_length=1, max_length=4000, description="本轮用户输入")
    intent: Literal["auto", "search", "compare", "inventory"] = "auto"


    max_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    category: str | None = Field(default=None, max_length=100)
    product_ids: list[str] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str
    tool_calls: int
    stop_reason: str
    context: dict
    cards: list[dict] = Field(default_factory=list)
    tool_trace: list[dict] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)
    conflicts: int = Field(description="成功提交时为 0；版本冲突返回 409，已提交状态保持不变")
    citations: list[str] = Field(default_factory=list)


def save_ui_turn(memory, query, response):
    memory.ui_history.extend([
        {"role": "user", "text": query},
        {"role": "assistant", "text": response.answer, "reply": response.model_dump(mode="json")},
    ])
    memory.ui_history = memory.ui_history[-40:]


def _chat(request: ChatRequest, principal: str, on_progress=None, cancelled=None):
    budget = ContextBudget(max_tokens=int(os.getenv("CONTEXT_BUDGET", "2048")))
    handle = SessionHandle(get_session_store(), session_key(principal, request.session_id), budget)
    try:
        reasoner = get_reasoner()
    except ValueError:
        raise HTTPException(status_code=503, detail="多轮决策配置无效") from None
    try:
        with handle as memory:
            runner = ReActAgent(get_tools(), reasoner=reasoner,
                max_steps=int(os.getenv("MAX_STEPS", "6")), budget=budget, session=memory)
            result = runner.run(request.query, request.intent,
                filters=request.model_dump(include=request.model_fields_set & {"max_price", "category"}),
                product_ids=request.product_ids, on_progress=on_progress, cancelled=cancelled)
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("request disconnected")
            response = ChatResponse(session_id=request.session_id, answer=result.answer,
                intent=result.intent, tool_calls=result.tool_calls, stop_reason=result.stop_reason,
                context=result.context, conflicts=0, citations=result.citations,
                cards=cards_from_trace(result.trace(), get_tools()),
                tool_trace=public_trace(result.trace()),
                constraints=memory.state.model_dump(include={"max_price", "category", "excluded_product_ids"}))
            save_ui_turn(memory, request.query, response)
    except SessionConflict as error:
        raise HTTPException(409, "会话已更新，请读取最新会话后重试") from error
    return response


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    return _chat(request, http_request.state.principal)


@app.get("/chat/history")
def chat_history(http_request: Request, session_id: str = Query(min_length=1, max_length=128)):
    memory = get_session_store().load(session_key(http_request.state.principal, session_id))
    return {"messages": memory.ui_history,
            "constraints": memory.state.model_dump(include={"max_price", "category", "excluded_product_ids"})}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, http_request: Request):
    # The shared bounded thread pool runs synchronous tools. SSE contains observed
    # tool lifecycle events, never generated progress or internal reasoning.
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    cancelled = threading.Event()

    def emit(event, data):
        if not cancelled.is_set():
            loop.call_soon_threadsafe(events.put_nowait, (event, data))

    def work():
        try:
            result = _chat(request, http_request.state.principal,
                           lambda data: emit("progress", data), cancelled)
            emit("result", result.model_dump(mode="json"))
        except InterruptedError:
            pass
        except HTTPException as error:
            emit("error", {"status": error.status_code, "detail": error.detail})
        except Exception:
            emit("error", {"status": 503, "detail": "服务暂时不可用，请恢复会话检查结果后再重试。"})
        finally:
            emit("done", {})

    async def generate():
        task = asyncio.create_task(run_in_threadpool(work))
        try:
            while True:
                try:
                    event, data = await asyncio.wait_for(events.get(), timeout=10)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if event == "done":
                    break
                yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        finally:
            cancelled.set()
            # Cancellation is cooperative between calls; the current external call
            # keeps its own timeout and is not falsely reported as terminated.
            await task
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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
    # 同步模型、图片处理和临时文件生命周期都放到框架的有界工作线程池。
    # 客户端取消等待时，工作线程仍负责在执行结束后清理文件。
    return await run_in_threadpool(
        _run_uploaded_image, content, image.content_type, query, max_price, category, top_k,
    )


def _run_uploaded_image(
    content: bytes, content_type: str, query: str,
    max_price: float | None, category: str | None, top_k: int,
    excluded_product_ids: list[str] | None = None,
) -> AgentResponse:
    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[content_type]
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(content)
            temp_path = Path(temp.name)
        try:
            with Image.open(temp_path) as uploaded:
                uploaded.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
            raise HTTPException(status_code=422, detail="上传内容不是有效图片") from error
        request = AgentRequest(
            query=query,
            image_path=str(temp_path),
            max_price=max_price,
            category=category,
            top_k=top_k,
            excluded_product_ids=excluded_product_ids or [],
            intent="search",
        )
        return ShoppingAgent(get_tools(), get_planner()).run(request)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)



@app.get("/ui/config")
def ui_config():
    tools = get_tools()
    image_available = (bool(tools.retriever.image_vectors)
                       if isinstance(tools.retriever, HybridRetriever) else True)
    return {"categories": sorted({p.category for p in tools.retriever.products}),
            "product_count": len(tools.retriever.products),
            "inventory_source": tools.inventory.source,
            "reasoner_backend": os.getenv("REASONER_BACKEND", "rule"),
            "image_search_available": image_available}


@app.post("/chat/image", response_model=ChatResponse)
async def chat_image(http_request: Request, image: UploadFile = File(...),
                     session_id: str = Form(min_length=1, max_length=128),
                     query: str = Form(default="", max_length=4000),
                     max_price: float | None = Form(default=None, ge=0, allow_inf_nan=False),
                     category: str | None = Form(default=None, max_length=100),
                     filters: str = Form(default="{}", max_length=1000)):
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(415, "仅支持 JPEG、PNG 和 WebP 图片")
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "图片不能超过 5 MB")
    try:
        parsed_filters = ImageFilters.model_validate_json(filters).model_dump(exclude_unset=True)
    except ValidationError:
        raise HTTPException(422, "图片筛选条件无效") from None
    return await run_in_threadpool(_image_chat, http_request.state.principal, session_id,
                                  content, image.content_type, query, max_price, category, parsed_filters)


class ImageFilters(BaseModel):
    model_config = {"extra": "forbid"}
    max_price: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    category: str | None = Field(default=None, max_length=100)


def _image_chat(principal, session_id, content, content_type, query, max_price, category, filters=None):
    config = ui_config()
    if not config["image_search_available"] and not query.strip():
        raise HTTPException(422, "当前演示目录没有商品图片，请补充文字描述；以图找货需要接入图文索引。")
    budget = ContextBudget(max_tokens=int(os.getenv("CONTEXT_BUDGET", "2048")))
    handle = SessionHandle(get_session_store(), session_key(principal, session_id), budget)
    try:
        with handle as memory:
            plan = RulePlanner(config["categories"]).plan(query, "search")
            memory.begin_turn(query or "以图找货", plan)
            if max_price is not None:
                memory.state.max_price = max_price
            if category:
                memory.state.category = category
            for name, value in (filters or {}).items():
                setattr(memory.state, name, value)
            result = _run_uploaded_image(content, content_type, query,
                memory.state.max_price, memory.state.category, 3, memory.state.excluded_product_ids)
            memory.record_observation("search_products", result.hits)
            for entry in result.tool_trace:
                if entry["tool"] == "check_inventory":
                    memory.record_observation("check_inventory", entry["result"])
            if not config["image_search_available"]:
                result.answer = "当前目录没有商品图片，本次仅按文字检索。" + result.answer
            memory.record_answer(result.answer)
            context = memory.window().stats()
            context["inventory_source"] = get_tools().inventory.source
            constraints = memory.state.model_dump(include={"max_price", "category", "excluded_product_ids"})
            trace = [{"tool": "search_products", "result": result.hits}] + result.tool_trace
            response = ChatResponse(session_id=session_id, answer=result.answer, intent="search",
                tool_calls=len(result.tool_trace), stop_reason="final_answer", context=context,
                conflicts=0, citations=result.citations, cards=cards_from_trace(trace, get_tools()),
                tool_trace=public_trace(result.tool_trace), constraints=constraints)
            save_ui_turn(memory, query + " [参考图片]", response)
    except SessionConflict as error:
        raise HTTPException(409, "会话已更新，请读取最新会话后重试") from error
    return response
