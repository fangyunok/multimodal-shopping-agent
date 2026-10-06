"""MCP server exposing the multimodal agent over the Model Context Protocol.

协议层：把检索、比价、库存、端到端 Agent 四类能力暴露为 MCP tools，商品目录与数据契约暴露为
resources，受约束的提示词暴露为 prompts；同时支持 stdio 与 sse / streamable-http 两种传输形态。

工程层：入参校验、超时、指数退避重试、只读工具幂等重放、并发闸门与逐工具延迟指标，均为
传输无关的横切能力，集中在本模块实现，业务代码（tools.py / agent.py）不需要感知。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import logging
import math
import os
import random
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, Field, ValidationError

from .agent import ShoppingAgent
from .models import AgentRequest, Product, SearchRequest
from .planner import RulePlanner
from .react import ReActAgent
from .retrieval import HybridRetriever
from .session import ContextBudget, SessionMemory
from .tools import ShoppingTools, ToolCall

try:  # pragma: no cover - 依赖缺失时的提示路径
    from mcp.server.mcpserver import MCPServer
except ModuleNotFoundError as error:  # pragma: no cover
    raise SystemExit(
        "运行 MCP Server 需要 mcp>=2.3（FastMCP 在 2.x 中更名为 MCPServer）。"
        "请先安装：pip install 'multimodal-shopping-agent[mcp]' 或 pip install 'mcp>=2.3,<3'"
    ) from error


logger = logging.getLogger("shopping_agent.mcp")

T = TypeVar("T")

DEFAULT_TOOL_TIMEOUT_SECONDS = 2.0
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_IDEMPOTENCY_TTL_SECONDS = 30.0
DEFAULT_MAX_REPLAY_ENTRIES = 1024
DEFAULT_BACKOFF_BASE_SECONDS = 0.02


class ServerConfig(BaseModel):
    """MCP Server 的运行期参数；全部可通过命令行或环境变量覆盖。"""

    name: str = "multimodal-shopping-agent"
    version: str = "0.1.0"
    timeout_seconds: float = Field(default=DEFAULT_TOOL_TIMEOUT_SECONDS, gt=0)
    max_retries: int = Field(default=1, ge=0)
    backoff_base_seconds: float = Field(default=DEFAULT_BACKOFF_BASE_SECONDS, ge=0)
    max_concurrency: int = Field(default=DEFAULT_MAX_CONCURRENCY, ge=1)
    idempotency_ttl_seconds: float = Field(default=DEFAULT_IDEMPOTENCY_TTL_SECONDS, ge=0)
    max_replay_entries: int = Field(
        default=DEFAULT_MAX_REPLAY_ENTRIES,
        ge=1,
        description="幂等重放缓存的条目上限；超出后先清理过期项，仍超限再淘汰最早写入的条目",
    )
    react_max_steps: int = Field(default=6, ge=1, description="单轮 ReAct 最大步数")
    session_context_budget: int = Field(
        default=2048, ge=512, description="每个会话的上下文 token 预算（需大于预留的回复额度）"
    )
    max_sessions: int = Field(default=128, ge=1, description="服务端保留的会话上限（超出按插入顺序淘汰）")


def percentile(samples: list[float], quantile: float) -> float | None:
    """最近秩法分位数；样本为空时返回 None。"""
    if not samples:
        return None
    ordered = sorted(samples)
    rank = max(1, math.ceil(quantile / 100 * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 3)


class ServerMetrics:
    """逐工具统计调用量、错误、重试、幂等命中与延迟分位数。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "errors": 0, "retries": 0, "replayed": 0, "latency_ms": []}
        )
        self.started_at = time.time()

    def record(
        self,
        tool: str,
        latency_ms: float,
        *,
        error: bool = False,
        retries: int = 0,
        replayed: bool = False,
    ) -> None:
        with self._lock:
            bucket = self._calls[tool]
            bucket["count"] += 1
            bucket["errors"] += int(error)
            bucket["retries"] += retries
            bucket["replayed"] += int(replayed)
            if not replayed:
                bucket["latency_ms"].append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            tools = {}
            for tool, bucket in sorted(self._calls.items()):
                samples = list(bucket["latency_ms"])
                tools[tool] = {
                    "calls": bucket["count"],
                    "errors": bucket["errors"],
                    "retries": bucket["retries"],
                    "replayed": bucket["replayed"],
                    "latency_ms": {
                        "p50": percentile(samples, 50),
                        "p95": percentile(samples, 95),
                        "p99": percentile(samples, 99),
                        "max": round(max(samples), 3) if samples else None,
                    },
                }
            return {
                "uptime_seconds": round(time.time() - self.started_at, 3),
                "total_calls": sum(bucket["count"] for bucket in self._calls.values()),
                "tools": tools,
            }


class ToolRuntime:
    """传输无关的横切能力：并发闸门、超时、退避重试、幂等重放、指标采集。

    同步 handler 会被卸载到线程池执行，因此超时可达、且并发请求能真正并行；CPU 密集的
    handler 在被取消后其线程仍会跑完，这一点在文档中已如实说明。
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.metrics = ServerMetrics()
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._replay_cache: dict[str, tuple[float, Any]] = {}
        self._replay_evictions = 0
        self._lock = threading.Lock()
        self._retryable = (TimeoutError, asyncio.TimeoutError, ConnectionError, OSError)

    @staticmethod
    def idempotency_key(tool: str, arguments: dict[str, Any]) -> str:
        canonical = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        digest = hashlib.sha256(f"{tool}:{canonical}".encode("utf-8")).hexdigest()
        return digest[:32]

    def _is_retryable(self, error: BaseException) -> bool:
        """重试只对瞬时故障有意义；权限与语义错误必须立刻失败。

        PermissionError 虽然是 OSError 的子类，但重试不会改变结果，因此显式排除。
        """
        if isinstance(error, PermissionError):
            return False
        return isinstance(error, self._retryable)

    async def _invoke(self, handler: Callable[[], Any]) -> Any:
        if inspect.iscoroutinefunction(handler):
            return await handler()
        return await asyncio.to_thread(handler)

    def _replay(self, key: str) -> Any | None:
        if self.config.idempotency_ttl_seconds <= 0:
            return None
        with self._lock:
            hit = self._replay_cache.get(key)
            if hit is None:
                return None
            stored_at, value = hit
            if time.monotonic() - stored_at > self.config.idempotency_ttl_seconds:
                self._replay_cache.pop(key, None)
                return None
            return value

    def _remember(self, key: str, value: Any) -> None:
        if self.config.idempotency_ttl_seconds <= 0:
            return
        with self._lock:
            now = time.monotonic()
            self._replay_cache[key] = (now, value)
            if len(self._replay_cache) <= self.config.max_replay_entries:
                return
            # 超过上限时先清过期项；仍超限再按写入顺序淘汰最旧条目，
            # 使长跑服务的内存占用有界，而不是随参数组合无限增长。
            ttl = self.config.idempotency_ttl_seconds
            stale_keys = [k for k, (stored_at, _) in self._replay_cache.items() if now - stored_at > ttl]
            for stale in stale_keys:
                self._replay_cache.pop(stale, None)
            while len(self._replay_cache) > self.config.max_replay_entries:
                self._replay_cache.pop(next(iter(self._replay_cache)), None)
                self._replay_evictions += 1

    def replay_cache_stats(self) -> dict[str, Any]:
        """重放缓存的可观测快照；entries 恒不超过 capacity。"""
        with self._lock:
            return {
                "entries": len(self._replay_cache),
                "capacity": self.config.max_replay_entries,
                "evicted": self._replay_evictions,
                "ttl_seconds": self.config.idempotency_ttl_seconds,
            }

    async def run(
        self,
        tool: str,
        arguments: dict[str, Any],
        handler: Callable[[], Any],
        *,
        idempotent: bool = True,
    ) -> Any:
        """执行工具。

        ``idempotent=False`` 用于**有状态**工具（会话式 ReAct）：重复调用会改变会话状态，
        按参数缓存并重放会返回陈旧快照并跳过状态更新，因此这类工具必须绕开重放缓存；
        同理它们也**不参与重试**——超时只代表没等到响应，handler 可能已经执行完毕，
        重跑会把副作用施加两次。
        """
        key = self.idempotency_key(tool, arguments)
        if idempotent:
            replayed = self._replay(key)
            if replayed is not None:
                self.metrics.record(tool, 0.0, replayed=True)
                return replayed

        allow_retry = idempotent and self.config.max_retries > 0
        started = time.perf_counter()
        attempts = 0
        async with self._semaphore:
            while True:
                try:
                    result = await asyncio.wait_for(self._invoke(handler), self.config.timeout_seconds)
                except Exception as error:
                    elapsed = (time.perf_counter() - started) * 1000
                    if not allow_retry or not self._is_retryable(error):
                        self.metrics.record(tool, elapsed, error=True, retries=attempts)
                        raise
                    attempts += 1
                    if attempts > self.config.max_retries:
                        self.metrics.record(tool, elapsed, error=True, retries=attempts - 1)
                        raise
                    delay = self.config.backoff_base_seconds * (2 ** (attempts - 1))
                    await asyncio.sleep(delay + random.uniform(0, self.config.backoff_base_seconds))
                    continue
                self.metrics.record(tool, (time.perf_counter() - started) * 1000, retries=attempts)
                if idempotent:
                    self._remember(key, result)
                return result


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _search_hit_payload(hit: Any) -> dict[str, Any]:
    product: Product = hit.product
    return {
        "product_id": product.id,
        "title": product.title,
        "category": product.category,
        "price": product.price,
        "rating": product.rating,
        "stock": product.stock,
        "score": hit.score,
        "text_score": hit.text_score,
        "image_score": hit.image_score,
        "reasons": list(hit.reasons),
    }


def _jsonable(value: Any) -> Any:
    """把工具返回值转成可 JSON 序列化的结构，保留嵌套模型的字段。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def load_tools(
    catalog: str | Path,
    retriever_backend: str = "baseline",
    index_dir: str | None = None,
    model: str = "OFA-Sys/chinese-clip-vit-base-patch16",
    device: str = "cpu",
    lexical_weight: float = 0.65,
) -> ShoppingTools:
    """构建检索后端；与 CLI / HTTP 服务保持一致的三后端语义（baseline / clip / fusion）。"""
    backend = retriever_backend.lower()
    if backend == "baseline":
        return ShoppingTools(HybridRetriever.from_jsonl(catalog))
    if backend not in {"clip", "fusion"}:
        raise ValueError(f"不支持的 RETRIEVER_BACKEND: {retriever_backend}")
    from .encoders import ChineseClipEncoder
    from .fusion_retrieval import ScoreFusionRetriever
    from .semantic_retrieval import SemanticRetriever

    encoder = ChineseClipEncoder(model, device)
    semantic = (
        SemanticRetriever.from_index(catalog, index_dir, encoder, model)
        if index_dir
        else SemanticRetriever.from_jsonl(catalog, encoder)
    )
    if backend == "fusion":
        return ShoppingTools(ScoreFusionRetriever(HybridRetriever.from_jsonl(catalog), semantic, lexical_weight))
    return ShoppingTools(semantic)


def build_server(
    tools: ShoppingTools,
    planner: Any | None = None,
    config: ServerConfig | None = None,
    allow_local_image_path: bool | None = None,
) -> MCPServer:
    """把 ShoppingTools / ShoppingAgent 装配成 MCP Server。"""
    config = config or ServerConfig()
    runtime = ToolRuntime(config)
    agent = ShoppingAgent(tools, planner)
    allow_image = (
        os.getenv("MCP_ALLOW_LOCAL_IMAGE_PATH", "0") == "1" if allow_local_image_path is None else allow_local_image_path
    )

    sessions: dict[str, SessionMemory] = {}
    sessions_lock = threading.Lock()

    def resolve_session(session_id: str) -> SessionMemory:
        """按 session_id 复用会话；超出上限时按插入顺序淘汰最早的会话。"""
        with sessions_lock:
            session = sessions.get(session_id)
            if session is None:
                if len(sessions) >= config.max_sessions:
                    sessions.pop(next(iter(sessions)), None)
                session = SessionMemory(
                    session_id=session_id,
                    budget=ContextBudget(max_tokens=config.session_context_budget),
                )
                sessions[session_id] = session
            return session

    server = MCPServer(
        name=config.name,
        version=config.version,
        instructions=(
            "商品图文检索与工具编排 Agent。检索、比价、库存三个工具均为只读；"
            "单轮端到端追问使用 ask_shopping_agent；"
            "需要多轮上下文（追问库存、追加预算、对比上一轮候选）时使用带 session_id 的 "
            "react_query，它会累积会话状态并返回可审计的多步轨迹。"
        ),
    )

    async def guarded(
        tool: str,
        arguments: dict[str, Any],
        handler: Callable[[], Any],
        *,
        idempotent: bool = True,
    ) -> str:
        """统一出口：成功返回业务载荷，失败返回结构化错误，不把异常直接抛给模型。

        失败被分类为 invalid_arguments / permission_denied / not_found / timeout / tool_failure，
        便于客户端按类型决定重试还是终止。
        """
        try:
            return _dumps(await runtime.run(tool, arguments, handler, idempotent=idempotent))
        except ValidationError as error:
            return _dumps(
                {
                    "ok": False,
                    "error": {
                        "type": "invalid_arguments",
                        "tool": tool,
                        "detail": error.errors(include_url=False, include_input=False),
                    },
                }
            )
        except PermissionError as error:
            return _dumps(
                {"ok": False, "error": {"type": "permission_denied", "tool": tool, "detail": str(error)}}
            )
        except KeyError as error:
            return _dumps(
                {"ok": False, "error": {"type": "not_found", "tool": tool, "detail": str(error)}}
            )
        except TimeoutError:
            return _dumps(
                {
                    "ok": False,
                    "error": {
                        "type": "timeout",
                        "tool": tool,
                        "detail": f"超过 {config.timeout_seconds}s 仍未返回",
                    },
                }
            )
        except Exception as error:  # 兜底：未知异常同样转为结构化错误
            logger.warning("tool %s failed: %s", tool, error)
            return _dumps(
                {"ok": False, "error": {"type": "tool_failure", "tool": tool, "detail": str(error)}}
            )

    @server.tool(
        name="search_products",
        description="按文字或图片检索商品，并应用预算、类目与返回数量约束。只读，可幂等重放。",
    )
    async def search_products(
        query: str = "",
        max_price: float | None = None,
        category: str | None = None,
        top_k: int = 5,
        image_path: str | None = None,
    ) -> str:
        arguments = {
            "query": query,
            "max_price": max_price,
            "category": category,
            "top_k": top_k,
            "image_path": image_path,
        }

        def handler() -> dict[str, Any]:
            if image_path and not allow_image:
                raise PermissionError("服务未开启本地图片路径（设置 MCP_ALLOW_LOCAL_IMAGE_PATH=1）")
            request = SearchRequest.model_validate(arguments)
            hits = tools.execute(ToolCall(name="search_products", arguments=request.model_dump()))
            return {"ok": True, "count": len(hits), "hits": [_search_hit_payload(hit) for hit in hits]}

        return await guarded("search_products", arguments, handler)

    @server.tool(
        name="compare_products",
        description="按商品 ID 对比两个或多个商品的价格、评分与库存。只读，可幂等重放。",
    )
    async def compare_products(product_ids: list[str]) -> str:
        arguments = {"product_ids": list(product_ids)}

        def handler() -> dict[str, Any]:
            products = tools.execute(ToolCall(name="compare_products", arguments=arguments))
            return {
                "ok": True,
                "count": len(products),
                "products": [product.model_dump(exclude={"image_path", "image_sha256"}) for product in products],
            }

        return await guarded("compare_products", arguments, handler)

    @server.tool(
        name="check_inventory",
        description="查询指定商品的库存可用性。只读，可幂等重放。",
    )
    async def check_inventory(product_id: str) -> str:
        arguments = {"product_id": product_id}

        def handler() -> dict[str, Any]:
            inventory = tools.execute(ToolCall(name="check_inventory", arguments=arguments))
            return {"ok": True, **inventory}

        return await guarded("check_inventory", arguments, handler)

    @server.tool(
        name="ask_shopping_agent",
        description="端到端追问：由 Planner 解析意图与约束，调用工具后返回带商品 ID 引用的回答与完整调用轨迹。",
    )
    async def ask_shopping_agent(
        query: str,
        intent: str = "auto",
        max_price: float | None = None,
        category: str | None = None,
        top_k: int = 5,
        product_ids: list[str] | None = None,
    ) -> str:
        arguments = {
            "query": query,
            "intent": intent,
            "max_price": max_price,
            "category": category,
            "top_k": top_k,
            "product_ids": list(product_ids or []),
        }

        def handler() -> dict[str, Any]:
            request = AgentRequest.model_validate(arguments)
            response = agent.run(request)
            return {
                "ok": True,
                "intent": response.intent,
                "answer": response.answer,
                "citations": list(response.citations),
                "tool_trace": list(response.tool_trace),
                "hits": [_search_hit_payload(hit) for hit in response.hits],
            }

        return await guarded("ask_shopping_agent", arguments, handler)

    @server.tool(
        name="react_query",
        description=(
            "多轮会话式 ReAct 执行：同一 session_id 下累积预算、类目与候选商品，"
            "返回可审计的多步轨迹（thought / action / observation）与上下文预算统计。"
            "有状态，因此不参与幂等重放。"
        ),
    )
    async def react_query(query: str, session_id: str = "default", intent: str = "auto") -> str:
        arguments = {"query": query, "session_id": session_id, "intent": intent}

        def handler() -> dict[str, Any]:
            session = resolve_session(session_id)
            react_agent = ReActAgent(
                tools,
                max_steps=config.react_max_steps,
                budget=session.budget,
                session=session,
            )
            result = react_agent.run(query, intent)
            return {
                "ok": True,
                "session_id": session_id,
                "intent": result.intent,
                "answer": result.answer,
                "citations": list(result.citations),
                "stop_reason": result.stop_reason,
                "tool_calls": result.tool_calls,
                "context": result.context,
                "steps": [
                    {
                        "step": step.index,
                        "thought": step.thought,
                        "action": step.action,
                        "arguments": step.arguments,
                        "error": step.error,
                        "latency_ms": step.latency_ms,
                        "observation": _jsonable(step.observation),
                    }
                    for step in result.steps
                ],
            }

        return await guarded("react_query", arguments, handler, idempotent=False)

    @server.tool(
        name="get_session_state",
        description="读取指定会话已累积的槽位（预算 / 类目 / 候选 / 排除项）与当前上下文预算占用。",
    )
    async def get_session_state(session_id: str = "default") -> str:
        arguments = {"session_id": session_id}
        with sessions_lock:
            session = sessions.get(session_id)
            active = len(sessions)
            if session is None:
                return _dumps(
                    {"ok": True, "session_id": session_id, "exists": False, "active_sessions": active}
                )
            payload = {
                "ok": True,
                "session_id": session_id,
                "exists": True,
                "state": session.state.model_dump(),
                "context": session.window().stats(),
                "turns": len(session.turns),
                "active_sessions": active,
            }
        return _dumps(payload)

    @server.tool(
        name="get_runtime_metrics",
        description="返回本服务的运行时指标：逐工具调用量、错误、重试、幂等命中与 P50/P95/P99 延迟。",
    )
    async def get_runtime_metrics() -> str:
        return _dumps(
            {
                "ok": True,
                "config": {
                    "timeout_seconds": config.timeout_seconds,
                    "max_retries": config.max_retries,
                    "max_concurrency": config.max_concurrency,
                    "idempotency_ttl_seconds": config.idempotency_ttl_seconds,
                    "max_replay_entries": config.max_replay_entries,
                    "react_max_steps": config.react_max_steps,
                    "session_context_budget": config.session_context_budget,
                    "max_sessions": config.max_sessions,
                    "allow_local_image_path": allow_image,
                },
                "metrics": runtime.metrics.snapshot(),
                "idempotency_cache": runtime.replay_cache_stats(),
                "active_sessions": len(sessions),
            }
        )

    @server.resource(
        "catalog://products",
        name="product-catalog",
        description="商品目录全量快照（JSON）。",
        mime_type="application/json",
    )
    def product_catalog() -> str:
        return _dumps(
            {
                "count": len(tools.retriever.products),
                "products": [
                    product.model_dump(exclude={"image_path", "image_sha256"})
                    for product in tools.retriever.products
                ],
            }
        )

    @server.resource(
        "catalog://products/{product_id}",
        name="product-detail",
        description="按商品 ID 读取单个商品的结构化事实。",
        mime_type="application/json",
    )
    def product_detail(product_id: str) -> str:
        product = tools.by_id.get(product_id)
        if product is None:
            return _dumps({"ok": False, "error": {"type": "not_found", "product_id": product_id}})
        return _dumps({"ok": True, "product": product.model_dump(exclude={"image_path", "image_sha256"})})

    @server.resource(
        "catalog://schema",
        name="catalog-schema",
        description="商品数据契约（Product 的 JSON Schema），供客户端校验与生成表单。",
        mime_type="application/json",
    )
    def catalog_schema() -> str:
        return _dumps(Product.model_json_schema())

    @server.prompt(
        name="grounded_recommendation",
        description="生成一条受预算与类目约束、必须引用商品 ID 的推荐提示词。",
    )
    def grounded_recommendation(query: str, budget: str = "未指定", tone: str = "简洁") -> str:
        return (
            f"你是商品导购助手。用户需求：{query}；预算：{budget}。\n"
            f"请先调用 search_products 检索，再对候选调用 check_inventory 确认库存。\n"
            f"价格与库存只能来自工具返回结果，不得凭记忆生成；每条结论必须附商品 ID 引用。\n"
            f"回答风格：{tone}。"
        )

    @server.prompt(
        name="tool_orchestration_plan",
        description="针对一个目标产出工具编排计划，说明调用顺序、依赖与失败回退路径。",
    )
    def tool_orchestration_plan(goal: str) -> str:
        return (
            f"目标：{goal}\n"
            "请输出编排计划，包含：\n"
            "1) 需要调用的工具及顺序；2) 每个工具的入参来源；3) 失败时的回退路径；"
            "4) 终止条件与最大步数。可用工具：search_products、compare_products、"
            "check_inventory、ask_shopping_agent、react_query。"
        )

    server.runtime = runtime  # type: ignore[attr-defined]  # 便于测试与压测读取指标
    return server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the multimodal shopping agent as an MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse", "streamable-http"), default="stdio")
    parser.add_argument("--catalog", default=os.getenv("CATALOG_PATH", "data/products.jsonl"))
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8770")))
    parser.add_argument("--retriever-backend", choices=("baseline", "clip", "fusion"), default=os.getenv("RETRIEVER_BACKEND", "baseline"))
    parser.add_argument("--planner-backend", choices=("rule", "llm"), default=os.getenv("PLANNER_BACKEND", "rule"))
    parser.add_argument("--index-dir", default=os.getenv("INDEX_DIR"))
    parser.add_argument("--model", default=os.getenv("CLIP_MODEL_PATH", "OFA-Sys/chinese-clip-vit-base-patch16"))
    parser.add_argument("--device", default=os.getenv("MODEL_DEVICE", "cpu"))
    parser.add_argument("--lexical-weight", type=float, default=float(os.getenv("LEXICAL_WEIGHT", "0.65")))
    parser.add_argument("--tool-timeout", type=float, default=float(os.getenv("MCP_TOOL_TIMEOUT", str(DEFAULT_TOOL_TIMEOUT_SECONDS))))
    parser.add_argument("--max-retries", type=int, default=int(os.getenv("MCP_MAX_RETRIES", "1")))
    parser.add_argument("--max-concurrency", type=int, default=int(os.getenv("MCP_MAX_CONCURRENCY", str(DEFAULT_MAX_CONCURRENCY))))
    parser.add_argument("--idempotency-ttl", type=float, default=float(os.getenv("MCP_IDEMPOTENCY_TTL", str(DEFAULT_IDEMPOTENCY_TTL_SECONDS))))
    parser.add_argument("--max-replay-entries", type=int, default=int(os.getenv("MCP_MAX_REPLAY_ENTRIES", str(DEFAULT_MAX_REPLAY_ENTRIES))))
    parser.add_argument("--react-max-steps", type=int, default=int(os.getenv("MCP_REACT_MAX_STEPS", "6")))
    parser.add_argument("--session-context-budget", type=int, default=int(os.getenv("MCP_SESSION_CONTEXT_BUDGET", "2048")))
    parser.add_argument("--max-sessions", type=int, default=int(os.getenv("MCP_MAX_SESSIONS", "128")))
    parser.add_argument("--allow-local-image-path", action="store_true", default=os.getenv("MCP_ALLOW_LOCAL_IMAGE_PATH", "0") == "1")
    parser.add_argument("--log-level", default=os.getenv("MCP_LOG_LEVEL", "INFO"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    tools = load_tools(
        args.catalog,
        args.retriever_backend,
        args.index_dir,
        model=args.model,
        device=args.device,
        lexical_weight=args.lexical_weight,
    )
    categories = [product.category for product in tools.retriever.products]
    if args.planner_backend == "llm":
        from .llm_planner import LLMPlanner

        endpoint, model = os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL")
        if not endpoint or not model:
            raise SystemExit("--planner-backend llm 需要设置 LLM_BASE_URL 与 LLM_MODEL")
        planner: Any = LLMPlanner(categories, endpoint, model, os.getenv("LLM_API_KEY", ""))
    else:
        planner = RulePlanner(categories)

    config = ServerConfig(
        timeout_seconds=args.tool_timeout,
        max_retries=args.max_retries,
        max_concurrency=args.max_concurrency,
        idempotency_ttl_seconds=args.idempotency_ttl,
        max_replay_entries=args.max_replay_entries,
        react_max_steps=args.react_max_steps,
        session_context_budget=args.session_context_budget,
        max_sessions=args.max_sessions,
    )
    server = build_server(tools, planner, config, allow_local_image_path=args.allow_local_image_path)

    logger.info(
        "MCP server 启动：transport=%s catalog=%s tools=%d",
        args.transport,
        args.catalog,
        len(tools.retriever.products),
    )
    if args.transport == "stdio":
        server.run("stdio")
    elif args.transport == "sse":
        server.run("sse", host=args.host, port=args.port)
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
