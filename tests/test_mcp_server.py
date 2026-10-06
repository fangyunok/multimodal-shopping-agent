"""Tests for the MCP server: protocol surface plus the resilience and metrics layer.

No external process is required — the suite drives the server through mcp's in-memory
transport (``Client(server)``), so it covers tool discovery, resource reads, prompt
rendering, argument validation, idempotent replay and metric collection in one pass.
"""

import asyncio
import json

import pytest

pytest.importorskip("mcp", reason="MCP server tests need mcp>=2.3")

from mcp import Client

from shopping_agent.mcp_server import ServerConfig, ToolRuntime, build_server, percentile
from shopping_agent.models import Product
from shopping_agent.planner import RulePlanner
from shopping_agent.retrieval import HybridRetriever
from shopping_agent.tools import ShoppingTools

EXPECTED_TOOLS = {
    "search_products",
    "compare_products",
    "check_inventory",
    "ask_shopping_agent",
    "react_query",
    "get_session_state",
    "get_runtime_metrics",
}


def make_tools(tmp_path) -> ShoppingTools:
    products = [
        Product(id="shoe-001", title="云步白色通勤运动鞋", category="运动鞋", description="轻量透气", price=299, rating=4.5, stock=5),
        Product(id="shoe-002", title="疾风越野跑鞋", category="运动鞋", description="防滑耐磨", price=459, rating=4.7, stock=0),
        Product(id="bag-001", title="通勤双肩包", category="箱包", description="大容量", price=199, rating=4.2, stock=8),
    ]
    return ShoppingTools(HybridRetriever(products, tmp_path / "products.jsonl"))


def make_server(tmp_path, **config_kwargs):
    tools = make_tools(tmp_path)
    planner = RulePlanner([product.category for product in tools.retriever.products])
    return build_server(tools, planner, ServerConfig(**config_kwargs))


def call(server, tool: str, arguments: dict) -> dict:
    async def scenario() -> dict:
        async with Client(server) as client:
            result = await client.call_tool(tool, arguments)
            assert result.is_error is False
            return json.loads(result.content[0].text)

    return asyncio.run(scenario())


def test_protocol_surface_exposes_tools_resources_and_prompts(tmp_path) -> None:
    server = make_server(tmp_path)

    async def scenario():
        async with Client(server) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            resources = {str(item.uri) for item in (await client.list_resources()).resources}
            templates = {item.uri_template for item in (await client.list_resource_templates()).resource_templates}
            prompts = {prompt.name for prompt in (await client.list_prompts()).prompts}
            return tools, resources, templates, prompts

    tools, resources, templates, prompts = asyncio.run(scenario())
    assert tools == EXPECTED_TOOLS
    assert resources == {"catalog://products", "catalog://schema"}
    assert templates == {"catalog://products/{product_id}"}
    assert prompts == {"grounded_recommendation", "tool_orchestration_plan"}


def test_tool_input_schemas_are_function_calling_ready(tmp_path) -> None:
    server = make_server(tmp_path)

    async def scenario():
        async with Client(server) as client:
            return {tool.name: tool.input_schema for tool in (await client.list_tools()).tools}

    schemas = asyncio.run(scenario())
    for name, schema in schemas.items():
        assert schema["type"] == "object", name
        if name != "get_runtime_metrics":
            assert schema.get("properties"), name
    assert "query" in schemas["search_products"]["properties"]
    assert schemas["search_products"]["properties"]["top_k"]["type"] == "integer"


def test_search_tool_returns_structured_hits(tmp_path) -> None:
    payload = call(make_server(tmp_path), "search_products", {"query": "运动鞋", "top_k": 3})
    assert payload["ok"] is True
    assert 1 <= payload["count"] <= 3
    hit = payload["hits"][0]
    assert set(hit) >= {"product_id", "title", "price", "stock", "score", "reasons"}
    # 未指定 category 时按文本相关性排序，最相关的结果应为运动鞋
    assert hit["category"] == "运动鞋"
    assert hit["reasons"] == ["商品文本与需求匹配"]


def test_category_filter_narrows_results(tmp_path) -> None:
    payload = call(make_server(tmp_path), "search_products", {"query": "运动", "category": "运动鞋", "top_k": 5})
    assert payload["ok"] is True
    assert payload["count"] >= 1
    assert all(item["category"] == "运动鞋" for item in payload["hits"])


def test_budget_constraint_is_enforced_by_the_tool(tmp_path) -> None:
    payload = call(make_server(tmp_path), "search_products", {"query": "", "max_price": 250, "top_k": 5})
    assert payload["ok"] is True
    assert all(item["price"] <= 250 for item in payload["hits"])


def test_out_of_range_argument_returns_structured_error(tmp_path) -> None:
    payload = call(make_server(tmp_path), "search_products", {"query": "运动鞋", "top_k": 999})
    assert payload["ok"] is False
    assert payload["error"]["type"] == "invalid_arguments"
    assert payload["error"]["tool"] == "search_products"


def test_compare_requires_at_least_two_product_ids(tmp_path) -> None:
    payload = call(make_server(tmp_path), "compare_products", {"product_ids": ["shoe-001"]})
    assert payload["ok"] is False
    assert payload["error"]["type"] == "invalid_arguments"


def test_inventory_reports_not_found_for_unknown_product(tmp_path) -> None:
    payload = call(make_server(tmp_path), "check_inventory", {"product_id": "not-exist"})
    assert payload["ok"] is False
    assert payload["error"]["type"] == "not_found"


def test_local_image_path_is_denied_by_default(tmp_path) -> None:
    payload = call(make_server(tmp_path), "search_products", {"query": "鞋", "image_path": "C:/secret/x.png"})
    assert payload["ok"] is False
    assert payload["error"]["type"] == "permission_denied"


def test_local_image_path_is_allowed_only_when_enabled(tmp_path) -> None:
    tools = make_tools(tmp_path)
    server = build_server(tools, None, ServerConfig(), allow_local_image_path=True)
    payload = call(server, "search_products", {"query": "鞋", "image_path": "C:/not/a/real/image.png"})
    # 放行后失败原因应变为文件不存在，而不是权限拒绝
    assert payload["error"]["type"] != "permission_denied"


def test_agent_tool_returns_auditable_trace_and_citations(tmp_path) -> None:
    payload = call(make_server(tmp_path), "ask_shopping_agent", {"query": "300元以内的运动鞋"})
    assert payload["ok"] is True
    assert payload["intent"] == "search"
    assert payload["citations"]
    assert [step["tool"] for step in payload["tool_trace"]][0] == "search_products"


def test_repeated_arguments_are_served_from_the_idempotency_cache(tmp_path) -> None:
    server = make_server(tmp_path)
    first = call(server, "search_products", {"query": "运动鞋", "top_k": 2})
    second = call(server, "search_products", {"query": "运动鞋", "top_k": 2})
    assert first == second

    metrics = call(server, "get_runtime_metrics", {})["metrics"]["tools"]["search_products"]
    assert metrics["calls"] == 2
    assert metrics["replayed"] == 1


def test_metrics_report_latency_percentiles(tmp_path) -> None:
    server = make_server(tmp_path)
    call(server, "search_products", {"query": "运动鞋"})
    call(server, "check_inventory", {"product_id": "shoe-001"})
    payload = call(server, "get_runtime_metrics", {})

    assert payload["ok"] is True
    assert payload["config"]["timeout_seconds"] > 0
    snapshot = payload["metrics"]
    assert snapshot["total_calls"] >= 2
    assert snapshot["tools"]["search_products"]["latency_ms"]["p50"] is not None
    assert snapshot["tools"]["search_products"]["errors"] == 0


def test_product_detail_resource_reads_single_product(tmp_path) -> None:
    server = make_server(tmp_path)

    async def scenario():
        async with Client(server) as client:
            found = await client.read_resource("catalog://products/shoe-001")
            missing = await client.read_resource("catalog://products/absent")
            return json.loads(found.contents[0].text), json.loads(missing.contents[0].text)

    found, missing = asyncio.run(scenario())
    assert found["ok"] is True
    assert found["product"]["id"] == "shoe-001"
    assert missing["ok"] is False
    assert missing["error"]["type"] == "not_found"


def test_recommendation_prompt_demands_product_id_citations(tmp_path) -> None:
    server = make_server(tmp_path)

    async def scenario():
        async with Client(server) as client:
            prompt = await client.get_prompt("grounded_recommendation", {"query": "跑鞋", "budget": "500元"})
            return prompt.messages[0].content.text

    text = asyncio.run(scenario())
    assert "跑鞋" in text
    assert "500元" in text
    assert "商品 ID 引用" in text
    assert "不得凭记忆生成" in text


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([], 50) is None
    assert percentile([1.0], 95) == 1.0
    samples = [float(value) for value in range(1, 101)]
    assert percentile(samples, 50) == 50.0
    assert percentile(samples, 95) == 95.0
    assert percentile(samples, 99) == 99.0


def test_react_query_returns_multi_step_trace_and_context_stats(tmp_path) -> None:
    payload = call(make_server(tmp_path), "react_query", {"query": "运动鞋", "session_id": "s1"})
    assert payload["ok"] is True
    assert payload["session_id"] == "s1"
    actions = [step["action"] for step in payload["steps"]]
    assert actions[0] == "search_products"
    assert actions.count("check_inventory") == 2
    assert payload["stop_reason"] == "final_answer"
    assert payload["tool_calls"] == 3
    assert payload["citations"]
    assert payload["context"]["tokens"] <= payload["context"]["max_tokens"]


def test_session_state_accumulates_across_react_calls(tmp_path) -> None:
    server = make_server(tmp_path)
    call(server, "react_query", {"query": "300元以内的运动鞋", "session_id": "s2"})
    state = call(server, "get_session_state", {"session_id": "s2"})
    assert state["exists"] is True
    assert state["state"]["max_price"] == 300
    assert state["state"]["category"] == "运动鞋"
    assert state["state"]["turn_count"] == 1
    assert state["state"]["last_candidates"]
    assert state["active_sessions"] == 1


def test_session_follow_up_reuses_remembered_candidates(tmp_path) -> None:
    server = make_server(tmp_path)
    first = call(server, "react_query", {"query": "运动鞋", "session_id": "s3"})
    second = call(server, "react_query", {"query": "那有货吗", "session_id": "s3"})
    # 追问直接命中会话记住的候选，不再重新检索
    assert second["intent"] == "inventory"
    assert [step["action"] for step in second["steps"]] == ["check_inventory", None]
    assert second["tool_calls"] == 1
    assert first["citations"][0] in second["answer"]


def test_sessions_are_isolated_by_id(tmp_path) -> None:
    server = make_server(tmp_path)
    call(server, "react_query", {"query": "300元以内的运动鞋", "session_id": "a"})
    other = call(server, "get_session_state", {"session_id": "b"})
    assert other["exists"] is False
    assert other["active_sessions"] == 1


def test_stateful_react_query_bypasses_the_idempotency_cache(tmp_path) -> None:
    server = make_server(tmp_path)
    call(server, "react_query", {"query": "运动鞋", "session_id": "cache"})
    call(server, "react_query", {"query": "运动鞋", "session_id": "cache"})
    metrics = call(server, "get_runtime_metrics", {})["metrics"]["tools"]["react_query"]
    # 参数完全相同，但会话状态已经改变：不能走幂等重放返回陈旧快照
    assert metrics["calls"] == 2
    assert metrics["replayed"] == 0


def test_session_eviction_is_bounded_by_max_sessions(tmp_path) -> None:
    server = make_server(tmp_path, max_sessions=2)
    call(server, "react_query", {"query": "运动鞋", "session_id": "one"})
    call(server, "react_query", {"query": "运动鞋", "session_id": "two"})
    call(server, "react_query", {"query": "运动鞋", "session_id": "three"})
    payload = call(server, "get_runtime_metrics", {})
    assert payload["active_sessions"] == 2
    assert payload["config"]["max_sessions"] == 2
    assert call(server, "get_session_state", {"session_id": "one"})["exists"] is False
    assert call(server, "get_session_state", {"session_id": "three"})["exists"] is True


def test_replay_cache_is_bounded_and_reports_evictions(tmp_path) -> None:
    server = make_server(tmp_path, max_replay_entries=2)
    for index in range(5):
        call(server, "search_products", {"query": f"运动鞋-{index}"})

    cache = call(server, "get_runtime_metrics", {})["idempotency_cache"]
    # 每个参数组合都会产生一个新的缓存键；没有上限的话条目数会随长跑服务无限增长
    assert cache["entries"] == 2
    assert cache["capacity"] == 2
    assert cache["evicted"] == 3


def test_evicted_entry_falls_back_to_real_execution(tmp_path) -> None:
    server = make_server(tmp_path, max_replay_entries=1)
    call(server, "search_products", {"query": "运动鞋"})
    call(server, "search_products", {"query": "双肩包"})
    call(server, "search_products", {"query": "运动鞋"})

    metrics = call(server, "get_runtime_metrics", {})["metrics"]["tools"]["search_products"]
    # 第一条已被淘汰，再次调用必须真执行，而不是取到过期结果
    assert metrics["calls"] == 3
    assert metrics["replayed"] == 0


def test_stateful_tool_is_not_retried_on_transient_failure() -> None:
    runtime = ToolRuntime(ServerConfig(max_retries=3, timeout_seconds=5))
    invocations = {"count": 0}

    def flaky() -> str:
        invocations["count"] += 1
        raise ConnectionError("瞬时故障")

    async def scenario() -> None:
        with pytest.raises(ConnectionError):
            await runtime.run("stateful_tool", {}, flaky, idempotent=False)

    asyncio.run(scenario())
    # 超时只代表没等到响应，有状态 handler 可能已经执行完毕；重跑会让副作用翻倍
    assert invocations["count"] == 1


def test_read_only_tool_still_retries_transient_failures() -> None:
    runtime = ToolRuntime(ServerConfig(max_retries=3, timeout_seconds=5, backoff_base_seconds=0.0))
    invocations = {"count": 0}

    def flaky() -> str:
        invocations["count"] += 1
        if invocations["count"] == 1:
            raise ConnectionError("瞬时故障")
        return "ok"

    async def scenario() -> str:
        return await runtime.run("read_only_tool", {}, flaky, idempotent=True)

    assert asyncio.run(scenario()) == "ok"
    assert invocations["count"] == 2
