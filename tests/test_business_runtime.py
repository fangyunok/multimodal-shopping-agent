"""真实本地 HTTP 契约测试；不调用外部模型、不需要业务密钥。"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi.testclient import TestClient

from shopping_agent import app as app_module
from shopping_agent.auth import session_key
from shopping_agent.http_json import JSONServiceClient, UpstreamError
from shopping_agent.inventory import HTTPInventory, InventoryUnavailable
from shopping_agent.llm_reasoner import LLMReasoner
from shopping_agent.react import Decision, ReActAgent


@pytest.fixture
def upstream():
    state = {"calls": [], "stocks": {}, "decide": None, "override": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.respond(None)

        def do_POST(self):
            self.respond(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

        def respond(self, body):
            call = {"path": self.path, "body": body, "authorization": self.headers.get("Authorization")}
            state["calls"].append(call)
            headers = {}
            if state["override"]:
                status, result, headers = state["override"](call)
            elif self.path.startswith("/inventory?"):
                product_id = parse_qs(urlsplit(self.path).query)["product_id"][0]
                status, result = 200, {"product_id": product_id, "stock": state["stocks"].get(product_id, 4),
                                       "observed_at": datetime.now(timezone.utc).isoformat()}
            else:
                assert self.path == "/v1/chat/completions"
                context = json.loads(body["messages"][-1]["content"])
                decision = state["decide"](context)
                content = decision if isinstance(decision, str) else json.dumps(decision)
                status, result = 200, {"choices": [{"message": {"content": content}}]}
            raw = result if isinstance(result, bytes) else json.dumps(result).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()
    server.server_close()
    worker.join(timeout=2)


@pytest.fixture
def api(monkeypatch):
    for name in ("AUTH_MODE", "API_KEYS_JSON", "INVENTORY_BACKEND", "INVENTORY_BASE_URL",
                 "INVENTORY_API_KEY", "REASONER_BACKEND", "LLM_BASE_URL", "LLM_MODEL",
                 "LLM_API_KEY", "SESSION_BACKEND", "PLANNER_BACKEND", "CONTEXT_BUDGET"):
        monkeypatch.delenv(name, raising=False)
    clears = [app_module.get_tools.cache_clear, app_module.get_session_store.cache_clear,
              app_module.get_planner.cache_clear]
    for clear in clears:
        clear()
    yield TestClient(app_module.app)
    for clear in clears:
        clear()


def configure_business(monkeypatch, url):
    monkeypatch.setenv("INVENTORY_BACKEND", "http")
    monkeypatch.setenv("INVENTORY_BASE_URL", url)
    monkeypatch.setenv("INVENTORY_API_KEY", "inventory-test-key")
    monkeypatch.setenv("REASONER_BACKEND", "llm")
    monkeypatch.setenv("LLM_BASE_URL", url + "/v1")
    monkeypatch.setenv("LLM_MODEL", "contract-test-model")
    monkeypatch.setenv("LLM_API_KEY", "model-test-key")


def shopping_decision(context):
    observations = context["observations"]
    if context["intent"] == "inventory":
        if observations:
            return {"action": "finish"}
        return {"action": "check_inventory", "arguments": {"product_id": context["state"]["last_candidates"][0]}}
    if not observations:
        # 刻意放宽模型参数：服务端必须保住用户已确认的约束。
        return {"action": "search_products", "arguments": {"query": "运动鞋", "max_price": 99999, "top_k": 20}}
    candidates = observations[0]["result"]
    checked = {item["arguments"].get("product_id") for item in observations if item["tool"] == "check_inventory"}
    for item in candidates:
        if item["product_id"] not in checked:
            return {"action": "check_inventory", "arguments": {"product_id": item["product_id"]}}
    available = [item["result"]["product_id"] for item in observations if item["tool"] == "check_inventory"
                 and not item["error"] and item["result"]["available"]]
    return {"action": "finish", "selected_product_ids": available[:1]}


def test_http_inventory_changes_without_reloading_catalog(api, upstream, monkeypatch):
    url, state = upstream
    monkeypatch.setenv("INVENTORY_BACKEND", "http")
    monkeypatch.setenv("INVENTORY_BASE_URL", url)
    state["stocks"]["shoe-001"] = 0
    request = {"intent": "inventory", "product_ids": ["shoe-001"]}
    assert "无货" in api.post("/agent", json=request).json()["answer"]
    state["stocks"]["shoe-001"] = 2
    result = api.post("/agent", json=request).json()
    assert "剩余 2 件" in result["answer"]
    assert result["inventory_source"] == "http"
    assert app_module.get_tools().by_id["shoe-001"].stock == 23
    hits = api.post("/agent", json={"query": "白色运动鞋", "max_price": 500}).json()["hits"]
    assert next(hit for hit in hits if hit["product"]["id"] == "shoe-001")["product"]["stock"] == 2
    state["stocks"].update({"shoe-001": 0, "shoe-002": 0})
    sold_out = api.post("/agent", json={"query": "运动鞋", "max_price": 500}).json()
    assert "没有库存" in sold_out["answer"]
    assert all(hit["product"]["stock"] == 0 for hit in sold_out["hits"])


@pytest.mark.parametrize("problem", ["negative", "bool", "string", "missing_time", "old", "future", "wrong_id",
                                     "naive_time", "server_error", "redirect", "large", "invalid_json"])
def test_inventory_failures_never_become_in_stock_or_zero(upstream, problem):
    url, state = upstream
    now = datetime.now(timezone.utc)
    record = {"product_id": "shoe-001", "stock": 2, "observed_at": now.isoformat()}
    status, headers = 200, {}
    if problem in {"negative", "bool", "string"}:
        record["stock"] = {"negative": -1, "bool": True, "string": "2"}[problem]
    elif problem == "missing_time":
        record.pop("observed_at")
    elif problem in {"old", "future"}:
        record["observed_at"] = (now + timedelta(seconds=-60 if problem == "old" else 60)).isoformat()
    elif problem == "naive_time":
        record["observed_at"] = now.replace(tzinfo=None).isoformat()
    elif problem == "wrong_id":
        record["product_id"] = "shoe-002"
    elif problem == "server_error":
        status = 500
    elif problem == "redirect":
        status, headers = 302, {"Location": url + "/must-not-follow"}
    elif problem == "large":
        record = b"x" * 9000
    else:
        record = b"not-json"
    state["override"] = lambda call: (status, record, headers)
    with pytest.raises(InventoryUnavailable):
        HTTPInventory(url).check("shoe-001")
    assert len(state["calls"]) == 1


def test_inventory_timeout_is_bounded(upstream):
    url, state = upstream

    def delayed(call):
        time.sleep(0.1)
        return 200, {}, {}

    state["override"] = delayed
    with pytest.raises(InventoryUnavailable):
        HTTPInventory(url, timeout_seconds=0.02).check("shoe-001")


def test_inventory_outage_is_unknown_and_sanitized(api, upstream, monkeypatch):
    url, state = upstream
    monkeypatch.setenv("INVENTORY_BACKEND", "http")
    monkeypatch.setenv("INVENTORY_BASE_URL", url)
    state["override"] = lambda call: (500, {"secret": "upstream-private-data"}, {})
    result = api.post("/chat", json={"session_id": "outage", "query": "500元以内的运动鞋"})
    assert result.status_code == 200
    assert "无法确认" in result.json()["answer"]
    assert "无货" not in result.json()["answer"]
    assert "upstream-private-data" not in result.text
    assert api.post("/agent", json={"query": "运动鞋"}).status_code == 503


def test_http_llm_loop_receives_observations_and_preserves_constraints(api, upstream, monkeypatch):
    url, state = upstream
    configure_business(monkeypatch, url)
    state["decide"] = shopping_decision
    first = api.post("/chat", json={"session_id": "llm", "query": "500元以内的运动鞋，不要 shoe-001"})
    assert first.status_code == 200, first.text
    result = first.json()
    assert result["context"]["reasoner_backend"] == "llm"
    assert result["context"]["fallback_used"] is False
    assert result["context"]["inventory_source"] == "http"
    assert result["citations"] == ["shoe-002"]
    assert "shoe-001" not in result["answer"] and "shoe-003" not in result["answer"]
    model_calls = [call for call in state["calls"] if call["body"] is not None]
    assert len(model_calls) == 3
    second_context = json.loads(model_calls[1]["body"]["messages"][-1]["content"])
    args = second_context["observations"][0]["arguments"]
    assert args["max_price"] == 500 and args["top_k"] == 3 and args["excluded_product_ids"] == ["shoe-001"]
    assert all(call["authorization"] == "Bearer model-test-key" for call in model_calls)
    assert all(call["authorization"] == "Bearer inventory-test-key" for call in state["calls"] if call["body"] is None)
    state["stocks"]["shoe-002"] = 0
    follow_up = api.post("/chat", json={"session_id": "llm", "query": "有货吗"}).json()
    assert follow_up["tool_calls"] == 1 and "无货" in follow_up["answer"]
    assert follow_up["citations"] == ["shoe-002"]


@pytest.mark.parametrize("decision", ["invalid JSON", {"action": "delete_products"},
                                     {"action": "finish", "selected_product_ids": ["invented-999"]}])
def test_bad_model_output_falls_back_once_without_fabricating_facts(api, upstream, monkeypatch, decision):
    url, state = upstream
    configure_business(monkeypatch, url)
    state["decide"] = lambda context: decision
    result = api.post("/chat", json={"session_id": "bad-model", "query": "500元以内运动鞋"}).json()
    assert result["context"]["fallback_used"] is True
    assert result["stop_reason"] == "final_answer"
    assert "invented-999" not in result["answer"]
    assert len([call for call in state["calls"] if call["body"]]) == 1


def test_model_cannot_finish_with_unverified_or_out_of_stock_product(api, upstream, monkeypatch):
    url, state = upstream
    configure_business(monkeypatch, url)
    state["stocks"].update({"shoe-001": 0, "shoe-002": 0})
    state["decide"] = lambda ctx: shopping_decision(ctx) if not ctx["observations"] else {
        "action": "finish", "selected_product_ids": ["shoe-001"],
    }
    result = api.post("/chat", json={"session_id": "unchecked", "query": "500元以内运动鞋"}).json()
    assert result["context"]["fallback_used"] is True
    assert "没有库存" in result["answer"] and "优先考虑" not in result["answer"]


@pytest.mark.parametrize("arguments", [{"image_path": "/etc/passwd"}, {"product_id": "shoe-003"},
                                       {"product_ids": ["shoe-001", "shoe-002"]}])
def test_tool_executor_enforces_file_and_product_boundaries(api, arguments):
    class BadReasoner:
        def decide(self, context):
            action = ("search_products" if "image_path" in arguments else
                      "check_inventory" if "product_id" in arguments else "compare_products")
            return Decision(thought="untrusted", action=action, arguments=arguments)

    agent = ReActAgent(app_module.get_tools(), reasoner=BadReasoner())
    result = agent.run("500元以内运动鞋，不要 shoe-001")
    assert result.steps[0].error == "constraint_violation"
    assert result.steps[0].observation is None
    assert result.stop_reason == "loop_detected"


def test_model_payload_budget_prevents_network_request(api):
    calls = []
    reasoner = LLMReasoner("http://127.0.0.1:9", "model", max_input_tokens=256,
                           transport=lambda payload: calls.append(payload))
    result = ReActAgent(app_module.get_tools(), reasoner=reasoner).run("500元以内运动鞋")
    assert result.context["fallback_used"] is True
    assert calls == []


@pytest.mark.parametrize("path", ["/chat", "/agent", "/agent/image", "/search", "/tools", "/index",
                                  "/docs", "/openapi.json", "/health"])
def test_business_routes_require_authentication(api, monkeypatch, path):
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv("API_KEYS_JSON", json.dumps({"a": "a" * 32}))
    response = api.get(path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert api.get("/healthz").status_code == 200
    assert api.get("/readyz").status_code == 200


@pytest.mark.parametrize("config", ["", "{}", "not-json", '{"a":"short"}',
                                   json.dumps({"a": "x" * 32, "b": "x" * 32})])
def test_auth_misconfiguration_fails_closed(api, monkeypatch, config):
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv("API_KEYS_JSON", config)
    assert api.get("/tools").status_code == 503
    assert api.get("/readyz").status_code == 503
    assert api.get("/healthz").status_code == 200


def test_principal_isolation_and_key_rotation_keep_session_ownership(api, monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv("API_KEYS_JSON", json.dumps({"a": "a" * 32, "b": "b" * 32}))
    headers_a, headers_b = {"Authorization": "Bearer " + "a" * 32}, {"Authorization": "Bearer " + "b" * 32}
    first = api.post("/chat", json={"session_id": "same", "query": "500元以内运动鞋"}, headers=headers_a)
    assert first.status_code == 200
    isolated = api.post("/chat", json={"session_id": "same", "query": "有货吗"}, headers=headers_b).json()
    assert isolated["tool_calls"] == 0 and "商品 ID" in isolated["answer"]
    monkeypatch.setenv("API_KEYS_JSON", json.dumps({"a": "new-a" * 8, "b": "b" * 32}))
    assert api.get("/tools", headers=headers_a).status_code == 401
    rotated = api.post("/chat", json={"session_id": "same", "query": "有货吗"},
                       headers={"Authorization": "Bearer " + "new-a" * 8}).json()
    assert rotated["tool_calls"] == 1
    store = app_module.get_session_store()
    assert store.load(session_key("a", "same")).state.max_price == 500
    assert store.load(session_key("b", "same")).state.max_price is None


def test_redirect_never_forwards_upstream_credentials(upstream):
    url, state = upstream
    state["override"] = lambda call: (302, {}, {"Location": url + "/other"})
    with pytest.raises(UpstreamError):
        JSONServiceClient(url, api_key="sensitive").request("/start")
    assert [call["path"] for call in state["calls"]] == ["/start"]
