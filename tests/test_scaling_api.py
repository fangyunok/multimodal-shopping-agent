"""阶段二接口的回归测试：就绪探针与无状态会话。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from shopping_agent import app as app_module
from shopping_agent.session import ContextBudget
from shopping_agent.session_store import InMemorySessionStore, create_session_store


def _reset_caches() -> None:
    """清掉 lru_cache。容错：某些用例会把 get_session_store 临时换成普通函数。

    fixture 的拆卸发生在 monkeypatch 还原之前，此刻属性可能已是无 cache_clear 的
    替身，因此这里必须容忍它不存在，否则 teardown 会盖过真实断言。
    """
    for name in ("get_tools", "get_session_store", "get_planner"):
        target = getattr(app_module, name)
        clear = getattr(target, "cache_clear", None)
        if callable(clear):
            clear()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SESSION_BACKEND", raising=False)
    monkeypatch.delenv("INDEX_ROOT", raising=False)
    _reset_caches()
    yield TestClient(app_module.app)
    _reset_caches()


def test_liveness_does_not_depend_on_dependencies(client: TestClient) -> None:
    """存活探针只回答"进程还在不在"，不检查下游——否则 Redis 挂掉会导致无意义的反复重启。"""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_reports_each_dependency(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["retriever"]["ok"] is True
    assert body["checks"]["retriever"]["products"] > 0
    assert body["checks"]["session_store"]["reachable"] is True


def test_readiness_degrades_when_store_is_unreachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """会话存储不可达时必须降级，而不是让这个副本继续接流量。"""

    class BrokenStore(InMemorySessionStore):
        def ping(self) -> bool:
            return False

    monkeypatch.setattr(app_module, "get_session_store", lambda: BrokenStore())
    body = TestClient(app_module.app).get("/readyz").json()
    assert body["status"] == "degraded"
    assert body["checks"]["session_store"]["reachable"] is False
    assert body["checks"]["retriever"]["ok"] is True


def test_chat_persists_state_across_requests(client: TestClient) -> None:
    """同一个 session_id 的连续请求必须共享状态——这是"无状态服务"的核心承诺。"""
    first = client.post("/chat", json={"session_id": "s-1", "query": "预算 500 以内的通勤运动鞋"})
    assert first.status_code == 200
    assert first.json()["conflicts"] == 0

    second = client.post("/chat", json={"session_id": "s-1", "query": "有货吗"})
    assert second.status_code == 200
    assert second.json()["context"]["kept_turns"] >= first.json()["context"]["kept_turns"]

    store = app_module.get_session_store()
    assert store.load("s-1").version >= 1, "会话必须被写回存储，而不是只活在进程里"
    assert len(store.load("s-1").turns) >= 2


def test_chat_keeps_sessions_isolated(client: TestClient) -> None:
    client.post("/chat", json={"session_id": "alpha", "query": "预算 500"})
    client.post("/chat", json={"session_id": "beta", "query": "预算 900"})
    store = app_module.get_session_store()
    assert store.versions()["alpha"] >= 1
    assert store.versions()["beta"] >= 1


def test_chat_rejects_invalid_payload(client: TestClient) -> None:
    assert client.post("/chat", json={"session_id": "", "query": "x"}).status_code == 422
    assert client.post("/chat", json={"session_id": "s", "query": ""}).status_code == 422
    assert client.post("/chat", json={"session_id": "s", "query": "x", "intent": "hack"}).status_code == 422
    assert client.post("/chat", json={"session_id": "s", "query": "x", "extra": 1}).status_code == 422


def test_index_status_without_version_root(client: TestClient) -> None:
    body = client.get("/index").json()
    assert body["versions"] is None
    assert "INDEX_ROOT" in body["note"]


def test_create_session_store_switches_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESSION_BACKEND", "memory")
    assert isinstance(create_session_store(), InMemorySessionStore)
    assert isinstance(create_session_store("memory", ttl_seconds=60), InMemorySessionStore)


def test_context_budget_env_is_respected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_BUDGET", "512")
    body = client.post("/chat", json={"session_id": "s-budget", "query": "运动鞋"}).json()
    assert body["context"]["max_tokens"] == 512
    assert body["context"]["tokens"] <= 512
    assert ContextBudget(max_tokens=512).usable_tokens == 256
