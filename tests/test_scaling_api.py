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


# ---------- 索引运行期旋钮 ----------


def test_rerank_candidates_env_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    """0 与"没配"必须走同一条路径——否则"配了 0 却仍在重排"只有读两处代码才能发现。"""
    monkeypatch.delenv("ANN_RERANK_CANDIDATES", raising=False)
    assert app_module.ann_runtime_kwargs() == {}
    monkeypatch.setenv("ANN_RERANK_CANDIDATES", "0")
    assert app_module.ann_runtime_kwargs() == {}
    monkeypatch.setenv("ANN_RERANK_CANDIDATES", "500")
    assert app_module.ann_runtime_kwargs() == {"rerank_candidates": 500}


def test_runtime_config_only_exposes_runtime_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """构建期参数（HNSW 的 m、IVF 的 nlist）刻意不暴露：改了它们必须重建索引。

    把它们做成环境变量只会制造"我改了配置为什么没变化"的假问题。
    """
    from shopping_agent.ann_index import AnnIndexConfig

    monkeypatch.setenv("ANN_EF_SEARCH", "128")
    monkeypatch.setenv("ANN_NPROBE", "8")
    monkeypatch.setenv("ANN_M", "64")
    monkeypatch.setenv("ANN_NLIST", "16")
    config = app_module.ann_runtime_config()
    assert (config.ef_search, config.nprobe) == (128, 8)
    defaults = AnnIndexConfig(kind="hnsw", dimension=512)
    assert (config.m, config.nlist) == (defaults.m, defaults.nlist)


def test_partitioned_backend_serves_a_partitioned_index(
    client: TestClient, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """app 层必须能直接服务分段索引，否则"过滤下推"只能停留在压测脚本里。"""
    from shopping_agent.ann_index import AnnIndexConfig
    from shopping_agent.partitioned import PartitionedRetriever
    from test_partitioned import QueryOnlyEncoder, banded_catalog, write_catalog

    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    PartitionedRetriever.from_vectors(
        products,
        catalog,
        QueryOnlyEncoder({"鞋": query}),
        vectors,
        AnnIndexConfig(kind="numpy", dimension=vectors.shape[1]),
        by="price",
        buckets=4,
    ).save(tmp_path / "index", catalog_snapshot=True)

    monkeypatch.setenv("RETRIEVER_BACKEND", "partitioned")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("ANN_KIND", "numpy")
    monkeypatch.setenv("ANN_DIMENSION", str(vectors.shape[1]))
    # 走远端编码器只是为了不加载真实模型：加载索引本身不编码，所以不会真的发请求。
    monkeypatch.setenv("ENCODER_BASE_URL", "http://127.0.0.1:9/")
    _reset_caches()

    tools = app_module.get_tools()
    assert isinstance(tools.retriever, PartitionedRetriever)
    assert tools.retriever.describe()["by"] == "price"

    body = TestClient(app_module.app).get("/readyz").json()
    assert body["checks"]["retriever"]["products"] == len(products)
    assert body["checks"]["index"]["partitions"] == 4
