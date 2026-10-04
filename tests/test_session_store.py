import pytest

from shopping_agent.session import ContextBudget, SessionMemory
from shopping_agent.session_store import (
    InMemorySessionStore,
    RedisSessionStore,
    SessionConflict,
    SessionHandle,
    create_session_store,
    open_session,
    redis_available,
)


def make_memory(session_id: str = "s-1") -> SessionMemory:
    return SessionMemory(session_id=session_id, budget=ContextBudget(max_tokens=512))


def test_snapshot_roundtrip_preserves_slots_and_turns() -> None:
    memory = make_memory()
    memory.begin_turn("预算 500 以内的白色通勤运动鞋")
    memory.record_observation("search_products", [{"product": {"id": "shoe-001"}}])
    memory.record_answer("给你找到了 shoe-001")
    snapshot = memory.to_snapshot()
    restored = SessionMemory.from_snapshot(snapshot)
    assert restored.state.digest() == memory.state.digest()
    assert len(restored.turns) == len(memory.turns)
    assert restored.session_id == memory.session_id


def test_snapshot_keeps_compression_state() -> None:
    """压缩是就地且持久的：快照必须带上摘要，否则恢复后上下文会重新膨胀。

    判据用的是 ``original_tokens``（本轮 window() 开始前的占用）：恢复后的对象若带着
    压缩状态，它的"压缩前占用"应当等于原对象压缩后的占用，而不是最初那 20 轮的体量。
    """
    budget = ContextBudget(max_tokens=128, reserve_tokens=16, keep_recent_turns=2)
    memory = SessionMemory(session_id="s-2", budget=budget)
    for index in range(20):
        memory.begin_turn(f"第 {index} 轮：只看黑色通勤包，预算 300")
        memory.record_answer("已记录。" * 12)
    window = memory.window()
    assert window.dropped_turns > 0
    assert window.tokens < window.original_tokens
    restored = SessionMemory.from_snapshot(memory.to_snapshot(), budget=budget)
    assert restored.window().original_tokens == window.tokens
    assert restored.window().tokens <= window.tokens
    assert restored.state.digest() == memory.state.digest()


def test_snapshot_is_a_deep_copy() -> None:
    """快照必须与原对象解耦，否则并发写入会互相踩。"""
    memory = make_memory()
    memory.begin_turn("预算 500")
    snapshot = memory.to_snapshot()
    snapshot.turns[0].content = "被改掉了"
    assert memory.turns[0].content == "预算 500"


def test_in_memory_store_roundtrip_and_versioning() -> None:
    store = InMemorySessionStore()
    memory = store.load("s-1", budget=ContextBudget(max_tokens=512))
    assert memory.version == 0
    memory.begin_turn("预算 500")
    assert store.save(memory) is True
    assert memory.version == 1

    reloaded = store.load("s-1")
    assert reloaded.state.max_price is None  # 槽位由 Planner 合并，这里只验证事件流与版本
    assert reloaded.version == 1
    assert len(reloaded.turns) == 1


def test_stale_write_is_rejected_instead_of_overwriting() -> None:
    """两个副本同时驱动同一会话时，后写的那次不能静默覆盖先写的。"""
    store = InMemorySessionStore()
    first = store.load("s-1")
    second = store.load("s-1")
    first.begin_turn("第一轮")
    assert store.save(first) is True
    second.begin_turn("第二轮")
    assert store.save(second) is False, "陈旧版本必须被拒绝"
    assert store.versions()["s-1"] == 1


def test_open_session_writes_back_on_exit() -> None:
    store = InMemorySessionStore()
    with open_session(store, "s-9", ContextBudget(max_tokens=1024)) as memory:
        memory.begin_turn("我要买鞋")
    assert store.load("s-9").version == 1
    assert len(store.load("s-9").turns) == 1


def test_open_session_does_not_persist_on_exception() -> None:
    store = InMemorySessionStore()
    with pytest.raises(RuntimeError):
        with open_session(store, "s-9") as memory:
            memory.begin_turn("半途失败")
            raise RuntimeError("boom")
    assert store.load("s-9").version == 0, "失败的一轮不应落盘"


def test_handle_retries_after_conflict_and_counts_it() -> None:
    store = InMemorySessionStore()
    handle = SessionHandle(store, "s-1")
    memory = handle.__enter__()
    memory.begin_turn("我的一轮")
    # 模拟另一个副本抢先写入，把版本推进到 1。
    other = store.load("s-1")
    other.begin_turn("别人的一轮")
    assert store.save(other) is True
    handle.__exit__(None, None, None)
    assert handle.conflicts == 1, "冲突必须被计数，而不是悄悄重试"
    assert store.load("s-1").version == 2


def test_handle_raises_when_retries_are_exhausted() -> None:
    class AlwaysStaleStore(InMemorySessionStore):
        def save(self, memory: SessionMemory) -> bool:  # noqa: D102
            return False

    handle = SessionHandle(AlwaysStaleStore(), "s-1", retries=2)
    handle.__enter__()
    handle.memory.begin_turn("x")  # type: ignore[union-attr]
    with pytest.raises(SessionConflict) as error:
        handle.__exit__(None, None, None)
    assert "版本冲突" in str(error.value)


def test_delete_removes_session() -> None:
    store = InMemorySessionStore()
    with open_session(store, "s-1") as memory:
        memory.begin_turn("x")
    store.delete("s-1")
    assert store.load("s-1").version == 0
    assert len(store) == 0


def test_create_session_store_defaults_to_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESSION_BACKEND", raising=False)
    assert isinstance(create_session_store(), InMemorySessionStore)


def test_create_session_store_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError) as error:
        create_session_store("memcached")
    assert "SESSION_BACKEND" in str(error.value)
    assert "memory" in str(error.value)


class FakeRedis:
    """最小 Redis 桩：只实现 CAS 脚本需要的 GET / SET / DELETE / PING。"""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key: str):
        return self.store.get(key)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)

    def ping(self) -> bool:
        return True

    def ttl(self, key: str) -> int:
        return self.ttls.get(key, -2)

    def register_script(self, script: str):
        def runner(keys, args):
            key = keys[0]
            expected_version, payload, ttl = args[0], args[1], int(args[2])
            current = self.store.get(key)
            if current is None:
                if expected_version != "0":
                    return 0
            else:
                import json

                if str(json.loads(current)["version"]) != expected_version:
                    return 0
            self.store[key] = payload
            self.ttls[key] = ttl
            return 1

        return runner


def test_redis_store_roundtrip_with_stub_client() -> None:
    client = FakeRedis()
    store = RedisSessionStore("redis://stub/0", ttl_seconds=120, client=client)
    memory = store.load("s-1")
    memory.begin_turn("预算 800")
    assert store.save(memory) is True
    assert memory.version == 1
    assert store.ttl("s-1") == 120
    assert store.load("s-1").version == 1
    assert store.ping() is True


def test_redis_store_rejects_stale_version() -> None:
    store = RedisSessionStore("redis://stub/0", client=FakeRedis())
    first = store.load("s-1")
    second = store.load("s-1")
    first.begin_turn("a")
    assert store.save(first) is True
    second.begin_turn("b")
    assert store.save(second) is False


def test_redis_store_requires_client_package(monkeypatch: pytest.MonkeyPatch) -> None:
    if redis_available():
        pytest.skip("本机已安装 redis 客户端，无法验证缺失提示")
    with pytest.raises(RuntimeError) as error:
        RedisSessionStore("redis://localhost:6379/0")
    assert "redis" in str(error.value)


def test_redis_store_validates_ttl() -> None:
    with pytest.raises(ValueError) as error:
        RedisSessionStore("redis://stub/0", ttl_seconds=0, client=FakeRedis())
    assert "ttl_seconds" in str(error.value)
