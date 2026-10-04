"""会话状态外置：让服务层真正可以无状态水平扩展。

背景
----
``SessionMemory`` 原本活在进程内存里。单进程单实例时这没问题，但一旦为了扛并发而
起多个副本，立刻碎掉：用户在副本 A 说的话，下一个请求被负载均衡丢到副本 B，
B 的内存里没有这个会话，于是"预算上限 ¥500"这种已确认约束凭空消失。
表现是"多轮记忆时好时坏"，而根因是状态位置错了。

这一层把"会话存在哪"抽成 ``SessionStore``，只做三件事：读、写、过期。

    InMemorySessionStore —— 单进程/测试用，语义与 Redis 版一致
    RedisSessionStore    —— 多副本共享；带 TTL 与版本化 CAS

为什么写操作要做版本化（CAS）
------------------------------
两个副本同时读到一个会话，各自处理一轮请求，然后先后写回——后写的那次会把先写的那次
**整段覆盖掉**，用户看到某轮对话凭空消失。所以写入必须带版本校验：

    只有"我读到的版本 == 当前存储里的版本"时才允许写；版本对不上就重读、重放、重写。

本模块用 Lua 脚本在 Redis 端原子完成这次比较与写入（``redis.call`` 保证原子性），
而不是"GET 出来比一下再 SET"——后者中间存在竞态窗口，等于没做。

关于并发写的诚实说明
--------------------
CAS 只能保证"不会静默覆盖"，不能凭空解决语义冲突。同一个 ``session_id`` 被两个客户端
同时驱动，本身就是客户端异常。本层的处理是：重试写入、采用最新版本、并**累计冲突计数**
（``SessionHandle.conflicts``）。计数存在的意义是让这种异常可见——如果它会持续发生，
正确的修法是在网关层按 ``session_id`` 做一致性路由，而不是把重试次数调大。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator, Protocol

from .session import ContextBudget, SessionMemory, SessionSnapshot

REDIS_MISSING_HINT = '未安装 redis 客户端，请执行 pip install -e ".[redis]"（或 pip install redis）'
DEFAULT_TTL_SECONDS = 3600
DEFAULT_NAMESPACE = "shopping-agent:session"

# 原子 CAS：版本一致才写入，同时续期。返回 1 表示写入成功，0 表示版本冲突。
CAS_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if not current then
  if ARGV[1] ~= '0' then
    return 0
  end
else
  local ok, decoded = pcall(cjson.decode, current)
  if not ok then
    return 0
  end
  if tostring(decoded['version']) ~= ARGV[1] then
    return 0
  end
end
redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
return 1
"""


class SessionConflict(RuntimeError):
    """版本冲突且重试用尽。调用方应重新读取会话而不是继续使用陈旧状态。"""


class SessionStore(Protocol):
    """会话存储契约。

    ``save`` 必须在版本不匹配时返回 ``False`` 而**不是抛异常**——冲突是预期内的情况，
    需要由上层决定重试策略。
    """

    def load(
        self,
        session_id: str,
        budget: ContextBudget | None = None,
        system_prompt: str = "你是商品导购助手，价格与库存只能来自工具返回结果。",
    ) -> SessionMemory: ...

    def save(self, memory: SessionMemory) -> bool: ...

    def delete(self, session_id: str) -> None: ...

    def ping(self) -> bool: ...


class InMemorySessionStore:
    """进程内实现。用于测试与单实例部署，语义与 Redis 版严格对齐（含版本冲突）。"""

    def __init__(self) -> None:
        self._snapshots: dict[str, SessionSnapshot] = {}

    def load(
        self,
        session_id: str,
        budget: ContextBudget | None = None,
        system_prompt: str = "你是商品导购助手，价格与库存只能来自工具返回结果。",
    ) -> SessionMemory:
        snapshot = self._snapshots.get(session_id)
        if snapshot is None:
            return SessionMemory(session_id=session_id, budget=budget, system_prompt=system_prompt)
        return SessionMemory.from_snapshot(snapshot, budget=budget, system_prompt=system_prompt)

    def save(self, memory: SessionMemory) -> bool:
        current = self._snapshots.get(memory.session_id)
        current_version = current.version if current else 0
        if current_version != memory.version:
            return False
        snapshot = memory.to_snapshot()
        snapshot.version = memory.version + 1
        self._snapshots[memory.session_id] = snapshot
        memory.version = snapshot.version
        return True

    def delete(self, session_id: str) -> None:
        self._snapshots.pop(session_id, None)

    def ping(self) -> bool:
        return True

    def __len__(self) -> int:
        return len(self._snapshots)

    def versions(self) -> dict[str, int]:
        return {session_id: snapshot.version for session_id, snapshot in self._snapshots.items()}


class RedisSessionStore:
    """Redis 实现：多副本共享会话，带 TTL 与原子 CAS。"""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        namespace: str = DEFAULT_NAMESPACE,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        client=None,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds 必须大于 0")
        self.url = url
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        if client is not None:
            self.client = client
        else:
            try:
                import redis  # noqa: PLC0415
            except ImportError as error:  # pragma: no cover - 取决于环境
                raise RuntimeError(REDIS_MISSING_HINT) from error
            self.client = redis.Redis.from_url(url, decode_responses=True)
        # 预加载脚本：避免每次请求都传一遍脚本体，也省掉 EVALSHA 的首轮回退。
        self._script = self.client.register_script(CAS_SCRIPT)

    def key(self, session_id: str) -> str:
        return f"{self.namespace}:{session_id}"

    def load(
        self,
        session_id: str,
        budget: ContextBudget | None = None,
        system_prompt: str = "你是商品导购助手，价格与库存只能来自工具返回结果。",
    ) -> SessionMemory:
        payload = self.client.get(self.key(session_id))
        if payload is None:
            return SessionMemory(session_id=session_id, budget=budget, system_prompt=system_prompt)
        snapshot = SessionSnapshot.model_validate_json(payload)
        return SessionMemory.from_snapshot(snapshot, budget=budget, system_prompt=system_prompt)

    def save(self, memory: SessionMemory) -> bool:
        snapshot = memory.to_snapshot()
        snapshot.version = memory.version + 1
        written = self._script(
            keys=[self.key(memory.session_id)],
            args=[str(memory.version), snapshot.model_dump_json(), self.ttl_seconds],
        )
        if not written:
            return False
        memory.version = snapshot.version
        return True

    def delete(self, session_id: str) -> None:
        self.client.delete(self.key(session_id))

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001 - 就绪探针不应因后端异常而抛穿
            return False

    def ttl(self, session_id: str) -> int:
        return int(self.client.ttl(self.key(session_id)))


class SessionHandle:
    """会话作用域：进入时加载、退出时用 CAS 写回，冲突时重读重试。

    用法：

        with open_session(store, "s-1", budget) as memory:
            memory.begin_turn("预算 500 以内")
            memory.record_answer("...")
        # 退出时自动写回
    """

    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        budget: ContextBudget | None = None,
        system_prompt: str = "你是商品导购助手，价格与库存只能来自工具返回结果。",
        retries: int = 3,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.budget = budget
        self.system_prompt = system_prompt
        self.retries = max(1, retries)
        self.memory: SessionMemory | None = None
        self.conflicts = 0

    def __enter__(self) -> SessionMemory:
        self.memory = self.store.load(self.session_id, budget=self.budget, system_prompt=self.system_prompt)
        return self.memory

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is not None or self.memory is None:
            # 本轮失败就不落盘：宁可丢掉一轮，也不要把半截状态写进会话。
            return False
        for _ in range(self.retries):
            if self.store.save(self.memory):
                return False
            self.conflicts += 1
            # 版本对不上：读回最新快照，用我们的状态覆盖，但采用存储端的版本号。
            fresh = self.store.load(self.session_id, budget=self.budget, system_prompt=self.system_prompt)
            latest_version = fresh.version
            self.memory.version = latest_version
        raise SessionConflict(
            f"会话 {self.session_id} 连续 {self.retries} 次写入均版本冲突；"
            "请检查是否有多个副本在并发驱动同一个 session_id"
        )


@contextmanager
def open_session(
    store: SessionStore,
    session_id: str,
    budget: ContextBudget | None = None,
    system_prompt: str = "你是商品导购助手，价格与库存只能来自工具返回结果。",
    retries: int = 3,
) -> Iterator[SessionMemory]:
    handle = SessionHandle(store, session_id, budget, system_prompt, retries)
    with handle:
        assert handle.memory is not None
        yield handle.memory


def create_session_store(
    backend: str | None = None,
    url: str | None = None,
    namespace: str = DEFAULT_NAMESPACE,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> SessionStore:
    """按配置构造会话存储。默认内存实现，保证纯 CPU 环境开箱可跑。"""
    resolved = (backend or os.getenv("SESSION_BACKEND", "memory")).lower()
    if resolved == "memory":
        return InMemorySessionStore()
    if resolved == "redis":
        resolved_url = url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        return RedisSessionStore(
            resolved_url,
            namespace=namespace,
            ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", ttl_seconds)),
        )
    raise ValueError(f"不支持的 SESSION_BACKEND: {resolved}（可选 memory / redis）")


def redis_available() -> bool:
    try:
        import redis  # noqa: F401,PLC0415

        return True
    except ImportError:
        return False


__all__ = [
    "DEFAULT_NAMESPACE",
    "DEFAULT_TTL_SECONDS",
    "InMemorySessionStore",
    "RedisSessionStore",
    "SessionConflict",
    "SessionHandle",
    "SessionStore",
    "create_session_store",
    "open_session",
    "redis_available",
]
