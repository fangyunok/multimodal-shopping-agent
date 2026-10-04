"""编码层：把"把图文变成向量"从离线脚本升级成可扩展的 GPU 服务。

为什么编码层会成为规模化的瓶颈
------------------------------
检索和索引都是毫秒级的事，而编码是**模型前向**。十万条商品文本要跑一次
Chinese-CLIP 文本塔，百万级还要加上等量的图片。这个过程有三个真实问题：

1. **GPU 利用率被请求粒度拖死**。在线场景里一次请求可能只编码 1 条文本，
   而 GPU 只有在批量足够大时才接近满载。单条编码等于把 GPU 当 CPU 用。
2. **重复编码**。首屏、追问、重试、多副本，都会对同一段文本/同一张图片重复前向。
3. **编码与检索被绑死在同一进程**。CPU 侧的检索副本要为 GPU 侧的编码能力付出
   显存和依赖代价，扩检索副本变成一件昂贵的事。

本模块的三块拼图分别对应上面三点：

- ``DynamicBatcher``：**动态批处理**。把很短时间窗口内到达的多个小请求合并成一个大批次，
  GPU 的吞吐由"批次大小"决定，而不是由"请求频率"决定。
- ``EmbeddingCache``：内容哈希为键的两级缓存（进程内 LRU + 可选 Redis）。
  图片按字节哈希、文本按原文哈希，因此改名、换路径都不会导致缓存失效。
- ``RemoteEncoder``：把编码变成 HTTP 调用，直接实现 ``MultimodalEncoder`` 协议，
  于是检索副本可以完全不带 torch —— 换编码器只需换一个 URL。
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .encoders import MultimodalEncoder, normalize

DEFAULT_CACHE_CAPACITY = 4096


def text_key(text: str) -> str:
    return "t:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def image_key(path: Path) -> str:
    """按**文件内容**哈希而不是路径：同一张图换个文件名不应该重新编码。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "i:" + digest.hexdigest()


class EmbeddingCache:
    """两级 embedding 缓存：进程内 LRU 优先，可挂一层 Redis 做多副本共享。

    缓存的是**归一化后的向量**，所以命中缓存不会引入与实时编码不一致的量纲。
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CACHE_CAPACITY,
        redis_client: Any | None = None,
        namespace: str = "shopping-agent:embedding",
        ttl_seconds: int = 86400,
    ) -> None:
        self.capacity = max(1, capacity)
        self.redis_client = redis_client
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self._local: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _redis_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        with self._lock:
            for key in keys:
                vector = self._local.get(key)
                if vector is not None:
                    self._local.move_to_end(key)
                    found[key] = vector
        self.hits += len(found)
        missing = [key for key in keys if key not in found]
        if missing and self.redis_client is not None:
            payloads = self.redis_client.mget([self._redis_key(key) for key in missing])
            for key, payload in zip(missing, payloads):
                if payload is None:
                    continue
                vector = decode_vector(payload)
                if vector is not None:
                    found[key] = vector
                    self._store_local(key, vector)
            self.hits += sum(1 for key in missing if key in found)
        self.misses += sum(1 for key in keys if key not in found)
        return found

    def put_many(self, pairs: dict[str, np.ndarray]) -> None:
        for key, vector in pairs.items():
            self._store_local(key, vector)
        if self.redis_client is not None and pairs:
            pipe = self.redis_client.pipeline()
            for key, vector in pairs.items():
                pipe.set(self._redis_key(key), encode_vector(vector), ex=self.ttl_seconds)
            pipe.execute()

    def _store_local(self, key: str, vector: np.ndarray) -> None:
        with self._lock:
            self._local[key] = np.asarray(vector, dtype=np.float32)
            self._local.move_to_end(key)
            while len(self._local) > self.capacity:
                self._local.popitem(last=False)

    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        return {
            "capacity": self.capacity,
            "size": len(self._local),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "shared": self.redis_client is not None,
        }

    def clear(self) -> None:
        with self._lock:
            self._local.clear()


def encode_vector(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def decode_vector(payload: bytes | str) -> np.ndarray | None:
    if payload is None:
        return None
    if isinstance(payload, str):
        payload = payload.encode("latin-1")
    try:
        return np.frombuffer(payload, dtype=np.float32)
    except ValueError:
        return None


class _Job:
    __slots__ = ("items", "future", "enqueued_at")

    def __init__(self, items: Sequence[Any]) -> None:
        self.items = list(items)
        self.future: Future = Future()
        self.enqueued_at = time.perf_counter()


class DynamicBatcher:
    """把并发到达的小请求合并成大批次后再送进模型。

    工作方式：后台线程等待第一个请求，然后在 ``max_wait_ms`` 窗口内尽量收集更多请求，
    最多凑到 ``max_batch`` 条，一次前向算完再按请求拆分结果。

    ``max_wait_ms`` 是**吞吐与延迟的显式取舍**：调大意味着批次更大、GPU 更满，
    但每个请求的等待变长。把它做成参数而不是写死一个常量，就是为了让这个取舍可观测、可调。
    """

    def __init__(
        self,
        encode_fn: Callable[[Sequence[Any]], np.ndarray],
        max_batch: int = 256,
        max_wait_ms: float = 8.0,
        name: str = "batcher",
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch 必须大于 0")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms 不能为负")
        self.encode_fn = encode_fn
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self.name = name
        self._pending: list[_Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, name=f"{name}-worker", daemon=True)
        self._worker.start()
        self.batches = 0
        self.encoded_items = 0
        self.max_observed_batch = 0

    def submit(self, items: Sequence[Any]) -> np.ndarray:
        """阻塞直到本批编码完成。批量超过上限时自动切块。"""
        items = list(items)
        if not items:
            return np.empty((0, 0), dtype=np.float32)
        chunks = [items[start : start + self.max_batch] for start in range(0, len(items), self.max_batch)]
        results: list[np.ndarray] = []
        if len(chunks) == 1:
            return self._submit_one(items)
        # 超大请求不占用批处理窗口，直接同步算，避免把其他请求一起堵住。
        for chunk in chunks:
            results.append(self._submit_one(chunk))
        return np.concatenate(results, axis=0)

    def _submit_one(self, items: list[Any]) -> np.ndarray:
        job = _Job(items)
        with self._lock:
            self._pending.append(job)
        self._wake.set()
        return job.future.result()

    def _take(self, capacity: int) -> list[_Job]:
        """按剩余容量取任务：**严格不超过 ``max_batch``**。

        刻意不做"先收下再截断"：那会让批次大小在批次边界上超出上限，
        而批次大小直接决定显存占用——超出的那部分正是 OOM 的来源。
        """
        if capacity <= 0:
            return []
        with self._lock:
            taken: list[_Job] = []
            leftover: list[_Job] = []
            budget = capacity
            for job in self._pending:
                if len(job.items) <= budget:
                    taken.append(job)
                    budget -= len(job.items)
                else:
                    leftover.append(job)
            self._pending = leftover
            return taken

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.2):
                continue
            self._wake.clear()
            jobs = self._take(self.max_batch)
            if not jobs:
                continue
            total = sum(len(job.items) for job in jobs)
            deadline = time.perf_counter() + self.max_wait_ms / 1000.0
            # 还没到窗口上限且批次没满，就再等一会儿凑更多请求。
            while total < self.max_batch and time.perf_counter() < deadline:
                time.sleep(min(0.002, max(0.0, deadline - time.perf_counter())))
                more = self._take(self.max_batch - total)
                if not more:
                    continue
                jobs.extend(more)
                total += sum(len(job.items) for job in more)
            self._run(jobs)

    def _run(self, jobs: list[_Job]) -> None:
        flat: list[Any] = []
        for job in jobs:
            flat.extend(job.items)
        try:
            vectors = self.encode_fn(flat)
        except Exception as error:  # noqa: BLE001 - 异常必须逐请求传播，不能只打日志
            for job in jobs:
                job.future.set_exception(error)
            return
        vectors = np.asarray(vectors, dtype=np.float32)
        cursor = 0
        for job in jobs:
            size = len(job.items)
            job.future.set_result(vectors[cursor : cursor + size])
            cursor += size
        self.batches += 1
        self.encoded_items += len(flat)
        self.max_observed_batch = max(self.max_observed_batch, len(flat))

    def stats(self) -> dict[str, float | int]:
        return {
            "batches": self.batches,
            "encoded_items": self.encoded_items,
            "avg_batch_size": round(self.encoded_items / self.batches, 2) if self.batches else 0.0,
            "max_observed_batch": self.max_observed_batch,
            "max_batch": self.max_batch,
            "max_wait_ms": self.max_wait_ms,
        }

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()


class CachedEncoder:
    """给任意 ``MultimodalEncoder`` 套上缓存与动态批处理，对外仍是同一个协议。"""

    def __init__(
        self,
        encoder: MultimodalEncoder,
        cache: EmbeddingCache | None = None,
        max_batch: int = 256,
        max_wait_ms: float = 8.0,
    ) -> None:
        self.encoder = encoder
        self.cache = cache or EmbeddingCache()
        self.text_batcher = DynamicBatcher(self.encoder.encode_texts, max_batch, max_wait_ms, "text")
        self.image_batcher = DynamicBatcher(self.encoder.encode_images, max_batch, max_wait_ms, "image")

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        keys = [text_key(text) for text in texts]
        cached = self.cache.get_many(keys)
        rows: list[np.ndarray | None] = [cached.get(key) for key in keys]
        pending = [(position, text) for position, (text, key) in enumerate(zip(texts, keys)) if rows[position] is None]
        if pending:
            fresh = normalize(self.text_batcher.submit([text for _, text in pending]))
            updates: dict[str, np.ndarray] = {}
            for (position, _), vector in zip(pending, fresh):
                rows[position] = vector
                updates[keys[position]] = vector
            self.cache.put_many(updates)
        return normalize(np.asarray([row for row in rows], dtype=np.float32))

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        paths = [Path(path) for path in paths]
        keys = [image_key(path) for path in paths]
        cached = self.cache.get_many(keys)
        rows: list[np.ndarray | None] = [cached.get(key) for key in keys]
        pending = [(position, path) for position, (path, key) in enumerate(zip(paths, keys)) if rows[position] is None]
        if pending:
            fresh = normalize(self.image_batcher.submit([path for _, path in pending]))
            updates: dict[str, np.ndarray] = {}
            for (position, _), vector in zip(pending, fresh):
                rows[position] = vector
                updates[keys[position]] = vector
            self.cache.put_many(updates)
        return normalize(np.asarray([row for row in rows], dtype=np.float32))

    def stats(self) -> dict[str, Any]:
        return {
            "cache": self.cache.stats(),
            "text_batcher": self.text_batcher.stats(),
            "image_batcher": self.image_batcher.stats(),
        }

    def stop(self) -> None:
        self.text_batcher.stop()
        self.image_batcher.stop()


class RemoteEncoder:
    """把编码器换成 HTTP 服务：检索副本因此可以完全不带 torch。

    实现的是同一个 ``MultimodalEncoder`` 协议，所以 ``SemanticRetriever`` /
    ``FaissRetriever`` 都不知道自己连的是本地模型还是远程 GPU。
    """

    def __init__(self, base_url: str, timeout: float = 30.0, client: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if client is not None:
            self.client = client
        else:
            try:
                import httpx  # noqa: PLC0415
            except ImportError as error:  # pragma: no cover - 取决于环境
                raise RuntimeError('未安装 httpx，请执行 pip install httpx') from error
            self.client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _post(self, path: str, payload: dict) -> np.ndarray:
        response = self.client.post(path, json=payload)
        response.raise_for_status()
        body = response.json()
        vectors = np.asarray(body["vectors"], dtype=np.float32)
        return normalize(vectors)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._post("/embed/text", {"texts": list(texts)})

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        paths = [Path(path) for path in paths]
        if not paths:
            return np.empty((0, 0), dtype=np.float32)
        # 按 base64 传输图片字节：服务端因此不需要共享文件系统，副本可以独立部署。
        import base64

        payloads = [
            base64.b64encode(path.read_bytes()).decode("ascii") for path in paths
        ]
        return self._post("/embed/image", {"images_base64": payloads})

    def health(self) -> dict[str, Any]:
        response = self.client.get("/healthz")
        response.raise_for_status()
        return response.json()


__all__ = [
    "CachedEncoder",
    "DynamicBatcher",
    "EmbeddingCache",
    "RemoteEncoder",
    "decode_vector",
    "encode_vector",
    "image_key",
    "text_key",
]
