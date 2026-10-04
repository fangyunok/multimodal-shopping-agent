import threading
import time
from pathlib import Path

import numpy as np

from shopping_agent.encoder_service import (
    CachedEncoder,
    DynamicBatcher,
    EmbeddingCache,
    RemoteEncoder,
    decode_vector,
    encode_vector,
    image_key,
    text_key,
)


class CountingEncoder:
    """记录每次实际收到的批次大小，用来证明批处理与缓存确实生效。"""

    dimension = 4

    def __init__(self) -> None:
        self.text_calls: list[int] = []
        self.image_calls: list[int] = []
        self.lock = threading.Lock()

    def encode_texts(self, texts):
        with self.lock:
            self.text_calls.append(len(texts))
        time.sleep(0.01)
        return np.asarray([[float(len(text))] * self.dimension for text in texts], dtype=np.float32)

    def encode_images(self, paths):
        with self.lock:
            self.image_calls.append(len(paths))
        return np.asarray([[float(len(str(path)))] * self.dimension for path in paths], dtype=np.float32)


def test_cache_put_get_and_stats() -> None:
    cache = EmbeddingCache(capacity=4)
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    cache.put_many({"a": vector})
    assert np.array_equal(cache.get_many(["a"])["a"], vector)
    cache.get_many(["missing"])
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["shared"] is False


def test_cache_evicts_least_recently_used() -> None:
    cache = EmbeddingCache(capacity=2)
    for name in ("a", "b", "c"):
        cache.put_many({name: np.asarray([float(ord(name))], dtype=np.float32)})
    assert set(cache.get_many(["a", "b", "c"])) == {"b", "c"}
    assert cache.stats()["size"] == 2


def test_cache_shares_through_redis_client() -> None:
    class FakeRedis:
        def __init__(self) -> None:
            self.data: dict[str, bytes] = {}

        def mget(self, keys):
            return [self.data.get(key) for key in keys]

        def pipeline(self):
            outer = self

            class Pipe:
                def __init__(self) -> None:
                    self.buffer: list[tuple[str, bytes]] = []

                def set(self, key, value, ex=None):
                    self.buffer.append((key, value))

                def execute(self):
                    for key, value in self.buffer:
                        outer.data[key] = value

            return Pipe()

    client = FakeRedis()
    writer = EmbeddingCache(redis_client=client)
    writer.put_many({"k": np.asarray([3.0, 4.0], dtype=np.float32)})

    reader = EmbeddingCache(redis_client=client)
    assert np.allclose(reader.get_many(["k"])["k"], [3.0, 4.0])
    assert reader.stats()["shared"] is True


def test_vector_codec_roundtrip() -> None:
    vector = np.asarray([1.5, -2.25], dtype=np.float32)
    assert np.array_equal(decode_vector(encode_vector(vector)), vector)
    assert decode_vector(b"\x01") is None


def test_text_key_is_content_based() -> None:
    assert text_key("运动鞋") == text_key("运动鞋")
    assert text_key("运动鞋") != text_key("运动鞋 ")


def test_image_key_ignores_filename(tmp_path: Path) -> None:
    """按内容哈希：同一张图换个文件名不应该重新编码。"""
    original = tmp_path / "a.png"
    original.write_bytes(b"fake-image-bytes")
    renamed = tmp_path / "b.png"
    renamed.write_bytes(b"fake-image-bytes")
    assert image_key(original) == image_key(renamed)


def test_batcher_merges_concurrent_requests() -> None:
    encoder = CountingEncoder()
    batcher = DynamicBatcher(encoder.encode_texts, max_batch=64, max_wait_ms=60)
    try:
        barrier = threading.Barrier(8)
        results: list[np.ndarray | None] = [None] * 8

        def worker(index: int) -> None:
            barrier.wait()
            results[index] = batcher.submit([f"text-{index}"])

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        batcher.stop()

    stats = batcher.stats()
    assert stats["encoded_items"] == 8
    assert stats["max_observed_batch"] >= 2, "并发小请求应该被合并成更大的批次"
    assert stats["max_observed_batch"] <= 64
    for index, result in enumerate(results):
        assert result is not None and result.shape == (1, 4)
        assert result[0][0] == float(len(f"text-{index}"))


def test_batcher_splits_oversized_requests_without_exceeding_limit() -> None:
    encoder = CountingEncoder()
    batcher = DynamicBatcher(encoder.encode_texts, max_batch=4, max_wait_ms=1)
    try:
        vectors = batcher.submit([f"t{index}" for index in range(10)])
    finally:
        batcher.stop()
    assert vectors.shape[0] == 10
    assert all(size <= 4 for size in encoder.text_calls), "任何一批都不能超过 max_batch"
    assert sum(encoder.text_calls) == 10


def test_batcher_validates_arguments() -> None:
    encoder = CountingEncoder()
    for kwargs, message in (({"max_batch": 0}, "max_batch"), ({"max_wait_ms": -1}, "max_wait_ms")):
        try:
            DynamicBatcher(encoder.encode_texts, **kwargs)
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError("非法参数必须被拒绝")


def test_cached_encoder_avoids_reencoding() -> None:
    encoder = CountingEncoder()
    cached = CachedEncoder(encoder, max_batch=32, max_wait_ms=1)
    try:
        first = cached.encode_texts(["运动鞋", "通勤包"])
        second = cached.encode_texts(["运动鞋", "通勤包"])
    finally:
        cached.stop()
    assert np.allclose(first, second)
    assert sum(encoder.text_calls) == 2, "第二次应该全部命中缓存"
    assert cached.stats()["cache"]["hits"] == 2


def test_cached_encoder_only_encodes_cache_misses() -> None:
    encoder = CountingEncoder()
    cached = CachedEncoder(encoder, max_batch=32, max_wait_ms=1)
    try:
        cached.encode_texts(["运动鞋"])
        cached.encode_texts(["运动鞋", "通勤包", "棒球帽"])
    finally:
        cached.stop()
    assert sum(encoder.text_calls) == 3, "第二次只应对未命中的两条做编码"


def test_cached_encoder_handles_images(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"bytes")
    encoder = CountingEncoder()
    cached = CachedEncoder(encoder, max_batch=8, max_wait_ms=1)
    try:
        first = cached.encode_images([image])
        second = cached.encode_images([image])
    finally:
        cached.stop()
    assert np.allclose(first, second)
    assert sum(encoder.image_calls) == 1


class FakeHttpClient:
    """替身 httpx.Client：只记录请求并返回确定性的向量。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def post(self, path: str, json: dict):
        self.calls.append((path, json))

        class Response:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self.payload

        count = len(json.get("texts") or json.get("images_base64") or [])
        return Response({"vectors": [[1.0, 0.0, 0.0, 0.0]] * count})

    def get(self, path: str):
        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"status": "ok"}

        return Response()


def test_remote_encoder_implements_protocol(tmp_path: Path) -> None:
    client = FakeHttpClient()
    encoder = RemoteEncoder("http://encoder:8100", client=client)
    texts = encoder.encode_texts(["运动鞋", "通勤包"])
    assert texts.shape == (2, 4)
    assert np.allclose(np.linalg.norm(texts, axis=1), 1.0)
    assert client.calls[0][0] == "/embed/text"

    image = tmp_path / "a.png"
    image.write_bytes(b"bytes")
    images = encoder.encode_images([image])
    assert images.shape == (1, 4)
    assert client.calls[1][0] == "/embed/image"
    # 图片走 base64 而不是路径：服务端不应能读客户端文件系统。
    payload = client.calls[1][1]
    assert "images_base64" in payload and "path" not in payload
    assert encoder.health()["status"] == "ok"


def test_remote_encoder_handles_empty_input() -> None:
    encoder = RemoteEncoder("http://encoder:8100", client=FakeHttpClient())
    assert encoder.encode_texts([]).shape == (0, 0)
    assert encoder.encode_images([]).shape == (0, 0)
