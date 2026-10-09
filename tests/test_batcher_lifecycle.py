"""验证突发请求、队列过载、关闭与失败后的实际可用性。"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from shopping_agent.encoder_service import BatchQueueFull, BatcherStopped, DynamicBatcher


def wait_queued(batcher, count):
    deadline = time.monotonic() + 2
    while batcher.stats()["pending_jobs"] < count:
        if time.monotonic() >= deadline:
            raise AssertionError("请求未按预期进入队列")
        time.sleep(0.001)


def blocking_encoder():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def encode(items):
        calls.extend(items)
        if items == ["busy"]:
            entered.set()
            if not release.wait(3):
                raise TimeoutError("测试未释放编码器")
        return np.ones((len(items), 2), dtype=np.float32)

    return encode, entered, release, calls


def test_burst_larger_than_batch_drains_without_new_arrivals():
    encode, entered, release, calls = blocking_encoder()
    batcher = DynamicBatcher(encode, max_batch=1, max_wait_ms=0)
    with ThreadPoolExecutor(max_workers=4) as pool:
        try:
            first = pool.submit(batcher.submit, ["busy"])
            assert entered.wait(1)
            pending = [pool.submit(batcher.submit, [str(i)]) for i in range(3)]
            wait_queued(batcher, 3)
            release.set()
            assert first.result(timeout=1).shape == (1, 2)
            for future in pending:
                assert future.result(timeout=1).shape == (1, 2)
            assert sorted(calls) == ["0", "1", "2", "busy"]
        finally:
            release.set()
            batcher.stop()


def test_queue_capacity_rejects_excess_work():
    encode, entered, release, _ = blocking_encoder()
    batcher = DynamicBatcher(encode, max_batch=1, max_wait_ms=0, max_pending_items=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(batcher.submit, ["busy"])
            assert entered.wait(1)
            queued = pool.submit(batcher.submit, ["queued"])
            wait_queued(batcher, 1)
            with pytest.raises(BatchQueueFull):
                batcher.submit(["excess"])
            assert batcher.stats()["rejected"] == 1
            release.set()
            first.result(timeout=1)
            queued.result(timeout=1)
        finally:
            release.set()
            batcher.stop()


def test_expired_queued_work_is_never_encoded():
    encode, entered, release, calls = blocking_encoder()
    batcher = DynamicBatcher(encode, max_batch=1, max_wait_ms=0, queue_timeout_seconds=0.05)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            first = pool.submit(batcher.submit, ["busy"])
            assert entered.wait(1)
            with pytest.raises(TimeoutError, match="排队超时"):
                batcher.submit(["expired"])
            assert batcher.stats()["expired"] == 1
            release.set()
            first.result(timeout=1)
            batcher.submit(["after"])
            assert calls == ["busy", "after"]
        finally:
            release.set()
            batcher.stop()


def test_shutdown_releases_queued_callers_and_rejects_new_work():
    encode, entered, release, calls = blocking_encoder()
    batcher = DynamicBatcher(encode, max_batch=1, max_wait_ms=0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(batcher.submit, ["busy"])
            assert entered.wait(1)
            queued = pool.submit(batcher.submit, ["queued"])
            wait_queued(batcher, 1)
            batcher.stop()
            with pytest.raises(BatcherStopped):
                queued.result(timeout=1)
            with pytest.raises(BatcherStopped):
                batcher.submit(["new"])
            release.set()
            first.result(timeout=1)
            assert calls == ["busy"]
        finally:
            release.set()
            batcher.stop()


def test_invalid_encoder_result_fails_request_and_worker_recovers():
    def encode(items):
        if items == ["bad"]:
            return np.zeros(1)
        return np.zeros((len(items), 2))

    batcher = DynamicBatcher(encode, max_wait_ms=0)
    try:
        with pytest.raises(ValueError, match="二维"):
            batcher.submit(["bad"])
        assert batcher.submit(["good"]).shape == (1, 2)
    finally:
        batcher.stop()
