from pathlib import Path

import numpy as np
import pytest

from shopping_agent.ann_index import (
    ANN_INDEX_FILENAME,
    AnnIndex,
    AnnIndexConfig,
    auto_pq_segments,
    faiss_available,
)

requires_faiss = pytest.mark.skipif(not faiss_available(), reason="未安装 faiss")


def sample(dimension: int = 32, count: int = 64, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(4, dimension)).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    return vectors, queries


def brute_force(vectors: np.ndarray, query: np.ndarray, top_k: int) -> list[int]:
    scores = vectors @ query
    return [int(index) for index in np.argsort(-scores)[:top_k]]


def test_numpy_backend_matches_brute_force() -> None:
    vectors, queries = sample()
    ann = AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=vectors.shape[1]))
    _, indices = ann.search(queries, 5)
    for query, row in zip(queries, indices):
        assert [int(item) for item in row] == brute_force(vectors, query, 5)


@pytest.mark.parametrize("metric", ["ip", "l2"])
@pytest.mark.parametrize("top_k", [1, 5, 64, 128])
def test_numpy_ties_preserve_catalog_order_at_candidate_boundary(metric: str, top_k: int) -> None:
    vectors = np.concatenate([np.tile([[1.0, 0.0]], (96, 1)), np.tile([[0.0, 1.0]], (32, 1))])
    queries = np.eye(2, dtype=np.float32)
    ann = AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=2, metric=metric))
    _, indices = ann.search(queries, top_k)
    for query, row in zip(queries, indices):
        scores = vectors @ query if metric == "ip" else -((vectors - query) ** 2).sum(axis=1)
        expected = np.argsort(-scores, kind="stable")[:top_k]
        assert row.tolist() == expected.tolist()


def test_numpy_backend_is_exact_but_faiss_flat_agrees() -> None:
    pytest.importorskip("faiss")
    vectors, queries = sample()
    exact = AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=vectors.shape[1]))
    flat = AnnIndex.build(vectors, AnnIndexConfig(kind="flat", dimension=vectors.shape[1]))
    _, left = exact.search(queries, 5)
    _, right = flat.search(queries, 5)
    # IndexFlatIP 也是精确检索，因此两个后端的结果必须逐位一致。
    assert np.array_equal(left, right)


@requires_faiss
@pytest.mark.parametrize("kind", ["flat", "hnsw", "ivf", "ivfpq"])
def test_faiss_kinds_return_valid_indices(kind: str) -> None:
    vectors, queries = sample(dimension=64, count=20_000)
    config = AnnIndexConfig(kind=kind, dimension=64, nlist=64, nprobe=8)
    ann = AnnIndex.build(vectors, config)
    scores, indices = ann.search(queries, 10)
    assert indices.shape == (queries.shape[0], 10)
    assert indices.min() >= 0 and indices.max() < vectors.shape[0]
    assert np.all(np.diff(scores, axis=1) <= 1e-6), "分数必须按降序返回"
    assert ann.index_bytes > 0


@requires_faiss
def test_ivfpq_compresses_memory_and_is_lossy() -> None:
    """PQ 的取舍：内存大幅下降，召回必然有损（不写成"召回不变"，那是自欺）。"""
    vectors, queries = sample(dimension=128, count=30_000)
    exact = AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=128))
    pq = AnnIndex.build(
        vectors, AnnIndexConfig(kind="ivfpq", dimension=128, nlist=128, nprobe=32, pq_segments=32)
    )
    _, truth = exact.search(queries, 10)
    _, got = pq.search(queries, 10)
    recall = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth.tolist(), got.tolist())]))
    assert pq.index_bytes < exact.index_bytes / 4, "PQ 必须显著压缩内存"
    assert recall < 0.95, "PQ 是有损的，召回率不可能与精确检索持平"
    assert recall > 0.05, "压缩不该把所有近邻都丢掉"


@requires_faiss
def test_ivf_nprobe_trades_recall_for_latency() -> None:
    """nprobe 是查询期旋钮：扫更多桶必然召回更高、延迟更差。"""
    vectors, queries = sample(dimension=64, count=20_000)
    exact = AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=64))
    _, truth = exact.search(queries, 10)

    def recall(nprobe: int) -> float:
        ann = AnnIndex.build(
            vectors, AnnIndexConfig(kind="ivf", dimension=64, nlist=64, nprobe=nprobe)
        )
        _, got = ann.search(queries, 10)
        return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth.tolist(), got.tolist())]))

    assert recall(8) <= recall(64)


@requires_faiss
def test_hnsw_cannot_return_full_neighbour_set_beyond_ef() -> None:
    """记录一条实测边界：HNSW 返回的是"束搜索访问到的节点"，不是"最近的 k 个"。

    这正是 FaissRetriever 需要精确兜底的原因——请求 k = 全库规模时，
    HNSW 仍可能返回 -1 填充，而提高 efSearch 并不会让图遍历自动覆盖全库。
    """
    rng = np.random.default_rng(3)
    dimension = 32
    query = np.zeros(dimension, dtype=np.float32)
    query[0] = 1.0
    near = np.tile(query, (250, 1)) + 0.05 * rng.normal(size=(250, dimension)).astype(np.float32)
    far = np.zeros((50, dimension), dtype=np.float32)
    far[:, 1] = 1.0
    far = far + 0.05 * rng.normal(size=(50, dimension)).astype(np.float32)
    vectors = np.vstack([near, far])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    ann = AnnIndex.build(vectors, AnnIndexConfig(kind="hnsw", dimension=dimension, m=16, ef_search=64))
    _, indices = ann.search(query.reshape(1, -1), vectors.shape[0])
    assert (indices[0] < 0).any(), "HNSW 在 k=全库规模时仍然会漏，这不是 bug 而是它的性质"


def test_missing_faiss_gives_actionable_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """faiss 缺失时必须给可执行的安装提示，而不是甩一个 ImportError 堆栈。

    这是"faiss 是可选依赖"这个承诺的实现细节：无论当前环境装没装 faiss，
    都要能验证这条降级路径，所以这里临时把 import 拦住。
    """
    import builtins
    import sys

    from shopping_agent import ann_index as module

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "faiss" or name.startswith("faiss."):
            raise ImportError("simulated missing faiss")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "faiss", raising=False)
    with pytest.raises(RuntimeError, match="pip install"):
        module.faiss_module()
    assert module.faiss_available() is False


@requires_faiss
def test_ivfpq_rejects_insufficient_training_data() -> None:
    """样本不足时显式拒绝，而不是跑出一个被低估的假分数。

    注意这条断言依赖 faiss 的存在：没有 faiss 时 `AnnIndex.build` 会先因为
    缺少依赖而抛 RuntimeError，那条路径由 `test_missing_faiss_gives_actionable_hint` 覆盖。
    """
    vectors, _ = sample(dimension=32, count=500)
    with pytest.raises(ValueError) as error:
        AnnIndex.build(vectors, AnnIndexConfig(kind="ivfpq", dimension=32))
    assert "9984" in str(error.value)


def test_config_rejects_bad_pq_segments() -> None:
    with pytest.raises(ValueError) as error:
        AnnIndexConfig(kind="ivfpq", dimension=100, pq_segments=64)
    assert "整除" in str(error.value)


def test_config_rejects_l2_for_faiss_backends() -> None:
    with pytest.raises(ValueError) as error:
        AnnIndexConfig(kind="hnsw", dimension=32, metric="l2")
    assert "ip" in str(error.value)


def test_auto_pq_segments_divides_dimension() -> None:
    assert auto_pq_segments(512) == 64
    assert auto_pq_segments(768) == 64
    assert 512 % auto_pq_segments(512) == 0
    assert 2 % auto_pq_segments(2) == 0


@requires_faiss
def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    vectors, queries = sample(dimension=64, count=5_000)
    config = AnnIndexConfig(kind="hnsw", dimension=64, ef_search=32)
    ann = AnnIndex.build(vectors, config)
    _, expected = ann.search(queries, 5)
    files = ann.save(tmp_path)
    assert files == [ANN_INDEX_FILENAME]
    assert (tmp_path / ANN_INDEX_FILENAME).exists()
    restored = AnnIndex.load(tmp_path, config, expected_count=vectors.shape[0])
    _, actual = restored.search(queries, 5)
    assert np.array_equal(expected, actual)


@requires_faiss
def test_load_rejects_wrong_count(tmp_path: Path) -> None:
    vectors, _ = sample(dimension=32, count=1_000)
    config = AnnIndexConfig(kind="flat", dimension=32)
    AnnIndex.build(vectors, config).save(tmp_path)
    with pytest.raises(ValueError) as error:
        AnnIndex.load(tmp_path, config, expected_count=999)
    assert "不一致" in str(error.value)


def test_numpy_backend_requires_vectors_on_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as error:
        AnnIndex.load(tmp_path, AnnIndexConfig(kind="numpy", dimension=8))
    assert "vectors.npy" in str(error.value)


@requires_faiss
def test_hnsw_ef_search_trades_recall_for_latency() -> None:
    """efSearch 是查询期旋钮：不需要重建索引就能调整召回率，这是文档里必须能复现的结论。"""
    vectors, queries = sample(dimension=128, count=20_000)
    exact = AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=128))
    _, truth = exact.search(queries, 10)

    def recall(ef_search: int) -> float:
        ann = AnnIndex.build(vectors, AnnIndexConfig(kind="hnsw", dimension=128, m=16, ef_search=ef_search))
        _, got = ann.search(queries, 10)
        return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth.tolist(), got.tolist())]))

    assert recall(64) >= recall(4)


def test_summary_reports_effective_params() -> None:
    vectors, _ = sample(dimension=16, count=100)
    ann = AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=16))
    summary = ann.summary()
    assert summary["count"] == 100
    assert summary["backend"] == "numpy"
    assert summary["bytes_per_vector"] == 64.0
    assert "numpy" in ann.config.describe()


def test_build_rejects_dimension_mismatch() -> None:
    vectors, _ = sample(dimension=16, count=8)
    with pytest.raises(ValueError) as error:
        AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=32))
    assert "维度" in str(error.value)


def test_build_rejects_empty_collection() -> None:
    with pytest.raises(ValueError) as error:
        AnnIndex.build(np.empty((0, 8), dtype=np.float32), AnnIndexConfig(kind="numpy", dimension=8))
    assert "空" in str(error.value)

