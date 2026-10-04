from pathlib import Path

import numpy as np
import pytest

from shopping_agent.ann_index import AnnIndex, AnnIndexConfig, faiss_available
from shopping_agent.encoders import normalize
from shopping_agent.faiss_retrieval import FaissRetriever
from shopping_agent.indexing import build_ann_index, load_ann_index
from shopping_agent.models import Product, SearchRequest
from shopping_agent.semantic_retrieval import SemanticRetriever

KEYWORDS = ("鞋", "包", "帽", "衣", "表", "灯", "白", "黑", "轻", "防水")
VECTOR_DIMENSION = 32

requires_faiss = pytest.mark.skipif(not faiss_available(), reason="未安装 faiss")


class FakeEncoder:
    """关键词 → 独热坐标。刻意保持确定性，让排序结果可被逐位断言。"""

    dimension = len(KEYWORDS)

    def encode_texts(self, texts):
        rows = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for position, keyword in enumerate(KEYWORDS):
                if keyword in text:
                    rows[row, position] = 1.0
        return normalize(rows)

    def encode_images(self, paths):
        return normalize(np.ones((len(paths), self.dimension), dtype=np.float32))


class QueryOnlyEncoder:
    """只负责把查询文本映射到给定向量；商品向量由测试直接构造。

    这是唯一能同时控制"谁是近邻"和"谁能通过过滤"的方式——靠关键词独热只能造出
    大量完全重复的向量，而重复向量会让 HNSW 的图退化（实测在同一簇上 k=300 只回 74 条）。
    """

    def __init__(self, mapping: dict[str, np.ndarray], dimension: int = VECTOR_DIMENSION) -> None:
        self.mapping = {key: normalize(np.asarray([value], dtype=np.float32))[0] for key, value in mapping.items()}
        self.dimension = dimension

    def encode_texts(self, texts):
        rows = [self.mapping.get(text, np.zeros(self.dimension, dtype=np.float32)) for text in texts]
        return normalize(np.asarray(rows, dtype=np.float32))

    def encode_images(self, paths):
        return normalize(np.ones((len(paths), self.dimension), dtype=np.float32) * 0.01)


def write_catalog(path: Path, products: list[Product]) -> Path:
    path.write_text("\n".join(product.model_dump_json() for product in products) + "\n", encoding="utf-8")
    return path


def clustered_catalog(
    near: int = 250, far: int = 50, seed: int = 11
) -> tuple[list[Product], np.ndarray, np.ndarray]:
    """``near`` 条"价位 1000、语义最近"的商品 + ``far`` 条"价位 100、语义很远"的商品。

    这样"最近的那批恰好全被预算过滤掉"，过滤压力才真实。
    """
    rng = np.random.default_rng(seed)
    query = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    query[0] = 1.0
    near_vectors = normalize(np.tile(query, (near, 1)) + 0.05 * rng.normal(size=(near, VECTOR_DIMENSION)).astype(np.float32))
    far_base = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    far_base[1] = 1.0
    far_vectors = normalize(np.tile(far_base, (far, 1)) + 0.05 * rng.normal(size=(far, VECTOR_DIMENSION)).astype(np.float32))
    vectors = np.vstack([near_vectors, far_vectors]).astype(np.float32)
    products = [
        Product(id=f"shoe-{index:04d}", title="高价运动鞋", category="运动鞋", description="鞋", price=1000)
        for index in range(near)
    ] + [
        Product(id=f"bag-{index:04d}", title="低价通勤包", category="箱包", description="包", price=100)
        for index in range(far)
    ]
    return products, vectors, query


def test_matches_semantic_retriever_on_small_catalog(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [
            Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300),
            Product(id="bag-1", title="简约通勤包", category="箱包", description="包", price=200),
        ],
    )
    encoder = FakeEncoder()
    semantic = SemanticRetriever.from_jsonl(catalog, encoder)
    faiss_like = FaissRetriever.from_jsonl(catalog, encoder)
    request = SearchRequest(query="鞋", top_k=2)
    assert [hit.product.id for hit in faiss_like.search(request)] == [
        hit.product.id for hit in semantic.search(request)
    ] == ["shoe-1", "bag-1"]


@requires_faiss
def test_faiss_flat_matches_exact_backend(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [
            Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300),
            Product(id="bag-1", title="简约通勤包", category="箱包", description="包", price=200),
            Product(id="hat-1", title="棒球帽", category="帽子", description="帽", price=150),
        ],
    )
    encoder = FakeEncoder()
    exact = FaissRetriever.from_jsonl(catalog, encoder)
    flat = FaissRetriever.from_jsonl(
        catalog, encoder, AnnIndexConfig(kind="flat", dimension=encoder.dimension)
    )
    request = SearchRequest(query="鞋 包", top_k=3)
    assert [hit.product.id for hit in flat.search(request)] == [hit.product.id for hit in exact.search(request)]


def duplicate_heavy_catalog(
    near: int = 250, far: int = 50
) -> tuple[list[Product], np.ndarray, np.ndarray]:
    """全部向量**完全重复**的语料，用来制造图索引退化。

    这不是人为的刁难：真实商品库里"同款商品被重复上架"极其常见，
    大量重复向量会让 HNSW 的邻居选择退化成退化图，实测 k=300 只返回 178~210 条，
    且另一簇的 50 个点最多只找到 2 个——即使把 efSearch 提到 1024 也没有改善。
    """
    query = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    query[0] = 1.0
    near_vectors = np.tile(query, (near, 1))
    far_base = np.zeros(VECTOR_DIMENSION, dtype=np.float32)
    far_base[1] = 1.0
    far_vectors = np.tile(far_base, (far, 1))
    vectors = np.vstack([near_vectors, far_vectors]).astype(np.float32)
    products = [
        Product(id=f"shoe-{index:04d}", title="高价运动鞋", category="运动鞋", description="鞋", price=1000)
        for index in range(near)
    ] + [
        Product(id=f"bag-{index:04d}", title="低价通勤包", category="箱包", description="包", price=100)
        for index in range(far)
    ]
    return products, vectors, query


@requires_faiss
def test_under_delivery_triggers_exact_fallback_even_when_top_k_is_filled(tmp_path: Path) -> None:
    """图索引"交付不足"时必须兜底，哪怕这次凑够了 top_k。

    这是最容易漏掉的一种错：超采样把 k 放大到图索引交付不出的规模（实测 efSearch=64 时
    约在 k>512 之后开始交付不足），拿回来的候选集是"束搜索访问到的节点"而不是"最近的 k 个"。
    此时 keep 的数量照样能填满 top_k，`collected < wanted` 这个判据永远不成立——
    但结果集已经和精确检索不一致了。规模压测里这正是 HNSW 严苛过滤下 88% 集合一致的根因。
    """
    products, vectors, query = duplicate_heavy_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})

    def build(exact_vectors):
        # 注意：from_vectors 会默认把向量当作精确兜底（kwargs.setdefault），
        # 所以"没有兜底"这一档必须走构造函数，不能借用 from_vectors。
        ann = AnnIndex.build(vectors, AnnIndexConfig(kind="hnsw", dimension=VECTOR_DIMENSION, m=16, ef_search=64))
        return FaissRetriever(
            products,
            catalog,
            encoder,
            ann,
            min_candidates=len(products),  # 故意把首轮 k 顶到接近全库，逼出交付不足
            oversample=1.0,
            exact_vectors=exact_vectors,
        )

    request = SearchRequest(query="鞋", max_price=2000, top_k=5)
    exact = FaissRetriever.from_vectors(
        products, catalog, encoder, vectors, AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION)
    )

    rescued = build(vectors)
    hits = rescued.search(request)
    assert rescued.last_stats["under_delivered"] is True
    assert len(hits) == 5, "凑够了 top_k，但结果来自被污染的候选集"
    assert rescued.last_stats["fallback"] == "exact"
    assert [hit.product.id for hit in hits] == [hit.product.id for hit in exact.search(request)]

    # 没有精确向量时不能假装没事：结果照样返回，但必须把"交付不足"这件事暴露出来。
    bare = build(None)
    assert len(bare.search(request)) == 5
    assert bare.last_stats["under_delivered"] is True
    assert bare.last_stats["fallback"] == "none"
    assert bare.describe()["exact_fallback"] is False


def test_over_fetch_recovers_from_aggressive_filtering(tmp_path: Path) -> None:
    """核心回归测试：候选被过滤吃掉时必须自动扩大检索规模，而不是默默少返回。"""
    products, vectors, query = clustered_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    retriever = FaissRetriever.from_vectors(
        products,
        catalog,
        encoder,
        vectors,
        AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION),
        min_candidates=8,
        oversample=1.0,
    )
    hits = retriever.search(SearchRequest(query="鞋", max_price=500, top_k=5))
    assert len(hits) == 5, "过滤吃掉候选后没有补足 top_k"
    assert all(hit.product.category == "箱包" for hit in hits)
    assert retriever.last_stats["rounds"] >= 2, "超采样应该发生多轮"
    assert retriever.last_stats["examined"] > 8
    assert retriever.last_stats["exhausted"] is False
    assert retriever.last_stats["fallback"] == "none"


@requires_faiss
def test_exact_fallback_rescues_hnsw_when_the_graph_degenerates(tmp_path: Path) -> None:
    """HNSW 撑到全库也凑不满时，精确向量矩阵必须能把结果救回来。"""
    products, vectors, query = duplicate_heavy_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    retriever = FaissRetriever.from_vectors(
        products,
        catalog,
        encoder,
        vectors,
        AnnIndexConfig(kind="hnsw", dimension=VECTOR_DIMENSION, m=16, ef_search=64),
        min_candidates=8,
        oversample=1.0,
    )
    hits = retriever.search(SearchRequest(query="鞋", max_price=500, top_k=5))
    assert len(hits) == 5
    assert all(hit.product.category == "箱包" for hit in hits)
    assert retriever.last_stats["fallback"] == "exact"
    assert retriever.last_stats["exhausted"] is False


@requires_faiss
def test_without_exact_vectors_degradation_is_explicit(tmp_path: Path) -> None:
    """没有精确兜底时不假装成功：结果不足必须被显式标记出来。"""
    products, vectors, query = duplicate_heavy_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    retriever = FaissRetriever(
        products,
        catalog,
        encoder,
        AnnIndex.build(vectors, AnnIndexConfig(kind="hnsw", dimension=VECTOR_DIMENSION, m=16, ef_search=64)),
        min_candidates=8,
        oversample=1.0,
        exact_vectors=None,
    )
    hits = retriever.search(SearchRequest(query="鞋", max_price=500, top_k=5))
    assert hits == [], "纯 ANN 在图退化 + 极端过滤下确实取不到，这是被记录下来的能力边界"
    assert retriever.last_stats["fallback"] == "none"
    assert retriever.last_stats["exhausted"] is True
    assert retriever.describe()["exact_fallback"] is False


def test_exhausted_flag_when_catalog_smaller_than_top_k(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300)],
    )
    retriever = FaissRetriever.from_jsonl(catalog, FakeEncoder())
    hits = retriever.search(SearchRequest(query="鞋", top_k=5))
    assert len(hits) == 1
    assert retriever.last_stats["exhausted"] is True


def test_no_filter_does_not_trigger_extra_rounds(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [
            Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300),
            Product(id="bag-1", title="简约通勤包", category="箱包", description="包", price=200),
        ],
    )
    retriever = FaissRetriever.from_jsonl(catalog, FakeEncoder(), min_candidates=128)
    hits = retriever.search(SearchRequest(query="鞋", top_k=2))
    assert len(hits) == 2
    assert retriever.last_stats["rounds"] == 1
    assert retriever.last_stats["fallback"] == "none"


def test_rating_fallback_skips_vector_search(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [
            Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300, rating=4.0),
            Product(id="bag-1", title="简约通勤包", category="箱包", description="包", price=200, rating=4.9),
        ],
    )
    retriever = FaissRetriever.from_jsonl(catalog, FakeEncoder())
    hits = retriever.search(SearchRequest(top_k=2))
    assert [hit.product.id for hit in hits] == ["bag-1", "shoe-1"]
    assert retriever.last_stats["mode"] == "rating_fallback"


def test_query_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300)],
    )
    retriever = FaissRetriever.from_jsonl(catalog, FakeEncoder())
    retriever.ann.config = retriever.ann.config.model_copy(update={"dimension": 512})
    with pytest.raises(ValueError) as error:
        retriever.search(SearchRequest(query="鞋"))
    assert "维度" in str(error.value)


def test_exact_vector_count_must_match_index(tmp_path: Path) -> None:
    products, vectors, query = clustered_catalog(near=10, far=5)
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    with pytest.raises(ValueError) as error:
        FaissRetriever.from_vectors(
            products,
            catalog,
            QueryOnlyEncoder({"鞋": query}),
            vectors,
            AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION),
            exact_vectors=vectors[:3],
        )
    assert "不一致" in str(error.value)


def test_from_index_roundtrip(tmp_path: Path) -> None:
    products = [
        Product(id=f"shoe-{index:04d}", title="高价运动鞋", category="运动鞋", description="鞋", price=1000)
        for index in range(120)
    ] + [
        Product(id=f"bag-{index:04d}", title="低价通勤包", category="箱包", description="包", price=100)
        for index in range(30)
    ]
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    index_dir = tmp_path / "index"
    encoder = FakeEncoder()
    manifest, _ = build_ann_index(
        catalog,
        index_dir,
        encoder,
        "fake-clip",
        AnnIndexConfig(kind="numpy", dimension=encoder.dimension),
        image_weight=0.0,
        catalog_snapshot=True,
    )
    assert manifest.version == 2
    assert manifest.index_type == "numpy"
    assert manifest.has_vectors is True

    products_loaded, loaded_manifest, ann = load_ann_index(catalog, index_dir, "fake-clip")
    assert len(products_loaded) == len(products)
    assert loaded_manifest.model_name == "fake-clip"
    retriever = FaissRetriever.from_index(catalog, index_dir, encoder, "fake-clip")
    hits = retriever.search(SearchRequest(query="鞋", max_price=500, top_k=3))
    assert len(hits) == 3 and all(hit.product.category == "箱包" for hit in hits)
    assert ann.count == len(products)
    # 索引目录里带了精确向量，因此兜底能力是开着的。
    assert retriever.describe()["exact_fallback"] is True


def test_config_kind_mismatch_against_manifest_is_rejected(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300)],
    )
    index_dir = tmp_path / "index"
    build_ann_index(
        catalog,
        index_dir,
        FakeEncoder(),
        "fake-clip",
        AnnIndexConfig(kind="numpy", dimension=FakeEncoder.dimension),
    )
    with pytest.raises(ValueError) as error:
        load_ann_index(catalog, index_dir, None, AnnIndexConfig(kind="hnsw", dimension=FakeEncoder.dimension))
    assert "不一致" in str(error.value)


def test_keep_vectors_false_removes_matrix(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300)],
    )
    index_dir = tmp_path / "index"
    manifest, _ = build_ann_index(
        catalog,
        index_dir,
        FakeEncoder(),
        "fake-clip",
        AnnIndexConfig(kind="numpy", dimension=FakeEncoder.dimension),
        keep_vectors=False,
    )
    assert manifest.has_vectors is False
    assert not (index_dir / "vectors.npy").exists()
    with pytest.raises(ValueError) as error:
        load_ann_index(catalog, index_dir)
    assert "vectors.npy" in str(error.value)


def test_oversample_must_be_at_least_one(tmp_path: Path) -> None:
    catalog = write_catalog(
        tmp_path / "products.jsonl",
        [Product(id="shoe-1", title="轻便运动鞋", category="运动鞋", description="鞋", price=300)],
    )
    with pytest.raises(ValueError) as error:
        FaissRetriever.from_jsonl(catalog, FakeEncoder(), oversample=0.5)
    assert "oversample" in str(error.value)


# ---------- 粗召回 + 精确重排 ----------


class LyingIndex:
    """候选集完整、但**分数是假的**——模拟乘积量化用压缩码本重建出来的有偏分数。

    这类偏差最隐蔽：候选集是对的，所以"有没有召回"看不出任何问题；
    但排序是错的，用户看到的顺序就是错的。只有把分数拿去和真实余弦比对才会暴露。
    """

    def __init__(self, ann) -> None:
        self._ann = ann
        self.config = ann.config
        self.count = ann.count

    def search_one(self, query, top_k):
        _, indices = self._ann.search_one(query, top_k)
        # 分数只与"返回位置"有关：真正最相似的商品被压到末尾。
        return np.linspace(0.0, 1.0, indices.shape[0]).astype(np.float32), indices


def test_rerank_is_off_by_default(tmp_path: Path) -> None:
    products, vectors, query = clustered_catalog(near=3, far=3)
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    retriever = FaissRetriever.from_vectors(
        products,
        catalog,
        QueryOnlyEncoder({"鞋": query}),
        vectors,
        AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION),
    )
    assert retriever.describe()["rerank_candidates"] == 0
    retriever.search(SearchRequest(query="鞋", top_k=3))
    assert retriever.last_stats["rerank"] == "off"
    assert retriever.last_stats["reranked"] == 0


def test_rerank_without_vectors_is_rejected(tmp_path: Path) -> None:
    """配了重排却拿不到向量必须报错。

    静默不生效的代价不是"白配了"，而是"得到一个"重排没用"的错误结论"——
    这会让人把已经有效的优化删掉。
    """
    products, vectors, query = clustered_catalog(near=4, far=4)
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    with pytest.raises(ValueError, match="vectors.npy"):
        FaissRetriever.from_vectors(
            products,
            catalog,
            QueryOnlyEncoder({"鞋": query}),
            vectors,
            AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION),
            exact_vectors=None,
            rerank_candidates=10,
        )


def test_rerank_recovers_the_true_ranking_from_corrupted_scores(tmp_path: Path) -> None:
    """把候选集完整但分数被打乱的索引接上重排，排序应当完全恢复。"""
    products, vectors, query = clustered_catalog(near=60, far=60)
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    config = AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION)
    request = SearchRequest(query="鞋", top_k=5)
    expected = [hit.product.id for hit in FaissRetriever.from_vectors(products, catalog, encoder, vectors, config).search(request)]

    def build(**kwargs) -> FaissRetriever:
        retriever = FaissRetriever.from_vectors(
            products,
            catalog,
            encoder,
            vectors,
            config,
            min_candidates=len(products),
            oversample=1.0,
            **kwargs,
        )
        retriever.ann = LyingIndex(retriever.ann)
        return retriever

    naive = build()
    corrupted = [hit.product.id for hit in naive.search(request)]
    # 先证明这个测试确实造出了偏差，否则后面的"恢复"是空断言。
    assert corrupted != expected
    assert naive.last_stats["rerank"] == "off"
    assert naive.last_stats["target"] == 5

    reranked = build(rerank_candidates=40)
    assert [hit.product.id for hit in reranked.search(request)] == expected
    assert reranked.last_stats["rerank"] == "exact"
    assert reranked.last_stats["reranked"] == len(products)
    assert reranked.last_stats["target"] == 40


@requires_faiss
def test_rerank_restores_exact_scores_on_a_compressed_index(tmp_path: Path) -> None:
    """PQ 重建出来的分数是有偏的，重排把它换回真实余弦相似度。

    这是 IVF-PQ 能用的前提：它直接用的 recall@10 只有 0.097，但候选集是好的，
    取大候选集 + 精确重排后能回到 0.94。没有重排，压缩换来的是静默的错排序。
    """
    products, vectors, query = clustered_catalog(near=90, far=90)
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    # pq_bits=2 有两个目的：让码本足够粗、偏差足够明显；把 PQ 训练门槛降到 39×4=156 条。
    config = AnnIndexConfig(kind="ivfpq", dimension=VECTOR_DIMENSION, pq_bits=2, nlist=8, nprobe=4)
    request = SearchRequest(query="鞋", top_k=5)
    position = {product.id: index for index, product in enumerate(products)}
    exact_scores = vectors @ query

    def search(**kwargs):
        retriever = FaissRetriever.from_vectors(
            products,
            catalog,
            encoder,
            vectors,
            config,
            min_candidates=len(products),
            oversample=1.0,
            **kwargs,
        )
        return retriever, retriever.search(request)

    naive, naive_hits = search()
    assert naive.last_stats["rerank"] == "off"
    assert any(
        abs(hit.score - float(exact_scores[position[hit.product.id]])) > 1e-3 for hit in naive_hits
    ), "IVF-PQ 的分数与真实余弦一致，说明这个测试没能体现出量化偏差"

    reranked, reranked_hits = search(rerank_candidates=len(products))
    assert reranked.last_stats["rerank"] == "exact"
    for hit in reranked_hits:
        assert abs(hit.score - float(exact_scores[position[hit.product.id]])) < 2e-4
    # 候选集完整 + 分数精确 ⇒ 结果必须与精确后端逐位相同。
    exact = FaissRetriever.from_vectors(
        products, catalog, encoder, vectors, AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION)
    )
    assert [hit.product.id for hit in reranked_hits] == [hit.product.id for hit in exact.search(request)]
