"""过滤下推（分段索引路由）的测试。

核心不变式只有一条：**路由不能改变结果集**。

路由判断一旦判错，代价不对称——多扫一个分段只是慢一点，少扫一个分段就是永久漏商品。
所以这里的测试重点不是"性能变好了"，而是"结果和单索引完全一致"。
性能差异属于压测（``scripts/benchmark_ann_scaling.py``）的职责，不该由单元测试来断言。
"""

from pathlib import Path

import numpy as np
import pytest

from shopping_agent.ann_index import AnnIndexConfig, faiss_available
from shopping_agent.encoders import normalize
from shopping_agent.faiss_retrieval import FaissRetriever
from shopping_agent.indexing import load_ann_index
from shopping_agent.models import Product, SearchRequest
from shopping_agent.partitioned import (
    PartitionEntry,
    PartitionedRetriever,
    plan_partitions,
)

VECTOR_DIMENSION = 32
requires_faiss = pytest.mark.skipif(not faiss_available(), reason="未安装 faiss")


class QueryOnlyEncoder:
    """只把查询文本映射到给定向量；商品向量由测试直接构造。"""

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


def banded_catalog(
    near: int = 60, far: int = 60, dimension: int = VECTOR_DIMENSION, seed: int = 5
) -> tuple[list[Product], np.ndarray, np.ndarray]:
    """语义最近的一批商品**全部超预算**，能过预算的那批语义很远。

    这正是过滤下推要对付的形态：过滤条件与向量相似度无关，
    于是"最像的"和"能用的"是两个几乎不相交的集合。
    """
    rng = np.random.default_rng(seed)
    query = np.zeros(dimension, dtype=np.float32)
    query[0] = 1.0
    near_vectors = normalize(
        np.tile(query, (near, 1)) + 0.05 * rng.normal(size=(near, dimension)).astype(np.float32)
    )
    far_base = np.zeros(dimension, dtype=np.float32)
    far_base[1] = 1.0
    far_vectors = normalize(
        np.tile(far_base, (far, 1)) + 0.05 * rng.normal(size=(far, dimension)).astype(np.float32)
    )
    vectors = np.vstack([near_vectors, far_vectors]).astype(np.float32)
    products = [
        Product(
            id=f"shoe-{index:03d}",
            title="高价运动鞋",
            category="运动鞋",
            description="鞋",
            price=1000.0 + index * 10,
        )
        for index in range(near)
    ] + [
        Product(
            id=f"bag-{index:03d}",
            title="低价通勤包",
            category="箱包",
            description="包",
            price=60.0 + index,
        )
        for index in range(far)
    ]
    return products, vectors, query


def exact_retriever(products, catalog, encoder, vectors) -> FaissRetriever:
    """单索引精确后端：分段路由的结果必须与它逐位一致。"""
    return FaissRetriever.from_vectors(
        products, catalog, encoder, vectors, AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION)
    )


def partitioned(products, catalog, encoder, vectors, **kwargs) -> PartitionedRetriever:
    return PartitionedRetriever.from_vectors(
        products,
        catalog,
        encoder,
        vectors,
        AnnIndexConfig(kind="numpy", dimension=VECTOR_DIMENSION),
        **kwargs,
    )


# ---------- 分段规划 ----------


def test_plan_partitions_is_a_partition_of_the_catalog() -> None:
    """分段的定义就是划分：两两不交、并集为全集。这是路由正确性的前提。"""
    products, _, _ = banded_catalog()
    for by in ("price", "category", "category_price"):
        planned = plan_partitions(products, by, buckets=4)  # type: ignore[arg-type]
        merged = sorted(row for _, members in planned for row in members)
        assert merged == list(range(len(products))), f"{by} 不是目录的一个划分"
        assert all(spec.size == len(members) for spec, members in planned)
        assert all(members for _, members in planned), "不应产出空分段"


def test_price_buckets_are_quantile_based_not_equal_width() -> None:
    """等宽切会让绝大多数商品挤进第一个桶；分位数保证每段商品数相近。"""
    products, _, _ = banded_catalog()
    planned = plan_partitions(products, "price", buckets=4)
    sizes = [spec.size for spec, _ in planned]
    assert len(sizes) == 4
    assert max(sizes) - min(sizes) <= 2, f"分段大小应接近，实际 {sizes}"


def test_may_match_uses_minimum_price_as_exact_lower_bound() -> None:
    """段内最低价就是下界：它精确，所以"跳过这个段"永远不会是错的。"""
    products, _, _ = banded_catalog()
    planned = plan_partitions(products, "price", buckets=4)
    cheap = next(spec for spec, _ in planned if spec.price_low <= 100)
    expensive = next(spec for spec, _ in planned if spec.price_low >= 1000)

    assert cheap.may_match(SearchRequest(query="鞋", max_price=100)) is True
    # 差一分钱也要跳过——边界必须严格，否则预算会被悄悄放宽。
    assert expensive.may_match(SearchRequest(query="鞋", max_price=999.99)) is False
    assert expensive.may_match(SearchRequest(query="鞋", max_price=1000)) is True
    # 不做价格过滤时不该剪掉任何段。
    assert expensive.may_match(SearchRequest(query="鞋")) is True


def test_category_routing_matches_substring_semantics() -> None:
    products, _, _ = banded_catalog()
    planned = plan_partitions(products, "category", buckets=1)
    shoes = next(spec for spec, _ in planned if spec.categories == ("运动鞋",))
    bags = next(spec for spec, _ in planned if spec.categories == ("箱包",))

    assert shoes.may_match(SearchRequest(query="鞋", category="运动")) is True
    assert bags.may_match(SearchRequest(query="鞋", category="运动")) is False
    # 与 FaissRetriever 的过滤语义一致：子串匹配，不是精确相等。
    assert bags.may_match(SearchRequest(query="鞋", category="通勤包")) is False


# ---------- 路由不改变结果集 ----------


def test_routing_does_not_change_the_result_set(tmp_path: Path) -> None:
    """本模块最重要的一条：加上过滤、加上路由之后，结果必须与单索引精确后端逐位相同。"""
    products, vectors, query = banded_catalog(far=120)
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})

    reference = exact_retriever(products, catalog, encoder, vectors)
    routed = partitioned(products, catalog, encoder, vectors, by="price", buckets=6)

    requests = [
        SearchRequest(query="鞋", top_k=5),
        SearchRequest(query="鞋", max_price=120, top_k=5),
        SearchRequest(query="鞋", max_price=2000, top_k=5),
        SearchRequest(query="鞋", category="箱包", top_k=4),
        SearchRequest(query="鞋", max_price=1000, category="运动鞋", top_k=3),
        SearchRequest(query="鞋", max_price=59, top_k=5),
    ]
    for request in requests:
        expected = [hit.product.id for hit in reference.search(request)]
        actual = [hit.product.id for hit in routed.search(request)]
        assert actual == expected, f"{request} 路由后结果不一致：{actual} != {expected}"


def test_routing_prunes_only_when_the_filter_is_selective(tmp_path: Path) -> None:
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(products, catalog, encoder, vectors, by="price", buckets=4)

    routed.search(SearchRequest(query="鞋", max_price=120, top_k=5))
    selective = dict(routed.last_stats)
    assert selective["partitions_scanned"] < selective["partitions_total"]
    assert selective["scanned_fraction"] < 1.0

    routed.search(SearchRequest(query="鞋", top_k=5))
    loose = dict(routed.last_stats)
    # 不过滤时所有分段都要扫——下推的收益完全来自过滤的选择性，这一点必须如实反映出来。
    assert loose["partitions_scanned"] == loose["partitions_total"]
    assert loose["scanned_fraction"] == 1.0


def test_category_partitioning_scans_only_the_matching_bucket(tmp_path: Path) -> None:
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(products, catalog, encoder, vectors, by="category", buckets=1)

    hits = routed.search(SearchRequest(query="鞋", category="箱包", top_k=5))
    assert {hit.product.category for hit in hits} == {"箱包"}
    assert routed.last_stats["partitions_scanned"] == 1
    assert routed.last_stats["partitions_total"] == 2


def test_empty_route_returns_nothing_rather_than_everything(tmp_path: Path) -> None:
    """没有分段能命中时，必须返回空——退化成"扫全库"会让预算过滤失效。"""
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(products, catalog, encoder, vectors, by="price", buckets=4)

    hits = routed.search(SearchRequest(query="鞋", max_price=1, top_k=5))
    assert hits == []
    assert routed.last_stats["partitions_scanned"] == 0


def test_rating_fallback_is_also_routed(tmp_path: Path) -> None:
    """无文字无图片时按评分排序，同样要走路由，而不是绕过分段扫全库。"""
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(products, catalog, encoder, vectors, by="price", buckets=4)

    hits = routed.search(SearchRequest(max_price=120, top_k=3))
    assert hits and all(hit.product.price <= 120 for hit in hits)
    assert all(hit.score == 0 for hit in hits)
    assert routed.last_stats["partitions_pruned"] > 0


def test_overlapping_partitions_are_rejected(tmp_path: Path) -> None:
    """不是划分的分段会重复或漏掉商品，必须在构造时就拒绝，而不是等结果出错。"""
    products, vectors, query = banded_catalog(near=4, far=4)
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(products, catalog, encoder, vectors, by="price", buckets=2)

    duplicated = [
        PartitionEntry(spec=routed.entries[0].spec, members=routed.entries[0].members, retriever=routed.entries[0].retriever),
        PartitionEntry(spec=routed.entries[0].spec, members=routed.entries[0].members, retriever=routed.entries[0].retriever),
    ]
    with pytest.raises(ValueError, match="划分"):
        PartitionedRetriever(products, catalog, encoder, routed.layout, duplicated)


# ---------- 精确兜底向量按全库语义切片 ----------


def test_full_catalog_vectors_are_sliced_per_partition(tmp_path: Path) -> None:
    """``exact_vectors`` 按**全库**语义传入，分段在内部切片。

    调用方不该为了用分段索引，先自己按分段把向量切一遍——那既多余，又极易切错段。
    """
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(
        products, catalog, encoder, vectors, by="price", buckets=4, exact_vectors=vectors
    )

    assert routed.layout.has_vectors is True
    for entry in routed.entries:
        subset = entry.retriever.exact_vectors
        assert subset is not None
        assert subset.shape[0] == entry.spec.size, "每段拿到的向量条数必须等于段内商品数"
        # 值也要对得上：形状对了但内容是别段的向量，症状是"静默返回错商品"，比报错难查得多。
        assert np.allclose(subset, normalize(vectors[entry.members]))


def test_explicit_none_disables_the_fallback(tmp_path: Path) -> None:
    """显式传 ``None`` 是"不要精确兜底"的明确意图，不能被默认值悄悄改回 True。"""
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(
        products, catalog, encoder, vectors, by="price", buckets=4, exact_vectors=None
    )

    assert routed.layout.has_vectors is False
    assert all(entry.retriever.exact_vectors is None for entry in routed.entries)


def test_vectors_of_the_wrong_length_point_at_the_full_catalog(tmp_path: Path) -> None:
    """传了"已经切好的一段"是常见误用，报错要说清正确用法，而不是抛一个形状错误。"""
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    with pytest.raises(ValueError, match="全库"):
        partitioned(
            products, catalog, encoder, vectors, by="price", buckets=4, exact_vectors=vectors[:5]
        )


def test_load_cannot_swap_the_vectors(tmp_path: Path) -> None:
    """向量在不在磁盘上是构建期决定的；加载时另给一份会让"能不能重排"有两个来源。"""
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    partitioned(products, catalog, encoder, vectors, by="price", buckets=4).save(tmp_path / "index")

    with pytest.raises(ValueError, match="构建期参数"):
        PartitionedRetriever.from_index(catalog, tmp_path / "index", encoder, exact_vectors=vectors)


# ---------- 落盘与加载 ----------


def test_save_and_load_round_trips(tmp_path: Path) -> None:
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(products, catalog, encoder, vectors, by="price", buckets=4, model_name="test-model")
    routed.save(tmp_path / "index", catalog_snapshot=True)

    assert (tmp_path / "index" / "manifest.json").exists(), "分段索引也要写标准清单，否则进不了发布/回滚链路"
    assert (tmp_path / "index" / "partitions.json").exists()

    reloaded = PartitionedRetriever.from_index(catalog, tmp_path / "index", encoder)
    assert reloaded.describe()["partitions"] == routed.describe()["partitions"]

    request = SearchRequest(query="鞋", max_price=120, top_k=5)
    reference = exact_retriever(products, catalog, encoder, vectors)
    assert [hit.product.id for hit in reloaded.search(request)] == [
        hit.product.id for hit in reference.search(request)
    ]


def test_load_rejects_a_mismatched_catalog(tmp_path: Path) -> None:
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = partitioned(products, catalog, encoder, vectors, by="price", buckets=4)
    routed.save(tmp_path / "index")

    changed = [product.model_copy(update={"price": product.price + 1}) for product in products]
    write_catalog(catalog, changed)
    with pytest.raises(ValueError, match="商品目录已变化"):
        PartitionedRetriever.from_index(catalog, tmp_path / "index", encoder)


def test_single_index_loader_points_at_partitioned_loader(tmp_path: Path) -> None:
    """按单索引入口加载分段索引必须给出可执行的报错，而不是一个溯源不出来的维度错误。"""
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    partitioned(products, catalog, encoder, vectors, by="price", buckets=4).save(tmp_path / "index")

    with pytest.raises(ValueError, match="PartitionedRetriever.from_index"):
        load_ann_index(catalog, tmp_path / "index")


@requires_faiss
def test_partitioned_hnsw_never_violates_the_filter(tmp_path: Path) -> None:
    """换到真实 ANN 后端时，路由 + 段内检索的结果不允许出现任何越界商品。"""
    products, vectors, query = banded_catalog()
    catalog = write_catalog(tmp_path / "products.jsonl", products)
    encoder = QueryOnlyEncoder({"鞋": query})
    routed = PartitionedRetriever.from_vectors(
        products,
        catalog,
        encoder,
        vectors,
        AnnIndexConfig(kind="hnsw", dimension=VECTOR_DIMENSION, ef_search=64),
        by="price",
        buckets=4,
    )
    hits = routed.search(SearchRequest(query="鞋", max_price=120, top_k=5))
    assert hits
    assert all(hit.product.price <= 120 for hit in hits)
    assert routed.last_stats["partitions_pruned"] > 0
