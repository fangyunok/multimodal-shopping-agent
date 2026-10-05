from __future__ import annotations

import numpy as np
import pytest

from shopping_agent.models import Product, SearchHit, SearchRequest
from shopping_agent.rerank import (
    ExactVectorReranker,
    RerankRetriever,
    product_document,
)


class StubBase:
    """按给定顺序返回候选的粗排替身，记录自己被要的候选数。"""

    route_name = "base"

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.products = [Product(id=pid, title=pid, category="x", description="", price=1) for pid in ids]
        self.catalog_path = None
        self.last_stats: dict[str, object] = {}
        self.requested: list[int] = []

    def search(self, request: SearchRequest) -> list[SearchHit]:
        return self.search_candidates(request, int(request.top_k))

    def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
        self.requested.append(count)
        return [
            SearchHit(
                product=self.products[index],
                score=1.0 - index * 0.01,
                text_score=1.0,
                image_score=0,
                reasons=["粗排"],
            )
            for index in range(min(len(self.ids), count))
        ]


class StubReranker:
    """按 product_id 查表的重排器替身；分数表外的商品给极低分。"""

    name = "stub"

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        self.last_stats: dict[str, object] = {}

    def score(self, query: str, candidates):
        return np.asarray(
            [self.table.get(product.id, float("-inf")) for product in candidates],
            dtype=np.float32,
        )


class PlainBase:
    """只有 ``search``、没有候选级入口的后端——用来验证降级路径。"""

    def __init__(self, ids: list[str]) -> None:
        self.products = [Product(id=pid, title=pid, category="x", description="", price=1) for pid in ids]
        self.catalog_path = None
        self.last_stats: dict[str, object] = {}
        self.asked: list[int] = []

    def search(self, request: SearchRequest) -> list[SearchHit]:
        self.asked.append(int(request.top_k))
        return [
            SearchHit(product=self.products[index], score=1.0 - index * 0.01, text_score=1.0, image_score=0, reasons=[])
            for index in range(min(len(self.products), request.top_k))
        ]


def test_product_document_puts_title_first() -> None:
    document = product_document(
        Product(
            id="p",
            title="5W-30 机油",
            category="AUTO",
            description=" engine oil ",
            price=1,
            attributes={"容量": "5升"},
        )
    )
    assert document.startswith("5W-30 机油。")
    assert "AUTO" in document
    assert "5升" in document


def test_exact_reranker_returns_cosine_and_normalizes_vectors() -> None:
    class Encoder:
        def encode_texts(self, texts):
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    products = [
        Product(id="aligned", title="a", category="x", description="", price=1),
        Product(id="orthogonal", title="b", category="x", description="", price=1),
    ]
    # 第 2 条刻意给未归一化向量：若构造时不归一化，内积就不再等于余弦。
    vectors = np.asarray([[3.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    reranker = ExactVectorReranker(Encoder(), vectors)
    reranker.bind_products(products)
    scores = reranker.score("query", products)
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)


def test_exact_reranker_rejects_non_matrix_vectors() -> None:
    class Encoder:
        def encode_texts(self, texts):
            return np.zeros((1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="二维"):
        ExactVectorReranker(Encoder(), np.zeros(4, dtype=np.float32))


def test_exact_reranker_handles_empty_candidate_list() -> None:
    class Encoder:
        def encode_texts(self, texts):
            raise AssertionError("空候选时不应触发编码")

    reranker = ExactVectorReranker(Encoder(), np.zeros((2, 3), dtype=np.float32))
    assert reranker.score("q", []).shape == (0,)


def test_rerank_reorders_by_new_scores() -> None:
    base = StubBase(["a", "b", "c"])
    retriever = RerankRetriever(base, StubReranker({"a": 0.1, "b": 0.9, "c": 0.5}), candidates=3, keep=3)
    hits = retriever.search(SearchRequest(query="x", top_k=3))
    assert [hit.product.id for hit in hits] == ["b", "c", "a"]


def test_rerank_requests_wider_candidate_set_than_top_k() -> None:
    """两阶段最常见的错误：粗排就截到 top_k，重排只能在原顺序里换先后。"""
    base = StubBase([f"p{i}" for i in range(20)])
    RerankRetriever(base, StubReranker({"p0": 9.0}), candidates=15).search(SearchRequest(query="x", top_k=3))
    assert base.requested and base.requested[0] >= 15


def test_rerank_falls_back_when_base_lacks_candidate_entry() -> None:
    base = PlainBase([f"p{i}" for i in range(20)])
    retriever = RerankRetriever(base, StubReranker({"p0": 9.0}), candidates=15)
    hits = retriever.search(SearchRequest(query="x", top_k=3))
    assert hits[0].product.id == "p0"
    # 降级时仍受 SearchRequest.top_k 上限约束，不能靠改写绕过契约。
    assert base.asked == [15] or base.asked == [20]


def test_rerank_is_skipped_on_size_mismatch() -> None:
    class ShortReranker:
        name = "short"

        def score(self, query, candidates):
            self.last_stats: dict[str, object] = {}
            return np.zeros(2, dtype=np.float32)

    base = StubBase(["a", "b", "c"])
    retriever = RerankRetriever(base, ShortReranker(), candidates=3)
    hits = retriever.search(SearchRequest(query="x", top_k=3))
    # 分数与候选数对不上时宁可粗排原序返回，也不能用错位的分数排序。
    assert [hit.product.id for hit in hits] == ["a", "b", "c"]
    assert retriever.last_stats["rerank"] == "skipped_size_mismatch"


def test_rerank_off_when_no_reranker_configured() -> None:
    base = StubBase(["a", "b"])
    retriever = RerankRetriever(base, None, candidates=2)
    hits = retriever.search(SearchRequest(query="x", top_k=2))
    assert [hit.product.id for hit in hits] == ["a", "b"]
    assert retriever.last_stats["rerank"] == "off"


def test_rerank_preserves_original_reasons_and_appends_its_own() -> None:
    base = StubBase(["a", "b"])
    retriever = RerankRetriever(base, StubReranker({"a": 0.1, "b": 0.9}), candidates=2)
    hits = retriever.search(SearchRequest(query="x", top_k=2))
    assert "粗排" in hits[0].reasons
    assert "stub 重排" in hits[0].reasons


def test_rerank_stats_report_candidate_count() -> None:
    base = StubBase(["a", "b", "c"])
    retriever = RerankRetriever(base, StubReranker({"a": 1.0, "b": 2.0, "c": 3.0}), candidates=3)
    retriever.search(SearchRequest(query="x", top_k=2))
    assert retriever.last_stats["reranked"] == 3
    assert retriever.last_stats["rerank"] == "stub"


def test_invalid_configuration_is_rejected() -> None:
    base = StubBase(["a"])
    with pytest.raises(ValueError, match="candidates"):
        RerankRetriever(base, None, candidates=0)
    with pytest.raises(ValueError, match="keep"):
        RerankRetriever(base, None, candidates=10, keep=0)


def test_describe_exposes_reranker_and_base() -> None:
    base = StubBase(["a"])
    described = RerankRetriever(base, StubReranker({"a": 1.0}), candidates=5).describe()
    assert described["reranker"] == "stub"
    assert described["candidates"] == 5


def test_cli_exact_vector_reranker_requires_vectors_file() -> None:
    """没有 vectors.npy 时必须给出可执行的报错，而不是静默退回粗排。"""
    from argparse import Namespace

    from shopping_agent.cli import build_reranker

    args = Namespace(reranker="exact-vector", index_dir=None)
    with pytest.raises(SystemExit) as error:
        build_reranker(args, None, [])
    message = str(error.value)
    assert "vectors.npy" in message
    assert "--index-dir" in message


def test_cli_exact_vector_reranker_binds_products(tmp_path) -> None:
    from argparse import Namespace

    from shopping_agent.cli import build_reranker
    from shopping_agent.indexing import VECTORS_FILENAME

    import numpy as np

    np.save(tmp_path / VECTORS_FILENAME, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    args = Namespace(reranker="exact-vector", index_dir=str(tmp_path))
    products = [
        Product(id="a", title="a", category="x", description="", price=1),
        Product(id="b", title="b", category="x", description="", price=1),
    ]
    reranker = build_reranker(args, None, products)
    assert reranker.name == "exact-vector"
    assert reranker.index == {"a": 0, "b": 1}
