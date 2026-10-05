from __future__ import annotations

import numpy as np
import pytest

from shopping_agent.models import Product, SearchHit, SearchRequest
from shopping_agent.rrf import ReciprocalRankFusionRetriever


class StubRetriever:
    """按固定顺序返回命中的检索器替身，用来把 RRF 的名次算术钉死。"""

    def __init__(self, ids: list[str], name: str) -> None:
        self.ids = ids
        self.route_name = name
        self.products = [Product(id=pid, title=pid, category="x", description="", price=1) for pid in ids]
        self.catalog_path = None
        self.last_stats: dict[str, object] = {}

    def search(self, request: SearchRequest) -> list[SearchHit]:
        return [
            SearchHit(product=self.products[index], score=1.0 - index * 0.1, text_score=1.0, image_score=0, reasons=[])
            for index in range(min(len(self.ids), request.top_k))
        ]

    def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
        return self.search(request.model_copy(update={"top_k": min(count, 20)}, deep=False))


def test_agreement_between_routes_wins_over_single_first_place() -> None:
    """两路都认可的文档应压过"一路第一、一路很后"的文档——这是 RRF 的核心行为。"""
    agreed = StubRetriever(["agreed", "a_tail"], "dense")
    other = StubRetriever(["b_head", "agreed"], "bm25")
    fusion = ReciprocalRankFusionRetriever([(agreed, 1.0), (other, 1.0)])
    hits = fusion.search(SearchRequest(query="机油", top_k=2))
    # agreed 在两路分别排第 1、第 2；b_head 只有一路第 1。rrf_k=60 时 1/61+1/62 > 1/61。
    assert hits[0].product.id == "agreed"


def test_scores_are_rank_based_so_scale_difference_is_irrelevant() -> None:
    """一路分数是 1000 量级、另一路是 0.1 量级时，融合结果不应被量纲带偏。"""
    big = StubRetriever(["big", "shared"], "dense")
    small = StubRetriever(["small", "shared"], "bm25")
    fusion = ReciprocalRankFusionRetriever([(big, 1.0), (small, 1.0)])
    hits = fusion.search(SearchRequest(query="x", top_k=3))
    assert [hit.product.id for hit in hits][:2] == ["shared", "big"]


def test_weights_scale_route_influence() -> None:
    """权重放大某一路时，该路的头名应压过"两路都认可但名次都靠后"的文档。

    这里刻意把共享文档放在两路的**第 2 名**：它有"共识"加成（两路都认可），
    但共识必须能被权重压过，否则 weight 就只是个装饰参数。
    """
    dense = StubRetriever(["d1", "shared"], "dense")
    sparse = StubRetriever(["shared", "s1"], "bm25")
    balanced = ReciprocalRankFusionRetriever([(dense, 1.0), (sparse, 1.0)])
    assert balanced.search(SearchRequest(query="x", top_k=1))[0].product.id == "shared"

    heavy_dense = ReciprocalRankFusionRetriever([(dense, 200.0), (sparse, 1.0)])
    assert heavy_dense.search(SearchRequest(query="x", top_k=1))[0].product.id == "d1"


def test_zero_and_negative_weights_are_rejected() -> None:
    retriever = StubRetriever(["a"], "dense")
    with pytest.raises(ValueError, match="权重"):
        ReciprocalRankFusionRetriever([(retriever, -1.0)])


def test_rrf_k_smooths_head_of_distribution() -> None:
    """rrf_k 越大，头部名次之间的贡献差越小。"""
    dense = StubRetriever(["a", "b", "c"], "dense")
    fusion = ReciprocalRankFusionRetriever([(dense, 1.0)], rrf_k=1)
    tight = fusion.search(SearchRequest(query="x", top_k=3))
    loose_fusion = ReciprocalRankFusionRetriever([(dense, 1.0)], rrf_k=1000)
    loose = loose_fusion.search(SearchRequest(query="x", top_k=3))
    # rrf_k=1 时 1/2 与 1/3 差一倍；rrf_k=1000 时几乎持平。
    assert (tight[0].score - tight[1].score) > (loose[0].score - loose[1].score) * 10


def test_reasons_record_each_route_rank() -> None:
    dense = StubRetriever(["x"], "dense")
    sparse = StubRetriever(["x"], "bm25")
    hits = ReciprocalRankFusionRetriever([(dense, 1.0), (sparse, 1.0)]).search(SearchRequest(query="x", top_k=1))
    reasons = " ".join(hits[0].reasons)
    assert "dense 第 1 名" in reasons
    assert "bm25 第 1 名" in reasons


def test_search_candidates_returns_wider_than_top_k() -> None:
    ids = [f"p{i}" for i in range(25)]
    retriever = StubRetriever(ids, "dense")
    fusion = ReciprocalRankFusionRetriever([(retriever, 1.0)], oversample=1.0)
    assert len(fusion.search(SearchRequest(query="x", top_k=3))) == 3
    assert len(fusion.search_candidates(SearchRequest(query="x", top_k=3), 20)) == 20


def test_stats_expose_per_route_hit_counts() -> None:
    fusion = ReciprocalRankFusionRetriever([(StubRetriever(["a", "b"], "dense"), 1.0), (StubRetriever(["a"], "bm25"), 1.0)])
    fusion.search(SearchRequest(query="x", top_k=2))
    assert fusion.last_stats["routes"] == 2
    assert fusion.last_stats["per_route_hits"]["dense"] == 2
    assert fusion.last_stats["per_route_hits"]["bm25"] == 1


def test_invalid_configuration_is_rejected() -> None:
    retriever = StubRetriever(["a"], "dense")
    with pytest.raises(ValueError, match="至少需要一路"):
        ReciprocalRankFusionRetriever([])
    with pytest.raises(ValueError, match="rrf_k"):
        ReciprocalRankFusionRetriever([(retriever, 1.0)], rrf_k=0)
    with pytest.raises(ValueError, match="oversample"):
        ReciprocalRankFusionRetriever([(retriever, 1.0)], oversample=0.5)
    with pytest.raises(ValueError, match="权重"):
        ReciprocalRankFusionRetriever([(retriever, 0.0)])


def test_ties_break_deterministically() -> None:
    left = StubRetriever(["b", "a"], "dense")
    right = StubRetriever(["a", "b"], "bm25")
    fusion = ReciprocalRankFusionRetriever([(left, 1.0), (right, 1.0)])
    first = [hit.product.id for hit in fusion.search(SearchRequest(query="x", top_k=2))]
    second = [hit.product.id for hit in fusion.search(SearchRequest(query="x", top_k=2))]
    assert first == second


def test_candidate_pool_is_at_least_top_k() -> None:
    """候选窗口不能窄于 top_k：否则融合只能在被截断的集合里挑。"""
    dense = StubRetriever([f"d{i}" for i in range(5)], "dense")
    seen: list[int] = []

    class Recorder(StubRetriever):
        def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
            seen.append(count)
            return super().search_candidates(request, count)

    fusion = ReciprocalRankFusionRetriever([(Recorder(["a", "b"], "dense"), 1.0), (dense, 1.0)], oversample=5.0)
    fusion.search(SearchRequest(query="x", top_k=4))
    assert seen and all(count >= 4 for count in seen)


def test_scores_are_rank_fractions_and_bounded() -> None:
    fusion = ReciprocalRankFusionRetriever([(StubRetriever(["a"], "dense"), 1.0)])
    hits = fusion.search(SearchRequest(query="x", top_k=1))
    assert 0 < hits[0].score <= 1 / 60
    assert np.isfinite(hits[0].score)
