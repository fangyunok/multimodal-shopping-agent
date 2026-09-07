from pathlib import Path

from shopping_agent.fusion_retrieval import ScoreFusionRetriever
from shopping_agent.models import Product, SearchHit, SearchRequest


class FixedRetriever:
    def __init__(self, products, scores, catalog_path):
        self.products = products
        self.scores = scores
        self.catalog_path = catalog_path

    def search(self, request):
        hits = [SearchHit(product=p, score=s, text_score=s, image_score=0, reasons=[]) for p, s in zip(self.products, self.scores)]
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[: request.top_k]


def test_score_fusion_combines_exact_and_semantic_signals(tmp_path: Path) -> None:
    products = [
        Product(id="exact", title="exact", category="x", description="", price=1),
        Product(id="semantic", title="semantic", category="x", description="", price=1),
    ]
    lexical = FixedRetriever(products, [1.0, 0.0], tmp_path / "products.jsonl")
    semantic = FixedRetriever(products, [0.1, 1.0], tmp_path / "products.jsonl")
    hits = ScoreFusionRetriever(lexical, semantic, lexical_weight=0.7).search(SearchRequest(query="exact"))
    assert hits[0].product.id == "exact"
    assert "精确词项匹配" in hits[0].reasons


def test_image_only_uses_semantic_retriever(tmp_path: Path) -> None:
    products = [Product(id="p", title="p", category="x", description="", price=1)]
    lexical = FixedRetriever(products, [0.0], tmp_path / "products.jsonl")
    semantic = FixedRetriever(products, [0.8], tmp_path / "products.jsonl")
    retriever = ScoreFusionRetriever(lexical, semantic)
    assert retriever.search(SearchRequest(image_path="query.jpg"))[0].score == 0.8


def test_invalid_lexical_weight_is_rejected(tmp_path: Path) -> None:
    products = [Product(id="p", title="p", category="x", description="", price=1)]
    backend = FixedRetriever(products, [1.0], tmp_path / "products.jsonl")
    try:
        ScoreFusionRetriever(backend, backend, lexical_weight=-0.1)
    except ValueError as error:
        assert "lexical_weight" in str(error)
    else:
        raise AssertionError("invalid lexical weight must be rejected")


def test_equal_scores_have_deterministic_product_id_tiebreak(tmp_path: Path) -> None:
    products = [
        Product(id="b", title="b", category="x", description="", price=1, rating=4),
        Product(id="a", title="a", category="x", description="", price=1, rating=4),
    ]
    backend = FixedRetriever(products, [1.0, 1.0], tmp_path / "products.jsonl")
    hits = ScoreFusionRetriever(backend, backend).search(SearchRequest(query="same"))
    assert [hit.product.id for hit in hits] == ["a", "b"]
