from pathlib import Path

from shopping_agent.models import Product
from shopping_agent.retrieval import HybridRetriever
from shopping_agent.retrieval_evaluation import evaluate_retrieval


def test_text_retrieval_metrics(tmp_path: Path) -> None:
    products = [
        Product(id="shoe", title="白色运动鞋", category="鞋", description="通勤", price=1),
        Product(id="bag", title="黑色托特包", category="包", description="旅行", price=1),
    ]
    retriever = HybridRetriever(products, tmp_path / "products.jsonl")
    result = evaluate_retrieval(retriever, products, mode="text", top_k=2)
    assert result["queries"] == 2
    assert result["recall_at_1"] == 1.0
    assert result["mrr"] == 1.0

