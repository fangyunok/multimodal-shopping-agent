from pathlib import Path

from shopping_agent.benchmark import RetrievalCase, evaluate_cases, generate_attribute_cases, read_benchmark, write_benchmark
from shopping_agent.models import Product
from shopping_agent.retrieval import HybridRetriever


def test_generate_cases_do_not_copy_title() -> None:
    product = Product(
        id="one", title="品牌专属名称", category="运动鞋", description="", price=1,
        attributes={"颜色": "白色", "场景": "通勤"}, split="test",
    )
    case = generate_attribute_cases([product])[0]
    assert product.title not in case.query
    assert case.relevant_ids == ["one"]


def test_benchmark_round_trip_and_metrics(tmp_path: Path) -> None:
    products = [
        Product(id="shoe", title="云步", category="运动鞋", description="白色通勤", price=1, attributes={"颜色": "白色"}),
        Product(id="bag", title="城市包", category="箱包", description="黑色旅行", price=1, attributes={"颜色": "黑色"}),
    ]
    cases = [RetrievalCase(id="q1", query_type="attribute_query", query="白色运动鞋", relevant_ids=["shoe"])]
    path = tmp_path / "benchmark.jsonl"
    write_benchmark(cases, path)
    result = evaluate_cases(HybridRetriever(products, tmp_path / "catalog.jsonl"), read_benchmark(path))
    assert result["overall"]["recall_at_1"] == 1.0
    assert result["by_query_type"]["attribute_query"]["mrr"] == 1.0

