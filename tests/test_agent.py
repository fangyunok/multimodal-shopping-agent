from pathlib import Path

from PIL import Image

from shopping_agent.agent import ShoppingAgent
from shopping_agent.evaluation import evaluate
from shopping_agent.models import AgentRequest, SearchRequest
from shopping_agent.retrieval import HybridRetriever
from shopping_agent.tools import ShoppingTools

ROOT = Path(__file__).resolve().parents[1]


def make_agent() -> ShoppingAgent:
    return ShoppingAgent(ShoppingTools(HybridRetriever.from_jsonl(ROOT / "data/products.jsonl")))


def test_text_search_and_budget_filter() -> None:
    hits = make_agent().tools.search_products(SearchRequest(query="白色通勤运动鞋", max_price=500))
    assert hits[0].product.id == "shoe-001"
    assert all(hit.product.price <= 500 for hit in hits)


def test_agent_calls_inventory_and_excludes_out_of_stock() -> None:
    response = make_agent().run(AgentRequest(query="浅蓝通勤衬衫"))
    assert any(item["tool"] == "check_inventory" for item in response.tool_trace)
    assert all(hit.product.stock > 0 for hit in response.hits)


def test_agent_extracts_budget_and_category_from_query() -> None:
    response = make_agent().run(AgentRequest(query="想找500元以内的运动鞋"))
    arguments = response.tool_trace[0]["arguments"]
    assert arguments["max_price"] == 500
    assert arguments["category"] == "运动鞋"
    assert all(hit.product.price <= 500 for hit in response.hits)


def test_compare_tool() -> None:
    response = make_agent().run(AgentRequest(intent="compare", product_ids=["shoe-001", "shoe-002"]))
    assert response.tool_trace[0]["tool"] == "compare_products"
    assert "云步白色通勤运动鞋" in response.answer


def test_inventory_intent_calls_inventory_tool() -> None:
    response = make_agent().run(AgentRequest(query="这个商品有货吗", product_ids=["shoe-001"]))
    assert response.intent == "inventory"
    assert response.tool_trace[0]["tool"] == "check_inventory"
    assert "剩余 23 件" in response.answer


def test_inventory_handles_unknown_product() -> None:
    response = make_agent().run(AgentRequest(intent="inventory", product_ids=["missing"]))
    assert response.tool_trace[0]["error"] == "product_not_found"


def test_agent_routes_all_tool_calls_through_dispatcher(monkeypatch) -> None:
    agent = make_agent()
    calls = []
    original_execute = agent.tools.execute

    def recording_execute(call):
        calls.append(call.name)
        return original_execute(call)

    monkeypatch.setattr(agent.tools, "execute", recording_execute)
    agent.run(AgentRequest(query="500元以内的运动鞋"))
    assert calls[0] == "search_products"
    assert "check_inventory" in calls


def test_offline_evaluation() -> None:
    result = evaluate(make_agent(), ROOT / "data/eval.jsonl")
    assert result.tool_selection_accuracy == 1.0
    assert result.constraint_pass_rate == 1.0
    assert result.retrieval_hit_rate >= 0.75
    assert result.parameter_accuracy == 1.0
    assert result.task_success_rate == 1.0
    assert result.average_tool_calls >= 1.0


def test_image_search_baseline(tmp_path: Path) -> None:
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    query = tmp_path / "query.png"
    Image.new("RGB", (32, 32), "red").save(red)
    Image.new("RGB", (32, 32), "blue").save(blue)
    Image.new("RGB", (32, 32), (240, 10, 10)).save(query)
    catalog = tmp_path / "products.jsonl"
    catalog.write_text(
        '\n'.join([
            '{"id":"red","title":"red","category":"test","description":"item","price":1,"image_path":"red.png"}',
            '{"id":"blue","title":"blue","category":"test","description":"item","price":1,"image_path":"blue.png"}',
        ]),
        encoding="utf-8",
    )
    hits = HybridRetriever.from_jsonl(catalog).search(SearchRequest(image_path=str(query)))
    assert hits[0].product.id == "red"
