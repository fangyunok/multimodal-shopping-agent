import json

from shopping_agent.planner import RulePlanner
from shopping_agent.planner_evaluation import evaluate_planner


def test_planner_evaluation_reports_component_and_tag_metrics(tmp_path) -> None:
    dataset = tmp_path / "planner.jsonl"
    cases = [
        {"id": "ok", "query": "500元以内的运动鞋", "expected": {"intent": "search", "max_price": 500, "category": "运动鞋"}, "tags": ["search"]},
        {"id": "hard", "query": "跑鞋哪个好", "expected": {"intent": "compare", "max_price": None, "category": "运动鞋"}, "tags": ["compare"]},
    ]
    dataset.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases), encoding="utf-8")
    result = evaluate_planner(RulePlanner(["运动鞋"]), dataset)
    assert result["overall"]["cases"] == 2
    assert result["overall"]["joint_accuracy"] == 0.5
    assert result["overall"]["invalid_output_rate"] == 0
    assert result["overall"]["token_usage"]["total_tokens"] == 0
    assert result["by_tag"]["search"]["joint_accuracy"] == 1
    assert result["failures"][0]["id"] == "hard"
