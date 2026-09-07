import pytest

from shopping_agent.llm_planner import LLMPlanner


def test_llm_planner_sends_schema_and_validates_response() -> None:
    captured = {}

    def transport(payload):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"intent":"search","max_price":500,"category":"运动鞋"}'}}]}

    planner = LLMPlanner(["运动鞋", "箱包"], "http://localhost:8001", "qwen", transport=transport)
    plan = planner.plan("找500元以内的运动鞋")
    assert plan.max_price == 500
    assert captured["temperature"] == 0
    assert captured["response_format"]["type"] == "json_schema"


def test_llm_planner_rejects_unknown_category() -> None:
    planner = LLMPlanner(
        ["运动鞋"], "http://localhost:8001", "qwen",
        transport=lambda _: {"choices": [{"message": {"content": '{"intent":"search","max_price":null,"category":"汽车"}'}}]},
    )
    with pytest.raises(ValueError, match="未知类目"):
        planner.plan("找汽车")


def test_llm_planner_rejects_malformed_response() -> None:
    planner = LLMPlanner([], "http://localhost:8001", "qwen", transport=lambda _: {"choices": []})
    with pytest.raises(ValueError, match="返回格式无效"):
        planner.plan("搜索")


def test_llm_planner_must_obey_explicit_intent() -> None:
    planner = LLMPlanner(
        [], "http://localhost:8001", "qwen",
        transport=lambda _: {"choices": [{"message": {"content": '{"intent":"search","max_price":null,"category":null}'}}]},
    )
    with pytest.raises(ValueError, match="显式 intent"):
        planner.plan("查询库存", "inventory")
