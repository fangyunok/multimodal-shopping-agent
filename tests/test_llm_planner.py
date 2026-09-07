import pytest

from shopping_agent.llm_planner import LLMPlanner


def test_llm_planner_sends_schema_and_validates_response() -> None:
    captured = {}

    def transport(payload):
        captured.update(payload)
        return {
            "choices": [{"message": {"content": '{"intent":"search","max_price":500,"category":"运动鞋"}'}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
        }

    planner = LLMPlanner(["运动鞋", "箱包"], "http://localhost:8001", "qwen", transport=transport)
    plan = planner.plan("找500元以内的运动鞋")
    assert plan.max_price == 500
    assert captured["temperature"] == 0
    assert captured["response_format"]["type"] == "json_schema"
    system_prompt = captured["messages"][0]["content"]
    assert "最高预算" in system_prompt
    assert "输出intent必须与其完全相同" in system_prompt
    schema = captured["response_format"]["json_schema"]["schema"]
    assert "500元以内" in schema["properties"]["max_price"]["description"]
    assert planner.last_usage["total_tokens"] == 28


def test_llm_planner_rejects_unknown_category() -> None:
    planner = LLMPlanner(
        ["运动鞋"], "http://localhost:8001", "qwen",
        transport=lambda _: {"choices": [{"message": {"content": '{"intent":"search","max_price":null,"category":"汽车"}'}}]},
    )
    with pytest.raises(ValueError, match="未知类目"):
        planner.plan("找汽车")


def test_llm_planner_rejects_malformed_response() -> None:
    planner = LLMPlanner([], "http://localhost:8001", "qwen", transport=lambda _: {"choices": []})
    planner.last_usage = {"total_tokens": 99}
    with pytest.raises(ValueError, match="返回格式无效"):
        planner.plan("搜索")
    assert planner.last_usage["total_tokens"] == 0


def test_llm_planner_must_obey_explicit_intent() -> None:
    planner = LLMPlanner(
        [], "http://localhost:8001", "qwen",
        transport=lambda _: {"choices": [{"message": {"content": '{"intent":"search","max_price":null,"category":null}'}}]},
    )
    with pytest.raises(ValueError, match="显式 intent"):
        planner.plan("查询库存", "inventory")
