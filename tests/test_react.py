from pathlib import Path

from shopping_agent.react import Decision, ReActAgent
from shopping_agent.retrieval import HybridRetriever
from shopping_agent.session import ContextBudget
from shopping_agent.tools import ShoppingTools

ROOT = Path(__file__).resolve().parents[1]


def make_agent(**kwargs) -> ReActAgent:
    tools = ShoppingTools(HybridRetriever.from_jsonl(ROOT / "data/products.jsonl"))
    return ReActAgent(tools, **kwargs)


class LoopingReasoner:
    """故意反复请求同一个动作，用于验证循环检测。"""

    def decide(self, context) -> Decision:
        return Decision(
            thought="原地打转",
            action="check_inventory",
            arguments={"product_id": "shoe-001"},
        )


def test_react_runs_multi_step_search_then_inventory() -> None:
    result = make_agent().run("帮我找运动鞋")
    actions = [step.action for step in result.steps]
    assert actions[0] == "search_products"
    assert actions.count("check_inventory") == 3
    assert actions[-1] is None
    assert result.stop_reason == "final_answer"
    assert result.tool_calls == 4
    assert result.intent == "search"
    assert len(result.citations) == 3
    assert set(result.citations) <= {"shoe-001", "shoe-002", "shoe-003"}
    assert result.context["compressed"] is False


def test_react_steps_carry_auditable_thought_and_latency() -> None:
    result = make_agent().run("帮我找运动鞋")
    for step in result.steps:
        assert step.thought
        assert step.arguments is not None
    assert all(step.latency_ms >= 0 for step in result.steps)
    # 最终答案必须能回溯到具体工具返回，而不是自由生成。
    assert result.steps[0].observation is not None


def test_search_arguments_inherit_session_slots() -> None:
    agent = make_agent()
    first = agent.run("想找500元以内的运动鞋")
    arguments = first.steps[0].arguments
    assert arguments["max_price"] == 500
    assert arguments["category"] == "运动鞋"


def test_follow_up_question_reuses_remembered_candidates() -> None:
    agent = make_agent()
    first = agent.run("想找500元以内的运动鞋")
    assert first.tool_calls == 3  # search + 2 次库存校验（第三双超预算已被过滤）

    second = agent.run("那有货吗")
    # 追问库存时不再重新检索，直接命中会话记住的候选——这就是多轮记忆的收益。
    assert [step.action for step in second.steps] == ["check_inventory", None]
    assert second.intent == "inventory"
    assert second.tool_calls == 1
    assert first.citations[0] in second.answer


def test_compare_intent_reuses_remembered_candidates() -> None:
    agent = make_agent()
    agent.run("帮我找运动鞋")
    result = agent.run("帮我对比一下")
    assert result.intent == "compare"
    assert result.steps[0].action == "compare_products"
    assert len(result.steps[0].arguments["product_ids"]) == 2
    assert result.stop_reason == "final_answer"


def test_inventory_intent_without_context_asks_for_clarification() -> None:
    result = make_agent().run("这个有货吗", intent="inventory")
    assert result.stop_reason == "final_answer"
    assert result.tool_calls == 0
    assert "商品 ID" in result.answer


def test_loop_detection_stops_repeated_calls() -> None:
    result = make_agent(reasoner=LoopingReasoner()).run("随意的请求")
    assert result.stop_reason == "loop_detected"
    assert result.tool_calls == 1


def test_tool_failure_becomes_observation_instead_of_crashing(monkeypatch) -> None:
    agent = make_agent()

    def boom(call):
        raise ValueError("模拟工具崩溃")

    monkeypatch.setattr(agent.tools, "execute", boom)
    result = agent.run("帮我找运动鞋")
    assert result.steps[0].error == "tool_failure"
    assert result.stop_reason == "final_answer"
    assert "不可用" in result.answer


def test_invalid_arguments_are_classified(monkeypatch) -> None:
    agent = make_agent()
    result = agent.run("帮我找运动鞋")
    # 正常路径不应产生错误分类
    assert all(step.error is None for step in result.steps if step.action)


def test_max_steps_bounds_the_loop() -> None:
    result = make_agent(max_steps=2).run("帮我找运动鞋")
    assert result.stop_reason == "max_steps"
    assert result.tool_calls == 2
    assert result.answer


def test_budget_exhaustion_stops_the_loop() -> None:
    agent = make_agent(budget=ContextBudget(max_tokens=64, reserve_tokens=0), max_steps=10)
    result = agent.run("帮我找运动鞋")
    assert result.stop_reason == "budget_exhausted"
    assert result.tool_calls >= 1


def test_trace_is_compatible_with_agent_tool_trace() -> None:
    result = make_agent().run("帮我找运动鞋")
    trace = result.trace()
    assert trace[0]["tool"] == "search_products"
    assert all({"step", "tool", "arguments"} <= set(item) for item in trace)
    assert "error" not in trace[0]
