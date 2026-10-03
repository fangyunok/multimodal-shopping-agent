from pathlib import Path

import pytest
from pydantic import ValidationError

from shopping_agent.planner import RulePlanner
from shopping_agent.retrieval import HybridRetriever
from shopping_agent.session import ContextBudget, SessionMemory, estimate_tokens
from shopping_agent.tools import ShoppingTools, ToolCall

ROOT = Path(__file__).resolve().parents[1]


def make_planner() -> RulePlanner:
    return RulePlanner(["运动鞋", "箱包", "服装"])


def test_estimate_tokens_is_heuristic_and_additive() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("运动鞋") == 3
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("运动鞋abcd") == 4


def test_slots_are_inherited_across_turns() -> None:
    planner = make_planner()
    memory = SessionMemory()
    memory.begin_turn("想找500元以内的运动鞋", planner.plan("想找500元以内的运动鞋"))
    assert memory.state.max_price == 500
    assert memory.state.category == "运动鞋"
    assert memory.state.turn_count == 1

    memory.begin_turn("那有货吗", planner.plan("那有货吗"))
    # 第二轮没有重复约束，应当继承而不是丢失。
    assert memory.state.max_price == 500
    assert memory.state.category == "运动鞋"
    assert memory.state.turn_count == 2
    assert "预算上限 ¥500" in memory.state.digest()


def test_new_explicit_value_overrides_inherited() -> None:
    planner = make_planner()
    memory = SessionMemory()
    memory.begin_turn("500元以内的运动鞋", planner.plan("500元以内的运动鞋"))
    memory.begin_turn("算了，1000元以内的箱包", planner.plan("算了，1000元以内的箱包"))
    assert memory.state.max_price == 1000
    assert memory.state.category == "箱包"


def test_exclusions_and_mentioned_products_are_tracked() -> None:
    memory = SessionMemory()
    memory.begin_turn("看看 shoe-001 和 shoe-002 这两双")
    assert "shoe-001" in memory.state.seen_product_ids
    assert "shoe-002" in memory.state.seen_product_ids

    memory.begin_turn("不要 shoe-002")
    assert "shoe-002" in memory.state.excluded_product_ids
    assert "已排除 shoe-002" in memory.state.digest()


def test_last_candidates_come_from_search_observation() -> None:
    tools = ShoppingTools(HybridRetriever.from_jsonl(ROOT / "data/products.jsonl"))
    hits = tools.execute(ToolCall(name="search_products", arguments={"query": "运动鞋", "top_k": 3}))
    memory = SessionMemory()
    memory.record_observation("search_products", hits)
    assert set(memory.state.last_candidates) == {"shoe-001", "shoe-002", "shoe-003"}
    assert "最近候选" in memory.state.digest()


def test_window_compresses_and_stays_within_budget() -> None:
    budget = ContextBudget(max_tokens=256, reserve_tokens=64, keep_recent_turns=4)
    memory = SessionMemory(budget=budget)
    planner = make_planner()
    for index in range(30):
        query = f"第{index}轮：想找一双适合通勤的白色运动鞋，预算500元以内"
        memory.begin_turn(query, planner.plan(query))

    window = memory.window()
    assert window.compressed
    assert window.dropped_turns > 0
    assert window.kept_turns <= 4
    assert window.tokens <= budget.max_tokens
    assert window.tokens < window.original_tokens
    assert window.compression_ratio < 1.0
    assert window.messages[0]["role"] == "system"
    # 槽位摘要必须始终在上下文里，否则压缩会把已确认的约束一起丢掉。
    assert "会话状态" in window.messages[0]["content"]


def test_window_compression_is_idempotent() -> None:
    budget = ContextBudget(max_tokens=256, reserve_tokens=64, keep_recent_turns=4)
    memory = SessionMemory(budget=budget)
    planner = make_planner()
    for index in range(30):
        query = f"第{index}轮：白色运动鞋 500元以内"
        memory.begin_turn(query, planner.plan(query))

    first = memory.window()
    second = memory.window()
    # 重复调用不应该继续丢历史，也不应该重复累积摘要。
    assert second.dropped_turns == 0
    assert second.tokens <= first.tokens
    assert second.summary == first.summary


def test_window_keeps_last_turn_even_under_tiny_budget() -> None:
    budget = ContextBudget(max_tokens=96, reserve_tokens=16, keep_recent_turns=2)
    memory = SessionMemory(budget=budget)
    for index in range(10):
        memory.begin_turn(f"第{index}轮补充说明")
    window = memory.window()
    assert window.kept_turns >= 1
    assert window.messages[-1]["content"]


def test_over_budget_detects_hopeless_context() -> None:
    memory = SessionMemory(budget=ContextBudget(max_tokens=64, reserve_tokens=0))
    assert not memory.over_budget()
    memory.record_answer("很长的回答" * 40)
    assert memory.over_budget()


def test_reset_clears_state_and_turns() -> None:
    planner = make_planner()
    memory = SessionMemory()
    memory.begin_turn("500元以内的运动鞋", planner.plan("500元以内的运动鞋"))
    memory.reset()
    assert memory.state.max_price is None
    assert memory.state.turn_count == 0
    assert memory.turns == []


def test_context_budget_rejects_reserve_that_leaves_no_room() -> None:
    with pytest.raises(ValidationError):
        ContextBudget(max_tokens=128, reserve_tokens=256)


def test_budget_is_a_hard_boundary_even_for_oversized_observations() -> None:
    """回归：单条观察比剩余额度还大时，必须截断内容，而不是突破预算。"""
    budget = ContextBudget(max_tokens=512, reserve_tokens=256, keep_recent_turns=4)
    memory = SessionMemory(budget=budget)
    planner = make_planner()
    for index in range(12):
        memory.begin_turn(f"第{index}轮：想找一双适合通勤的白色运动鞋，预算500元以内")
        memory.record_observation("search_products", "x" * 2000)

    window = memory.window()
    assert window.compressed
    assert window.tokens <= budget.usable_tokens
    assert window.kept_turns >= 1
    # 最近一条被截断而不是整条丢弃，当前问题仍然在上下文里。
    assert window.messages[-1]["content"]
