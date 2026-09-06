from shopping_agent.planner import RulePlanner


def test_rule_planner_extracts_intent_budget_and_category() -> None:
    planner = RulePlanner(["运动鞋", "箱包"])
    plan = planner.plan("比较两双500元以内的运动鞋")
    assert plan.intent == "compare"
    assert plan.max_price == 500
    assert plan.category == "运动鞋"


def test_explicit_intent_is_preserved() -> None:
    assert RulePlanner([]).plan("比较商品", "search").intent == "search"


def test_inventory_intent() -> None:
    assert RulePlanner([]).plan("这个商品还有库存吗").intent == "inventory"
