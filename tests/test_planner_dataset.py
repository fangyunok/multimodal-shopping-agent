from pathlib import Path

from shopping_agent.planner_evaluation import load_planner_cases

ROOT = Path(__file__).resolve().parents[1]


def test_planner_datasets_are_locked_disjoint_and_well_formed() -> None:
    dev = load_planner_cases(ROOT / "data/planner_dev.jsonl")
    test = load_planner_cases(ROOT / "data/planner_test.jsonl")
    assert len(dev) == 20
    assert len(test) == 100
    assert len({case.id for case in dev + test}) == 120
    assert not {case.query for case in dev}.intersection(case.query for case in test)
    assert all(case.expected.category in {None, "运动鞋", "箱包", "服装"} for case in dev + test)
    assert all(sum(tag in {"search", "compare", "inventory"} for tag in case.tags) == 1 for case in test)


def test_locked_test_set_has_required_challenge_coverage() -> None:
    cases = load_planner_cases(ROOT / "data/planner_test.jsonl")
    counts = {tag: sum(tag in case.tags for case in cases) for tag in {
        "search", "compare", "inventory", "budget", "synonym",
        "explicit_intent", "underspecified", "adversarial",
    }}
    assert counts == {
        "search": 56,
        "compare": 22,
        "inventory": 22,
        "budget": 26,
        "synonym": 50,
        "explicit_intent": 10,
        "underspecified": 13,
        "adversarial": 5,
    }
