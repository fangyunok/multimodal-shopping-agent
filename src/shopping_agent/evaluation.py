from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .agent import ShoppingAgent
from .models import AgentRequest


@dataclass
class EvaluationResult:
    cases: int
    retrieval_cases: int
    retrieval_hit_rate: float
    tool_selection_accuracy: float
    constraint_pass_rate: float
    parameter_accuracy: float
    task_success_rate: float
    average_tool_calls: float

    def as_dict(self) -> dict[str, int | float]:
        return self.__dict__


def evaluate(agent: ShoppingAgent, dataset_path: str | Path) -> EvaluationResult:
    cases = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    retrieval_hits = retrieval_cases = tool_hits = constraint_hits = parameter_hits = task_hits = tool_call_total = 0
    for case in cases:
        response = agent.run(AgentRequest.model_validate(case["request"]))
        tool_call_total += len(response.tool_trace)
        returned_ids = {hit.product.id for hit in response.hits}
        relevant_ids = case.get("relevant_ids", [])
        if relevant_ids:
            retrieval_cases += 1
            retrieval_hits += bool(returned_ids.intersection(relevant_ids))
        tool_hits += bool(response.tool_trace) and response.tool_trace[0]["tool"] == case["expected_tool"]
        actual_arguments = response.tool_trace[0].get("arguments", {}) if response.tool_trace else {}
        expected_arguments = case.get("expected_arguments", {})
        parameter_hits += all(actual_arguments.get(key) == value for key, value in expected_arguments.items())
        answer_ok = all(fragment in response.answer for fragment in case.get("expected_answer_contains", []))
        expected_returned = set(case.get("expected_returned_ids", []))
        returned_ok = not expected_returned or expected_returned.issubset(returned_ids)
        task_hits += answer_ok and returned_ok
        max_price = case["request"].get("max_price")
        constraint_hits += max_price is None or all(hit.product.price <= max_price for hit in response.hits)
    total = max(len(cases), 1)
    return EvaluationResult(
        len(cases), retrieval_cases, retrieval_hits / max(retrieval_cases, 1),
        tool_hits / total, constraint_hits / total, parameter_hits / total,
        task_hits / total, tool_call_total / total,
    )
