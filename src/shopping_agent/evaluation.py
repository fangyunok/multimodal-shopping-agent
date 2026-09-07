from __future__ import annotations

import json
import math
import time
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
    invalid_tool_call_rate: float
    latency_p50_ms: float
    latency_p95_ms: float

    def as_dict(self) -> dict[str, int | float]:
        return self.__dict__


def evaluate(agent: ShoppingAgent, dataset_path: str | Path) -> EvaluationResult:
    cases = [json.loads(line) for line in Path(dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    retrieval_hits = retrieval_cases = tool_hits = constraint_hits = parameter_hits = task_hits = tool_call_total = 0
    invalid_tool_calls = 0
    latencies_ms: list[float] = []
    valid_tools = {definition.name for definition in agent.tools.definitions()}
    for case in cases:
        started = time.perf_counter()
        response = agent.run(AgentRequest.model_validate(case["request"]))
        latencies_ms.append((time.perf_counter() - started) * 1000)
        tool_call_total += len(response.tool_trace)
        invalid_tool_calls += sum(
            trace.get("tool") not in valid_tools or trace.get("error") == "invalid_arguments"
            for trace in response.tool_trace
        )
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
    ordered_latency = sorted(latencies_ms)

    def percentile(fraction: float) -> float:
        if not ordered_latency:
            return 0.0
        index = min(math.ceil(fraction * len(ordered_latency)) - 1, len(ordered_latency) - 1)
        return ordered_latency[max(index, 0)]

    return EvaluationResult(
        len(cases), retrieval_cases, retrieval_hits / max(retrieval_cases, 1),
        tool_hits / total, constraint_hits / total, parameter_hits / total,
        task_hits / total, tool_call_total / total,
        invalid_tool_calls / max(tool_call_total, 1), percentile(0.50), percentile(0.95),
    )
