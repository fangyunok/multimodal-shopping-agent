from __future__ import annotations

import json
import math
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .planner import Plan, Planner


class PlannerCase(BaseModel):
    id: str
    query: str
    requested_intent: str = "auto"
    expected: Plan
    tags: list[str] = Field(default_factory=list)


def load_planner_cases(path: str | Path) -> list[PlannerCase]:
    return [PlannerCase.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_planner(planner: Planner, dataset_path: str | Path) -> dict:
    cases = load_planner_cases(dataset_path)
    rows = []
    latencies = []
    failures = []
    for case in cases:
        started = time.perf_counter()
        error = None
        try:
            actual = planner.plan(case.query, case.requested_intent)
        except Exception as caught:
            actual = None
            error = type(caught).__name__
        latencies.append((time.perf_counter() - started) * 1000)
        checks = {
            "intent": actual is not None and actual.intent == case.expected.intent,
            "max_price": actual is not None and actual.max_price == case.expected.max_price,
            "category": actual is not None and actual.category == case.expected.category,
        }
        row = {"case": case, "checks": checks, "valid": actual is not None}
        rows.append(row)
        if not all(checks.values()):
            failures.append({
                "id": case.id,
                "tags": case.tags,
                "expected": case.expected.model_dump(),
                "actual": actual.model_dump() if actual else None,
                "error": error,
            })

    def summarize(group: list[dict]) -> dict[str, float | int]:
        total = max(len(group), 1)
        return {
            "cases": len(group),
            "intent_accuracy": sum(row["checks"]["intent"] for row in group) / total,
            "max_price_accuracy": sum(row["checks"]["max_price"] for row in group) / total,
            "category_accuracy": sum(row["checks"]["category"] for row in group) / total,
            "joint_accuracy": sum(all(row["checks"].values()) for row in group) / total,
            "invalid_output_rate": sum(not row["valid"] for row in group) / total,
        }

    ordered = sorted(latencies)
    percentile = lambda fraction: ordered[max(0, min(math.ceil(fraction * len(ordered)) - 1, len(ordered) - 1))] if ordered else 0.0
    overall = summarize(rows)
    overall.update({"latency_p50_ms": percentile(0.5), "latency_p95_ms": percentile(0.95)})
    tags = sorted({tag for case in cases for tag in case.tags})
    by_tag = {tag: summarize([row for row in rows if tag in row["case"].tags]) for tag in tags}
    return {"overall": overall, "by_tag": by_tag, "failures": failures}
