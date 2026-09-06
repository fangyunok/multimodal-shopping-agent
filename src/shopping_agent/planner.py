from __future__ import annotations

import re
from dataclasses import dataclass


PRICE_PATTERNS = (
    re.compile(r"(\d+(?:\.\d+)?)\s*元\s*(?:以内|以下|之内)"),
    re.compile(r"预算\s*(?:为|是|不超过|低于)?\s*(\d+(?:\.\d+)?)"),
    re.compile(r"(?:不超过|低于|小于)\s*(\d+(?:\.\d+)?)\s*元?"),
)


@dataclass(frozen=True)
class Plan:
    intent: str
    max_price: float | None
    category: str | None


class RulePlanner:
    def __init__(self, categories: list[str]):
        self.categories = sorted(set(categories), key=len, reverse=True)

    def plan(self, query: str, requested_intent: str = "auto") -> Plan:
        intent = requested_intent
        if intent == "auto":
            if any(word in query for word in ("对比", "比较", "区别")):
                intent = "compare"
            elif any(word in query for word in ("库存", "有货", "现货")):
                intent = "inventory"
            else:
                intent = "search"
        max_price = None
        for pattern in PRICE_PATTERNS:
            match = pattern.search(query)
            if match:
                max_price = float(match.group(1))
                break
        category = next((category for category in self.categories if category.lower() in query.lower()), None)
        return Plan(intent=intent, max_price=max_price, category=category)
