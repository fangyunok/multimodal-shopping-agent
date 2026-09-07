from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field


PRICE_PATTERNS = (
    re.compile(r"(\d+(?:\.\d+)?)\s*元\s*(?:以内|以下|之内)"),
    re.compile(r"预算\s*(?:为|是|不超过|低于)?\s*(\d+(?:\.\d+)?)"),
    re.compile(r"(?:不超过|低于|小于)\s*(\d+(?:\.\d+)?)\s*元?"),
)


class Plan(BaseModel):
    model_config = {"extra": "forbid"}
    intent: Literal["search", "compare", "inventory"] = Field(
        description="用户目标：搜索商品、比较多个商品，或查询库存"
    )
    max_price: float | None = Field(
        default=None,
        ge=0,
        description="用户明确表达的最高预算数值；例如'500元以内'填500，未表达上限才填null",
    )
    category: str | None = Field(
        description="从系统给出的可用类目中选择；可根据同义词映射，无法确定时填null"
    )


class Planner(Protocol):
    def plan(self, query: str, requested_intent: str = "auto") -> Plan: ...


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
