from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable

from .planner import Plan

Transport = Callable[[dict], dict]


class LLMPlanner:
    """OpenAI-compatible structured planner; no provider SDK required."""

    def __init__(
        self,
        categories: list[str],
        endpoint: str,
        model: str,
        api_key: str = "",
        timeout_seconds: float = 30,
        transport: Transport | None = None,
    ):
        self.categories = sorted(set(categories), key=len, reverse=True)
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.transport = transport or self._request

    def _request(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def plan(self, query: str, requested_intent: str = "auto") -> Plan:
        schema = Plan.model_json_schema()
        prompt = (
            "你是电商购物Agent的规划器。只输出一个JSON对象，不要Markdown。"
            "intent只能是search、compare或inventory。提取明确预算和商品类目；未提及时填null。"
            f"可用类目：{self.categories}。用户显式intent为{requested_intent}，非auto时必须遵守。"
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            "response_format": {"type": "json_schema", "json_schema": {"name": "shopping_plan", "schema": schema}},
        }
        response = self.transport(payload)
        try:
            content = response["choices"][0]["message"]["content"]
            plan = Plan.model_validate_json(content)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("LLM Planner 返回格式无效") from error
        if plan.category is not None and plan.category not in self.categories:
            raise ValueError(f"LLM Planner 返回未知类目: {plan.category}")
        if requested_intent != "auto" and plan.intent != requested_intent:
            raise ValueError("LLM Planner 未遵守显式 intent")
        return plan
