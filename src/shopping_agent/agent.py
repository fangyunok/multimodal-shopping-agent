from __future__ import annotations

from .models import AgentRequest, AgentResponse, SearchRequest
from .tools import ShoppingTools


class ShoppingAgent:
    """Deterministic baseline whose trace can be evaluated and later replaced by an LLM planner."""

    def __init__(self, tools: ShoppingTools):
        self.tools = tools

    def run(self, request: AgentRequest) -> AgentResponse:
        intent = request.intent
        if intent == "auto":
            intent = "compare" if len(request.product_ids) >= 2 or "对比" in request.query else "search"
        trace: list[dict] = []
        if intent == "compare":
            products = self.tools.compare_products(request.product_ids)
            trace.append({"tool": "compare_products", "arguments": {"product_ids": request.product_ids}, "result_count": len(products)})
            if len(products) < 2:
                return AgentResponse(intent=intent, answer="请至少提供两个有效商品 ID 进行对比。", tool_trace=trace)
            lines = [f"{p.title}：¥{p.price:g}，评分 {p.rating:.1f}，库存 {p.stock}" for p in products]
            best = max(products, key=lambda p: (p.rating, -p.price))
            answer = "；".join(lines) + f"。综合评分与价格，优先考虑 {best.title}。"
            return AgentResponse(intent=intent, answer=answer, tool_trace=trace)

        search_request = SearchRequest(**request.model_dump(include={"query", "image_path", "max_price", "category", "top_k"}))
        hits = self.tools.search_products(search_request)
        trace.append({"tool": "search_products", "arguments": search_request.model_dump(), "result_count": len(hits)})
        if not hits:
            return AgentResponse(intent=intent, answer="没有找到满足条件的商品，请放宽预算或更换描述。", tool_trace=trace)
        available = []
        for hit in hits:
            inventory = self.tools.check_inventory(hit.product.id)
            trace.append({"tool": "check_inventory", "arguments": {"product_id": hit.product.id}, "result": inventory})
            if inventory["available"]:
                available.append(hit)
        if not available:
            return AgentResponse(intent=intent, answer="检索到了相关商品，但目前都没有库存。", tool_trace=trace, hits=hits)
        top = available[0]
        reason = "、".join(top.reasons) or "评分较高"
        answer = f"推荐 {top.product.title}（¥{top.product.price:g}，评分 {top.product.rating:.1f}），因为{reason}。"
        return AgentResponse(intent=intent, answer=answer, tool_trace=trace, hits=available)

