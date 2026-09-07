from __future__ import annotations

from pydantic import ValidationError

from .models import AgentRequest, AgentResponse, SearchRequest
from .planner import RulePlanner
from .tools import ShoppingTools, ToolCall


class ShoppingAgent:
    """Deterministic baseline whose trace can be evaluated and later replaced by an LLM planner."""

    def __init__(self, tools: ShoppingTools):
        self.tools = tools
        self.planner = RulePlanner([product.category for product in tools.retriever.products])

    def run(self, request: AgentRequest) -> AgentResponse:
        plan = self.planner.plan(request.query, request.intent)
        intent = "compare" if len(request.product_ids) >= 2 else plan.intent
        trace: list[dict] = []
        if intent == "inventory":
            if not request.product_ids:
                return AgentResponse(intent=intent, answer="请提供需要查询库存的商品 ID。", tool_trace=trace)
            product_id = request.product_ids[0]
            if product_id not in self.tools.by_id:
                trace.append({"tool": "check_inventory", "arguments": {"product_id": product_id}, "error": "product_not_found"})
                return AgentResponse(intent=intent, answer=f"没有找到商品 {product_id}。", tool_trace=trace)
            inventory = self.tools.execute(ToolCall(name="check_inventory", arguments={"product_id": product_id}))
            trace.append({"tool": "check_inventory", "arguments": {"product_id": product_id}, "result": inventory})
            status = f"有货，剩余 {inventory['stock']} 件" if inventory["available"] else "暂时无货"
            return AgentResponse(intent=intent, answer=f"商品 [{product_id}] {status}。", tool_trace=trace, citations=[product_id])
        if intent == "compare":
            arguments = {"product_ids": request.product_ids}
            try:
                products = self.tools.execute(ToolCall(name="compare_products", arguments=arguments))
            except ValidationError as error:
                trace.append({"tool": "compare_products", "arguments": arguments, "error": "invalid_arguments"})
                return AgentResponse(intent=intent, answer="请至少提供两个有效商品 ID 进行对比。", tool_trace=trace)
            trace.append({"tool": "compare_products", "arguments": {"product_ids": request.product_ids}, "result_count": len(products)})
            if len(products) < 2:
                return AgentResponse(intent=intent, answer="请至少提供两个有效商品 ID 进行对比。", tool_trace=trace)
            lines = [f"{p.title} [{p.id}]：¥{p.price:g}，评分 {p.rating:.1f}，库存 {p.stock}" for p in products]
            best = max(products, key=lambda p: (p.rating, -p.price))
            answer = "；".join(lines) + f"。综合评分与价格，优先考虑 {best.title}。"
            return AgentResponse(intent=intent, answer=answer, tool_trace=trace, citations=[p.id for p in products])

        values = request.model_dump(include={"query", "image_path", "max_price", "category", "top_k"})
        values["max_price"] = request.max_price if request.max_price is not None else plan.max_price
        values["category"] = request.category or plan.category
        search_request = SearchRequest(**values)
        hits = self.tools.execute(ToolCall(name="search_products", arguments=search_request.model_dump()))
        trace.append({"tool": "search_products", "arguments": search_request.model_dump(), "result_count": len(hits)})
        if not hits:
            return AgentResponse(intent=intent, answer="没有找到满足条件的商品，请放宽预算或更换描述。", tool_trace=trace)
        available = []
        for hit in hits:
            inventory = self.tools.execute(ToolCall(name="check_inventory", arguments={"product_id": hit.product.id}))
            trace.append({"tool": "check_inventory", "arguments": {"product_id": hit.product.id}, "result": inventory})
            if inventory["available"]:
                available.append(hit)
        if not available:
            return AgentResponse(intent=intent, answer="检索到了相关商品，但目前都没有库存。", tool_trace=trace, hits=hits)
        top = available[0]
        reason = "、".join(top.reasons) or "评分较高"
        answer = f"推荐 {top.product.title} [{top.product.id}]（¥{top.product.price:g}，评分 {top.product.rating:.1f}），因为{reason}。"
        return AgentResponse(intent=intent, answer=answer, tool_trace=trace, hits=available, citations=[top.product.id])
