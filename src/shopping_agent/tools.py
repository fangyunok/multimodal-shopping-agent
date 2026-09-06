from __future__ import annotations

from .models import Product, SearchHit, SearchRequest
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class Retriever(Protocol):
    products: list[Product]

    def search(self, request: SearchRequest) -> list[SearchHit]: ...


class ProductIdsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_ids: list[str] = Field(min_length=2, max_length=10)


class ProductIdInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict


class ToolCall(BaseModel):
    name: str
    arguments: dict


class UnknownToolError(ValueError):
    pass


class ShoppingTools:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.by_id = {product.id: product for product in retriever.products}

    def search_products(self, request: SearchRequest) -> list[SearchHit]:
        return self.retriever.search(request)

    def compare_products(self, product_ids: list[str]) -> list[Product]:
        return [self.by_id[product_id] for product_id in product_ids if product_id in self.by_id]

    def check_inventory(self, product_id: str) -> dict[str, int | str | bool]:
        product = self.by_id[product_id]
        return {"product_id": product_id, "stock": product.stock, "available": product.stock > 0}

    def execute(self, call: ToolCall):
        if call.name == "search_products":
            request = SearchRequest.model_validate(call.arguments)
            return self.search_products(request)
        if call.name == "compare_products":
            request = ProductIdsInput.model_validate(call.arguments)
            return self.compare_products(request.product_ids)
        if call.name == "check_inventory":
            request = ProductIdInput.model_validate(call.arguments)
            if request.product_id not in self.by_id:
                raise KeyError(f"商品不存在: {request.product_id}")
            return self.check_inventory(request.product_id)
        raise UnknownToolError(f"未知工具: {call.name}")

    @staticmethod
    def definitions() -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="search_products",
                description="根据文字或图片检索商品，并应用预算、类目和返回数量约束。",
                parameters=SearchRequest.model_json_schema(),
            ),
            ToolDefinition(
                name="compare_products",
                description="根据两个或多个商品 ID 对比价格、评分和库存。",
                parameters=ProductIdsInput.model_json_schema(),
            ),
            ToolDefinition(
                name="check_inventory",
                description="查询指定商品的实时库存可用性。",
                parameters=ProductIdInput.model_json_schema(),
            ),
        ]
