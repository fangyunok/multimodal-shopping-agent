from __future__ import annotations

from .models import Product, SearchHit, SearchRequest
from typing import Protocol


class Retriever(Protocol):
    products: list[Product]

    def search(self, request: SearchRequest) -> list[SearchHit]: ...


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
