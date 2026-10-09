"""库存接入契约：目录仅用于演示，HTTP 来源失败时不回退到目录库存。"""
from __future__ import annotations

import os
import math
from datetime import datetime, timezone
from typing import Protocol
from urllib.parse import quote

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from .http_json import JSONServiceClient, UpstreamError
from .models import Product


class InventoryUnavailable(RuntimeError):
    pass


class InventoryProvider(Protocol):
    source: str

    def check(self, product_id: str) -> dict: ...


class CatalogInventory:
    source = "catalog_demo"

    def __init__(self, products: list[Product]) -> None:
        self.by_id = {product.id: product for product in products}

    def check(self, product_id: str) -> dict:
        stock = self.by_id[product_id].stock
        return {"product_id": product_id, "stock": stock, "available": stock > 0,
                "source": self.source, "observed_at": None}


class StockResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str = Field(min_length=1, max_length=128, strict=True)
    stock: int = Field(ge=0, strict=True)
    observed_at: AwareDatetime


class HTTPInventory:
    source = "http"

    def __init__(self, base_url: str, api_key: str = "", timeout_seconds: float = 3,
                 max_age_seconds: float = 30) -> None:
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("库存新鲜度上限必须大于 0")
        self.client = JSONServiceClient(base_url, api_key, timeout_seconds, max_response_bytes=8192)
        self.max_age_seconds = max_age_seconds

    def check(self, product_id: str) -> dict:
        try:
            raw = self.client.request("/inventory?product_id=" + quote(product_id, safe=""))
            record = StockResponse.model_validate(raw)
            age = (datetime.now(timezone.utc) - record.observed_at).total_seconds()
            if record.product_id != product_id or age > self.max_age_seconds or age < -5:
                raise InventoryUnavailable("库存响应的商品或时间不匹配")
        except (UpstreamError, ValidationError):
            raise InventoryUnavailable("库存服务暂时不可用，当前库存未知") from None
        return {**record.model_dump(mode="json"), "available": record.stock > 0, "source": self.source}


def inventory_from_env(products: list[Product]) -> InventoryProvider:
    backend = os.getenv("INVENTORY_BACKEND", "catalog").lower()
    if backend == "catalog":
        return CatalogInventory(products)
    if backend != "http" or not os.getenv("INVENTORY_BASE_URL"):
        raise ValueError("INVENTORY_BACKEND 必须为 catalog 或已配置地址的 http")
    return HTTPInventory(
        os.environ["INVENTORY_BASE_URL"], os.getenv("INVENTORY_API_KEY", ""),
        float(os.getenv("INVENTORY_TIMEOUT_SECONDS", "3")),
        float(os.getenv("INVENTORY_MAX_AGE_SECONDS", "30")),
    )
