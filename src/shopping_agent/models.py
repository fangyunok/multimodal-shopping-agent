from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class Product(BaseModel):
    id: str
    title: str
    category: str
    description: str
    price: float = Field(ge=0)
    attributes: dict[str, str] = Field(default_factory=dict)
    image_path: str | None = None
    rating: float = Field(default=0, ge=0, le=5)
    stock: int = Field(default=0, ge=0)
    source: str | None = None
    license: str | None = None
    split: Literal["train", "validation", "test"] | None = None
    image_sha256: str | None = None

    def searchable_text(self) -> str:
        attrs = " ".join(f"{key} {value}" for key, value in self.attributes.items())
        return f"{self.title} {self.category} {self.description} {attrs}".lower()

    def resolved_image(self, catalog_path: Path) -> Path | None:
        if not self.image_path:
            return None
        path = Path(self.image_path)
        return path if path.is_absolute() else catalog_path.parent / path


class SearchRequest(BaseModel):
    model_config = {"extra": "forbid"}
    query: str = ""
    image_path: str | None = None
    max_price: float | None = Field(default=None, ge=0)
    category: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchHit(BaseModel):
    product: Product
    score: float
    text_score: float
    image_score: float
    reasons: list[str]


class AgentRequest(SearchRequest):
    intent: Literal["auto", "search", "compare", "inventory"] = "auto"
    product_ids: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    intent: str
    answer: str
    tool_trace: list[dict]
    hits: list[SearchHit] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
