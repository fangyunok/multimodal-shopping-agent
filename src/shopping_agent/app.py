from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI

from .agent import ShoppingAgent
from .models import AgentRequest, AgentResponse, SearchHit, SearchRequest
from .retrieval import HybridRetriever
from .tools import ShoppingTools

ROOT = Path(__file__).resolve().parents[2]


@lru_cache
def get_tools() -> ShoppingTools:
    catalog = Path(os.getenv("CATALOG_PATH", ROOT / "data" / "products.jsonl"))
    return ShoppingTools(HybridRetriever.from_jsonl(catalog))


app = FastAPI(title="Multimodal Shopping Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest) -> list[SearchHit]:
    return get_tools().search_products(request)


@app.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest) -> AgentResponse:
    return ShoppingAgent(get_tools()).run(request)

