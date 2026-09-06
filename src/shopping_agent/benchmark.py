from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from .models import Product, SearchRequest


class RetrievalCase(BaseModel):
    id: str
    query_type: str
    query: str = ""
    image_path: str | None = None
    relevant_ids: list[str] = Field(min_length=1)


def generate_attribute_cases(products: list[Product]) -> list[RetrievalCase]:
    cases: list[RetrievalCase] = []
    for product in products:
        values = [value for value in product.attributes.values() if value]
        if not values:
            continue
        need = "、".join(values[:3])
        query = f"想找一款{need}的{product.category}，请推荐符合这些属性的商品"
        cases.append(RetrievalCase(
            id=f"attr-{product.id}",
            query_type="attribute_query",
            query=query,
            relevant_ids=[product.id],
        ))
    return cases


def write_benchmark(cases: list[RetrievalCase], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(case.model_dump_json() for case in cases) + "\n", encoding="utf-8")


def read_benchmark(path: str | Path) -> list[RetrievalCase]:
    return [RetrievalCase.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_cases(retriever, cases: list[RetrievalCase], top_k: int = 10) -> dict:
    ranks: list[int | None] = []
    by_type: dict[str, list[int | None]] = {}
    for case in cases:
        hits = retriever.search(SearchRequest(query=case.query, image_path=case.image_path, top_k=top_k))
        rank = next((index for index, hit in enumerate(hits, start=1) if hit.product.id in case.relevant_ids), None)
        ranks.append(rank)
        by_type.setdefault(case.query_type, []).append(rank)

    def metrics(group: list[int | None]) -> dict[str, float | int]:
        total = max(len(group), 1)
        return {
            "queries": len(group),
            "recall_at_1": sum(rank == 1 for rank in group) / total,
            "recall_at_5": sum(rank is not None and rank <= 5 for rank in group) / total,
            f"recall_at_{top_k}": sum(rank is not None and rank <= top_k for rank in group) / total,
            "mrr": sum(1 / rank for rank in group if rank is not None) / total,
        }

    return {"overall": metrics(ranks), "by_query_type": {name: metrics(group) for name, group in sorted(by_type.items())}}

