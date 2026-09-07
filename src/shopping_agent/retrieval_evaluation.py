from __future__ import annotations

from pathlib import Path
import math

from .models import Product, SearchRequest


def evaluate_retrieval(retriever, products: list[Product], mode: str = "text", top_k: int = 10) -> dict[str, float | int | str]:
    if mode not in {"text", "image"}:
        raise ValueError("mode 仅支持 text 或 image")
    ranks: list[int | None] = []
    for product in products:
        if mode == "image" and not product.image_path:
            continue
        image_path = None
        query = product.title if mode == "text" else ""
        if mode == "image":
            resolved = product.resolved_image(retriever.catalog_path)
            if not resolved or not resolved.exists():
                continue
            image_path = str(resolved)
        hits = retriever.search(SearchRequest(query=query, image_path=image_path, top_k=top_k))
        rank = next((index for index, hit in enumerate(hits, start=1) if hit.product.id == product.id), None)
        ranks.append(rank)
    total = max(len(ranks), 1)
    reciprocal_rank = sum(1 / rank for rank in ranks if rank is not None) / total
    return {
        "mode": mode,
        "queries": len(ranks),
        "recall_at_1": sum(rank == 1 for rank in ranks) / total,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / total,
        f"recall_at_{top_k}": sum(rank is not None and rank <= top_k for rank in ranks) / total,
        "mrr": reciprocal_rank,
        f"ndcg_at_{top_k}": sum(1 / math.log2(rank + 1) for rank in ranks if rank is not None) / total,
    }
