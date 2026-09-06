from __future__ import annotations

from pathlib import Path

import numpy as np

from .encoders import MultimodalEncoder, normalize
from .models import Product, SearchHit, SearchRequest


class SemanticRetriever:
    """Exact cosine search baseline using shared Chinese CLIP embeddings."""

    def __init__(self, products: list[Product], catalog_path: Path, encoder: MultimodalEncoder):
        self.products = products
        self.catalog_path = catalog_path
        self.encoder = encoder
        text_vectors = encoder.encode_texts([product.searchable_text() for product in products])
        product_vectors = normalize(text_vectors)
        for index, product in enumerate(products):
            image_path = product.resolved_image(catalog_path)
            if image_path and image_path.exists():
                image_vector = normalize(encoder.encode_images([image_path]))[0]
                product_vectors[index] = normalize(np.asarray([text_vectors[index] + image_vector]))[0]
        self.product_vectors = normalize(product_vectors)

    @classmethod
    def from_jsonl(cls, path: str | Path, encoder: MultimodalEncoder) -> "SemanticRetriever":
        catalog_path = Path(path).resolve()
        products = [Product.model_validate_json(line) for line in catalog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(products, catalog_path, encoder)

    def search(self, request: SearchRequest) -> list[SearchHit]:
        parts: list[np.ndarray] = []
        if request.query:
            parts.append(self.encoder.encode_texts([request.query])[0])
        if request.image_path:
            parts.append(self.encoder.encode_images([Path(request.image_path)])[0])
        if parts:
            query_vector = normalize(np.asarray([sum(parts)]))[0]
            scores = self.product_vectors @ query_vector
        else:
            scores = np.asarray([product.rating / 5 for product in self.products])
        hits: list[SearchHit] = []
        for product, score in zip(self.products, scores):
            if request.max_price is not None and product.price > request.max_price:
                continue
            if request.category and request.category.lower() not in product.category.lower():
                continue
            reasons = ["CLIP 图文语义相似"] if parts else ["按商品评分排序"]
            if request.max_price is not None:
                reasons.append(f"价格不超过 ¥{request.max_price:g}")
            hits.append(SearchHit(product=product, score=round(float(score), 4), text_score=round(float(score), 4) if request.query else 0, image_score=round(float(score), 4) if request.image_path else 0, reasons=reasons))
        return sorted(hits, key=lambda hit: (hit.score, hit.product.rating), reverse=True)[: request.top_k]

