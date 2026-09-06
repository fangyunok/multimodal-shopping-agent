from __future__ import annotations

from pathlib import Path

import numpy as np

from .encoders import MultimodalEncoder, normalize
from .indexing import encode_products, load_catalog, load_index
from .models import Product, SearchHit, SearchRequest


class SemanticRetriever:
    """Exact cosine search baseline using shared Chinese CLIP embeddings."""

    def __init__(self, products: list[Product], catalog_path: Path, encoder: MultimodalEncoder):
        self.products = products
        self.catalog_path = catalog_path
        self.encoder = encoder
        self.product_vectors = encode_products(products, catalog_path, encoder)

    @classmethod
    def from_jsonl(cls, path: str | Path, encoder: MultimodalEncoder) -> "SemanticRetriever":
        catalog_path = Path(path).resolve()
        products = load_catalog(catalog_path)
        return cls(products, catalog_path, encoder)

    @classmethod
    def from_index(
        cls,
        catalog_path: str | Path,
        index_dir: str | Path,
        encoder: MultimodalEncoder,
        expected_model: str | None = None,
    ) -> "SemanticRetriever":
        catalog = Path(catalog_path).resolve()
        products, vectors, _ = load_index(catalog, index_dir, expected_model)
        instance = cls.__new__(cls)
        instance.products = products
        instance.catalog_path = catalog
        instance.encoder = encoder
        instance.product_vectors = vectors
        return instance

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
