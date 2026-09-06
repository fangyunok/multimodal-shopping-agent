from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .models import Product, SearchHit, SearchRequest

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def image_descriptor(path: Path) -> np.ndarray:
    """Small CPU baseline: RGB histograms plus average brightness."""
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((64, 64))
        pixels = np.asarray(rgb, dtype=np.float32) / 255.0
    parts = [np.histogram(pixels[..., channel], bins=16, range=(0, 1), density=True)[0] for channel in range(3)]
    descriptor = np.concatenate([*parts, pixels.mean(axis=(0, 1))])
    norm = np.linalg.norm(descriptor)
    return descriptor / norm if norm else descriptor


class HybridRetriever:
    def __init__(self, products: list[Product], catalog_path: Path):
        self.products = products
        self.catalog_path = catalog_path
        self.documents = [Counter(tokenize(product.searchable_text())) for product in products]
        document_frequency = Counter(token for doc in self.documents for token in doc)
        count = max(len(products), 1)
        self.idf = {token: math.log((count + 1) / (freq + 1)) + 1 for token, freq in document_frequency.items()}
        self.image_vectors: dict[str, np.ndarray] = {}
        for product in products:
            image_path = product.resolved_image(catalog_path)
            if image_path and image_path.exists():
                self.image_vectors[product.id] = image_descriptor(image_path)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "HybridRetriever":
        catalog_path = Path(path).resolve()
        products = [Product.model_validate_json(line) for line in catalog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(products, catalog_path)

    def _text_scores(self, query: str) -> list[float]:
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return [0.0] * len(self.products)
        scores: list[float] = []
        for document in self.documents:
            overlap = sum(min(freq, document[token]) * self.idf.get(token, 1.0) for token, freq in query_tokens.items())
            scores.append(overlap / max(sum(query_tokens.values()), 1))
        maximum = max(scores, default=0)
        return [score / maximum if maximum else 0.0 for score in scores]

    def search(self, request: SearchRequest) -> list[SearchHit]:
        text_scores = self._text_scores(request.query)
        query_image = image_descriptor(Path(request.image_path)) if request.image_path else None
        hits: list[SearchHit] = []
        for product, text_score in zip(self.products, text_scores):
            if request.max_price is not None and product.price > request.max_price:
                continue
            if request.category and request.category.lower() not in product.category.lower():
                continue
            image_score = 0.0
            if query_image is not None and product.id in self.image_vectors:
                image_score = max(0.0, cosine(query_image, self.image_vectors[product.id]))
            text_weight = 0.65 if request.query else 0.0
            image_weight = 0.35 if query_image is not None and request.query else (1.0 if query_image is not None else 0.0)
            score = text_weight * text_score + image_weight * image_score
            if not request.query and query_image is None:
                score = product.rating / 5
            reasons = []
            if text_score > 0:
                reasons.append("商品文本与需求匹配")
            if image_score > 0:
                reasons.append("商品图片视觉特征相似")
            if request.max_price is not None:
                reasons.append(f"价格不超过 ¥{request.max_price:g}")
            hits.append(SearchHit(product=product, score=round(score, 4), text_score=round(text_score, 4), image_score=round(image_score, 4), reasons=reasons))
        return sorted(hits, key=lambda hit: (hit.score, hit.product.rating), reverse=True)[: request.top_k]


def load_products(path: str | Path) -> list[Product]:
    return [Product.model_validate(item) for item in json.loads(Path(path).read_text(encoding="utf-8"))]

