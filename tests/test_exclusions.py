"""同一排除条件经过不同检索后端时必须保持一致，并补足剩余候选。"""
from pathlib import Path

import numpy as np
import pytest

from shopping_agent.ann_index import AnnIndexConfig
from shopping_agent.bm25 import BM25Retriever
from shopping_agent.faiss_retrieval import FaissRetriever
from shopping_agent.fusion_retrieval import ScoreFusionRetriever
from shopping_agent.models import Product, SearchRequest
from shopping_agent.partitioned import PartitionedRetriever
from shopping_agent.retrieval import HybridRetriever
from shopping_agent.rrf import ReciprocalRankFusionRetriever
from shopping_agent.semantic_retrieval import SemanticRetriever


class Encoder:
    def encode_texts(self, texts):
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))

    def encode_images(self, paths):
        return self.encode_texts(paths)


@pytest.mark.parametrize("backend", ["baseline", "semantic", "ann", "partitioned", "bm25", "fusion", "rrf"])
@pytest.mark.parametrize("query", ["运动鞋", ""])
def test_exclusion_filters_before_final_top_k(tmp_path: Path, backend: str, query: str) -> None:
    products = [
        Product(id=f"shoe-{i:03}", title="运动鞋", category="运动鞋", description="通勤", price=100 + i,
                rating=5 - i / 10, stock=1)
        for i in range(1, 5)
    ]
    catalog = tmp_path / "products.jsonl"
    catalog.write_text("\n".join(p.model_dump_json() for p in products), encoding="utf-8")
    encoder = Encoder()
    baseline = HybridRetriever(products, catalog)
    semantic = SemanticRetriever(products, catalog, encoder)
    bm25 = BM25Retriever.from_products(products, catalog)
    if backend == "baseline":
        retriever = baseline
    elif backend == "semantic":
        retriever = semantic
    elif backend == "bm25":
        retriever = bm25
    elif backend == "fusion":
        retriever = ScoreFusionRetriever(baseline, semantic)
    elif backend == "rrf":
        retriever = ReciprocalRankFusionRetriever([(bm25, 1.0), (semantic, 1.0)])
    elif backend == "partitioned":
        retriever = PartitionedRetriever.from_vectors(
            products, catalog, encoder, encoder.encode_texts(products),
            AnnIndexConfig(kind="numpy", dimension=2), by="price", buckets=2,
        )
    else:
        retriever = FaissRetriever.from_vectors(
            products, catalog, encoder, encoder.encode_texts(products),
            AnnIndexConfig(kind="numpy", dimension=2),
        )
    excluded = [p.id for p in products[:3]]
    hits = retriever.search(SearchRequest(query=query, top_k=1, excluded_product_ids=excluded))
    assert [hit.product.id for hit in hits] == [products[-1].id]
    assert retriever.search(SearchRequest(query=query, excluded_product_ids=[p.id for p in products])) == []
