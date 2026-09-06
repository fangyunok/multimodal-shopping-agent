from pathlib import Path

import numpy as np

from shopping_agent.indexing import build_index, load_index
from shopping_agent.models import Product, SearchRequest
from shopping_agent.semantic_retrieval import SemanticRetriever


class FakeEncoder:
    def encode_texts(self, texts):
        return np.asarray([[1, 0] if "鞋" in text else [0, 1] for text in texts], dtype=np.float32)

    def encode_images(self, paths):
        return np.asarray([[1, 0] for _ in paths], dtype=np.float32)


def write_catalog(path: Path) -> None:
    products = [
        Product(id="shoe", title="鞋", category="运动鞋", description="通勤", price=300),
        Product(id="bag", title="包", category="箱包", description="旅行", price=200),
    ]
    path.write_text("\n".join(product.model_dump_json() for product in products), encoding="utf-8")


def test_build_load_and_search_persistent_index(tmp_path: Path) -> None:
    catalog = tmp_path / "products.jsonl"
    index = tmp_path / "index"
    write_catalog(catalog)
    manifest = build_index(catalog, index, FakeEncoder(), "fake-clip")
    assert manifest.product_count == 2
    assert (index / "vectors.npy").exists()
    retriever = SemanticRetriever.from_index(catalog, index, FakeEncoder(), "fake-clip")
    assert retriever.search(SearchRequest(query="鞋"))[0].product.id == "shoe"


def test_index_rejects_changed_catalog(tmp_path: Path) -> None:
    catalog = tmp_path / "products.jsonl"
    index = tmp_path / "index"
    write_catalog(catalog)
    build_index(catalog, index, FakeEncoder(), "fake-clip")
    catalog.write_text(catalog.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    try:
        load_index(catalog, index)
    except ValueError as error:
        assert "重新构建" in str(error)
    else:
        raise AssertionError("changed catalog must invalidate the index")


def test_index_rejects_wrong_model(tmp_path: Path) -> None:
    catalog = tmp_path / "products.jsonl"
    index = tmp_path / "index"
    write_catalog(catalog)
    build_index(catalog, index, FakeEncoder(), "fake-clip")
    try:
        load_index(catalog, index, "other-model")
    except ValueError as error:
        assert "索引模型" in str(error)
    else:
        raise AssertionError("wrong model must invalidate the index")


def test_index_rejects_invalid_image_weight(tmp_path: Path) -> None:
    catalog = tmp_path / "products.jsonl"
    write_catalog(catalog)
    try:
        build_index(catalog, tmp_path / "index", FakeEncoder(), "fake-clip", 1.1)
    except ValueError as error:
        assert "image_weight" in str(error)
    else:
        raise AssertionError("invalid image weight must be rejected")
