from pathlib import Path

import numpy as np
from PIL import Image

from shopping_agent.models import Product, SearchRequest
from shopping_agent.semantic_retrieval import SemanticRetriever


class FakeSharedEncoder:
    def encode_texts(self, texts):
        return np.asarray([[1, 0] if "鞋" in text else [0, 1] for text in texts], dtype=np.float32)

    def encode_images(self, paths):
        return np.asarray([[1, 0] if "shoe" in path.name else [0, 1] for path in paths], dtype=np.float32)


def test_cross_modal_image_to_product_search(tmp_path: Path) -> None:
    shoe_image = tmp_path / "shoe.png"
    bag_image = tmp_path / "bag.png"
    Image.new("RGB", (8, 8), "white").save(shoe_image)
    Image.new("RGB", (8, 8), "black").save(bag_image)
    products = [
        Product(id="shoe", title="运动鞋", category="鞋", description="通勤", price=300, image_path="shoe.png"),
        Product(id="bag", title="托特包", category="包", description="通勤", price=200, image_path="bag.png"),
    ]
    catalog = tmp_path / "products.jsonl"
    retriever = SemanticRetriever(products, catalog, FakeSharedEncoder())
    hits = retriever.search(SearchRequest(image_path=str(shoe_image)))
    assert hits[0].product.id == "shoe"
    assert hits[0].reasons == ["CLIP 图文语义相似"]


def test_semantic_search_keeps_business_constraints(tmp_path: Path) -> None:
    products = [
        Product(id="cheap", title="鞋", category="鞋", description="", price=300),
        Product(id="expensive", title="鞋", category="鞋", description="", price=900),
    ]
    retriever = SemanticRetriever(products, tmp_path / "products.jsonl", FakeSharedEncoder())
    hits = retriever.search(SearchRequest(query="鞋", max_price=500))
    assert [hit.product.id for hit in hits] == ["cheap"]
