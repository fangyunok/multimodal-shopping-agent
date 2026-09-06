import json
from pathlib import Path

import pytest
from PIL import Image

from shopping_agent.dataset import prepare_dataset, stable_split


def raw_product(product_id: str, image_path: str | None = None) -> dict:
    return {
        "id": product_id,
        "title": f"商品 {product_id}",
        "category": "测试",
        "description": "可复现实验商品",
        "price": 100,
        "image_path": image_path,
        "source": "self-created-test-data",
        "license": "CC0-1.0",
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records), encoding="utf-8")


def test_prepare_dataset_validates_copies_and_deduplicates_images(tmp_path: Path) -> None:
    image = tmp_path / "shoe.png"
    Image.new("RGB", (16, 16), "white").save(image)
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, [raw_product("one", "shoe.png"), raw_product("two", "shoe.png")])
    catalog = tmp_path / "processed" / "products.jsonl"
    summary = prepare_dataset(raw, catalog, tmp_path / "processed" / "images")
    records = [json.loads(line) for line in catalog.read_text(encoding="utf-8").splitlines()]
    assert summary["unique_images"] == 1
    assert summary["duplicate_images"] == 1
    assert records[0]["image_sha256"] == records[1]["image_sha256"]
    assert (catalog.parent / records[0]["image_path"]).exists()


def test_split_is_stable() -> None:
    assert stable_split("sku-123") == stable_split("sku-123")
    assert stable_split("sku-123") in {"train", "validation", "test"}


def test_prepare_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, [raw_product("same"), raw_product("same")])
    with pytest.raises(ValueError, match="商品 ID 重复"):
        prepare_dataset(raw, tmp_path / "out.jsonl", tmp_path / "images")


def test_prepare_dataset_requires_provenance(tmp_path: Path) -> None:
    record = raw_product("one")
    record["license"] = None
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, [record])
    with pytest.raises(ValueError, match="source 或 license"):
        prepare_dataset(raw, tmp_path / "out.jsonl", tmp_path / "images")


def test_prepare_dataset_rejects_invalid_image(tmp_path: Path) -> None:
    (tmp_path / "fake.png").write_text("not an image", encoding="utf-8")
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, [raw_product("one", "fake.png")])
    with pytest.raises(ValueError, match="有效图片"):
        prepare_dataset(raw, tmp_path / "out.jsonl", tmp_path / "images")

