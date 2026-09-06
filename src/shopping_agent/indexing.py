from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from .encoders import MultimodalEncoder, normalize
from .models import Product


class IndexManifest(BaseModel):
    version: int = 1
    model_name: str
    catalog_sha256: str
    product_count: int
    vector_dimension: int
    created_at: str
    image_weight: float = 0.5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> list[Product]:
    return [Product.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_products(
    products: list[Product], catalog_path: Path, encoder: MultimodalEncoder, image_weight: float = 0.5
) -> np.ndarray:
    if not 0 <= image_weight <= 1:
        raise ValueError("image_weight 必须在 0 到 1 之间")
    text_vectors = normalize(encoder.encode_texts([product.searchable_text() for product in products]))
    product_vectors = text_vectors.copy()
    image_rows: list[int] = []
    image_paths: list[Path] = []
    for row, product in enumerate(products):
        image_path = product.resolved_image(catalog_path)
        if image_path and image_path.exists():
            image_rows.append(row)
            image_paths.append(image_path)
    if image_paths and image_weight > 0:
        image_vectors = normalize(encoder.encode_images(image_paths))
        for row, image_vector in zip(image_rows, image_vectors):
            product_vectors[row] = normalize(
                np.asarray([(1 - image_weight) * text_vectors[row] + image_weight * image_vector])
            )[0]
    return normalize(product_vectors)


def build_index(
    catalog_path: str | Path,
    output_dir: str | Path,
    encoder: MultimodalEncoder,
    model_name: str,
    image_weight: float = 0.5,
) -> IndexManifest:
    catalog = Path(catalog_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    products = load_catalog(catalog)
    if not products:
        raise ValueError("商品目录不能为空")
    vectors = encode_products(products, catalog, encoder, image_weight)
    manifest = IndexManifest(
        model_name=model_name,
        catalog_sha256=file_sha256(catalog),
        product_count=len(products),
        vector_dimension=int(vectors.shape[1]),
        created_at=datetime.now(timezone.utc).isoformat(),
        image_weight=image_weight,
    )
    np.save(output / "vectors.npy", vectors, allow_pickle=False)
    (output / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_index(
    catalog_path: str | Path,
    index_dir: str | Path,
    expected_model: str | None = None,
) -> tuple[list[Product], np.ndarray, IndexManifest]:
    catalog = Path(catalog_path).resolve()
    index = Path(index_dir).resolve()
    manifest = IndexManifest.model_validate_json((index / "manifest.json").read_text(encoding="utf-8"))
    if manifest.catalog_sha256 != file_sha256(catalog):
        raise ValueError("商品目录已变化，请重新构建向量索引")
    if expected_model and manifest.model_name != expected_model:
        raise ValueError(f"索引模型为 {manifest.model_name}，但当前配置为 {expected_model}")
    products = load_catalog(catalog)
    vectors = np.load(index / "vectors.npy", allow_pickle=False)
    expected_shape = (manifest.product_count, manifest.vector_dimension)
    if vectors.shape != expected_shape or len(products) != manifest.product_count:
        raise ValueError("索引文件维度或商品数量与清单不一致")
    return products, normalize(vectors), manifest
