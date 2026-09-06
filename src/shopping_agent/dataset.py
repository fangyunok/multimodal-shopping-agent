from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .models import Product

SUPPORTED_IMAGE_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def stable_split(product_id: str) -> str:
    bucket = int(hashlib.sha256(product_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    raise ValueError("原始数据仅支持 .jsonl 或 .csv")


def prepare_dataset(
    input_path: str | Path,
    output_catalog: str | Path,
    image_dir: str | Path,
) -> dict[str, int | dict[str, int]]:
    source_path = Path(input_path).resolve()
    catalog_path = Path(output_catalog).resolve()
    images_path = Path(image_dir).resolve()
    raw_records = read_raw_records(source_path)
    if not raw_records:
        raise ValueError("原始商品数据不能为空")
    products: list[Product] = []
    seen_ids: set[str] = set()
    image_targets: dict[str, Path] = {}
    duplicate_images = 0
    for row, record in enumerate(raw_records, start=1):
        product = Product.model_validate(record)
        if product.id in seen_ids:
            raise ValueError(f"第 {row} 行商品 ID 重复: {product.id}")
        seen_ids.add(product.id)
        if not product.source or not product.license:
            raise ValueError(f"第 {row} 行缺少 source 或 license，无法追踪数据来源")
        if product.image_path:
            original = Path(product.image_path)
            if not original.is_absolute():
                original = source_path.parent / original
            if not original.exists():
                raise ValueError(f"第 {row} 行图片不存在: {original}")
            try:
                with Image.open(original) as image:
                    image.verify()
                    suffix = SUPPORTED_IMAGE_FORMATS.get(image.format or "")
            except (UnidentifiedImageError, OSError) as error:
                raise ValueError(f"第 {row} 行不是有效图片: {original}") from error
            if suffix is None:
                raise ValueError(f"第 {row} 行图片格式不支持，仅允许 JPEG、PNG、WebP")
            digest = file_sha256(original)
            target = image_targets.get(digest)
            if target is None:
                target = images_path / f"{digest}{suffix}"
                image_targets[digest] = target
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(original, target)
            else:
                duplicate_images += 1
            product.image_path = Path(target).relative_to(catalog_path.parent).as_posix()
            product.image_sha256 = digest
        product.split = stable_split(product.id)
        products.append(product)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text("\n".join(product.model_dump_json() for product in products) + "\n", encoding="utf-8")
    split_counts = Counter(product.split for product in products)
    return {
        "products": len(products),
        "unique_images": len(image_targets),
        "duplicate_images": duplicate_images,
        "splits": {name: split_counts.get(name, 0) for name in ("train", "validation", "test")},
    }

