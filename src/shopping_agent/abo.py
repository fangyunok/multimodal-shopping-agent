from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from .models import Product

ABO_SOURCE = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html"
ABO_LICENSE = "CC-BY-4.0"
ABO_SMALL_IMAGE_BASE = "http://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/images/small"
HTML_TAG = re.compile(r"<[^>]+>")


def localized_value(values: object, languages: tuple[str, ...]) -> str:
    if not isinstance(values, list):
        return str(values or "")
    candidates = [item for item in values if isinstance(item, dict) and item.get("value") not in (None, "")]
    for language in languages:
        for item in candidates:
            if item.get("language_tag") == language:
                return str(item["value"])
    return str(candidates[0]["value"]) if candidates else ""


def joined_localized(values: object, languages: tuple[str, ...]) -> str:
    if not isinstance(values, list):
        return str(values or "")
    selected = []
    for item in values:
        if not isinstance(item, dict) or item.get("value") in (None, ""):
            continue
        if item.get("language_tag") in languages:
            selected.append(str(item["value"]))
    if not selected:
        fallback = localized_value(values, languages)
        selected = [fallback] if fallback else []
    return " ".join(selected)


def find_abo_paths(root: Path) -> tuple[list[Path], Path, Path]:
    listings = sorted(root.glob("listings/metadata/listings_*.json.gz"))
    image_metadata = root / "images" / "metadata" / "images.csv.gz"
    image_root = root / "images" / "small"
    if not listings:
        raise ValueError("未找到 listings/metadata/listings_*.json.gz")
    if not image_metadata.exists():
        raise ValueError("未找到 images/metadata/images.csv.gz")
    if not image_root.exists():
        raise ValueError("未找到 images/small 目录")
    return listings, image_metadata, image_root


def load_image_paths(metadata_path: Path) -> dict[str, str]:
    with gzip.open(metadata_path, "rt", encoding="utf-8", newline="") as stream:
        return {row["image_id"]: row["path"] for row in csv.DictReader(stream)}


def iter_listings(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=30) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def fetch_abo_subset_images(
    root_dir: str | Path,
    limit: int = 1000,
    workers: int = 8,
    downloader: Callable[[str, Path], None] = download_file,
) -> dict[str, int]:
    root = Path(root_dir).resolve()
    listings, image_metadata, image_root = find_abo_paths(root)
    image_paths = load_image_paths(image_metadata)
    selected: list[str] = []
    for item in iter_listings(listings):
        relative = image_paths.get(str(item.get("main_image_id")))
        if relative and relative not in selected:
            selected.append(relative)
        if len(selected) >= limit:
            break
    existing = [relative for relative in selected if (image_root / relative).exists()]
    pending = [relative for relative in selected if not (image_root / relative).exists()]
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(downloader, f"{ABO_SMALL_IMAGE_BASE}/{relative}", image_root / relative): relative
            for relative in pending
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                failed += 1
    return {"selected": len(selected), "downloaded": len(pending) - failed, "existing": len(existing), "failed": failed}


def convert_abo(
    root_dir: str | Path,
    output_path: str | Path,
    limit: int = 1000,
    languages: tuple[str, ...] = ("zh_CN", "en_US"),
) -> dict[str, int]:
    if limit < 1:
        raise ValueError("limit 必须大于 0")
    root = Path(root_dir).resolve()
    output = Path(output_path).resolve()
    listings, image_metadata, image_root = find_abo_paths(root)
    image_paths = load_image_paths(image_metadata)
    products: list[Product] = []
    skipped_missing_image = skipped_missing_title = 0
    for item in iter_listings(listings):
        image_id = item.get("main_image_id")
        relative_image = image_paths.get(str(image_id))
        if not relative_image or not (image_root / relative_image).exists():
            skipped_missing_image += 1
            continue
        title = localized_value(item.get("item_name"), languages)
        if not title:
            skipped_missing_title += 1
            continue
        category_value = item.get("product_type", "unknown")
        if isinstance(category_value, list):
            category = localized_value(category_value, languages)
        else:
            category = str(category_value or "unknown")
        description = " ".join(
            part for part in (
                joined_localized(item.get("bullet_point"), languages),
                joined_localized(item.get("product_description"), languages),
            ) if part
        )
        description = HTML_TAG.sub(" ", description).strip()[:4000]
        attributes = {}
        for output_key, source_key in (("品牌", "brand"), ("颜色", "color"), ("材质", "material"), ("风格", "style")):
            value = localized_value(item.get(source_key), languages)
            if value:
                attributes[output_key] = value
        product_id = f"{item.get('domain_name', 'amazon')}:{item['item_id']}"
        products.append(Product(
            id=product_id,
            title=title,
            category=category,
            description=description or title,
            price=0,
            attributes=attributes,
            image_path=str((image_root / relative_image).resolve()),
            rating=0,
            stock=1,
            source=ABO_SOURCE,
            license=ABO_LICENSE,
        ))
        if len(products) >= limit:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(product.model_dump_json() for product in products) + "\n", encoding="utf-8")
    return {
        "converted": len(products),
        "skipped_missing_image": skipped_missing_image,
        "skipped_missing_title": skipped_missing_title,
    }
