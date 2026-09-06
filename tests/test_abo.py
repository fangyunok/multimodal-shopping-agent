import csv
import gzip
import json
from pathlib import Path

from PIL import Image

from shopping_agent.abo import convert_abo, fetch_abo_subset_images, localized_value


def test_localized_value_prefers_requested_language() -> None:
    values = [
        {"language_tag": "en_US", "value": "Running shoes"},
        {"language_tag": "zh_CN", "value": "跑鞋"},
    ]
    assert localized_value(values, ("zh_CN", "en_US")) == "跑鞋"


def test_convert_extracted_abo_subset(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "listings" / "metadata"
    image_metadata_dir = tmp_path / "images" / "metadata"
    image_dir = tmp_path / "images" / "small" / "ab"
    metadata_dir.mkdir(parents=True)
    image_metadata_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16), "white").save(image_dir / "image.jpg")
    with gzip.open(image_metadata_dir / "images.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "height", "width", "path"])
        writer.writeheader()
        writer.writerow({"image_id": "image-1", "height": 16, "width": 16, "path": "ab/image.jpg"})
    listing = {
        "item_id": "SKU1",
        "domain_name": "amazon.com",
        "main_image_id": "image-1",
        "item_name": [{"language_tag": "zh_CN", "value": "白色运动鞋"}],
        "product_description": [{"language_tag": "zh_CN", "value": "<p>轻量通勤</p>"}],
        "product_type": "SHOES",
        "color": [{"language_tag": "zh_CN", "value": "白色"}],
    }
    with gzip.open(metadata_dir / "listings_0.json.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(listing, ensure_ascii=False) + "\n")
    output = tmp_path / "raw" / "abo.jsonl"
    summary = convert_abo(tmp_path, output, limit=10)
    product = json.loads(output.read_text(encoding="utf-8"))
    assert summary["converted"] == 1
    assert product["title"] == "白色运动鞋"
    assert product["description"] == "轻量通勤"
    assert product["license"] == "CC-BY-4.0"
    assert Path(product["image_path"]).exists()


def test_fetches_only_selected_images_and_resumes(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "listings" / "metadata"
    image_metadata_dir = tmp_path / "images" / "metadata"
    (tmp_path / "images" / "small").mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    image_metadata_dir.mkdir(parents=True)
    with gzip.open(image_metadata_dir / "images.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "height", "width", "path"])
        writer.writeheader()
        writer.writerow({"image_id": "one", "height": 16, "width": 16, "path": "aa/one.jpg"})
    with gzip.open(metadata_dir / "listings_0.json.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"item_id": "one", "main_image_id": "one"}) + "\n")

    calls = []

    def fake_download(url: str, target: Path) -> None:
        calls.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), "white").save(target)

    first = fetch_abo_subset_images(tmp_path, limit=1, workers=2, downloader=fake_download)
    second = fetch_abo_subset_images(tmp_path, limit=1, workers=2, downloader=fake_download)
    assert first == {"selected": 1, "downloaded": 1, "existing": 0, "failed": 0}
    assert second == {"selected": 1, "downloaded": 0, "existing": 1, "failed": 0}
    assert len(calls) == 1
