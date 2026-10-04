from pathlib import Path

import numpy as np
import pytest

from shopping_agent.ann_index import AnnIndexConfig
from shopping_agent.incremental import (
    CURRENT_POINTER,
    CatalogChange,
    ChangeLog,
    IndexVersionManager,
    apply_changes,
    build_incremental_version,
)
from shopping_agent.indexing import build_ann_index, encode_products, load_version_catalog
from shopping_agent.models import Product

KEYWORDS = ("鞋", "包", "帽", "衣")


class FakeEncoder:
    dimension = len(KEYWORDS)

    def encode_texts(self, texts):
        rows = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for position, keyword in enumerate(KEYWORDS):
                if keyword in text:
                    rows[row, position] = 1.0
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.maximum(norms, 1e-12)

    def encode_images(self, paths):
        return np.ones((len(paths), self.dimension), dtype=np.float32) / np.sqrt(self.dimension)


def product(identifier: str, title: str, price: float = 100.0) -> Product:
    return Product(id=identifier, title=title, category="测试", description=title, price=price)


def write_catalog(path: Path, products: list[Product]) -> Path:
    path.write_text("\n".join(item.model_dump_json() for item in products) + "\n", encoding="utf-8")
    return path


def base_products(count: int = 6) -> list[Product]:
    return [product(f"p-{index:03d}", "运动鞋" if index % 2 else "通勤包") for index in range(count)]


def assert_rows_aligned(products: list[Product], vectors: np.ndarray, catalog: Path) -> None:
    """逐行核对：向量矩阵第 i 行必须恰好是商品列表第 i 条的向量。

    这是增量更新最容易出错的地方——删一个商品后忘了压实，行号就会整体错位，
    而且错位不会抛异常，只会静默返回错误的商品。
    """
    expected = encode_products(products, catalog, FakeEncoder(), image_weight=0.0)
    assert vectors.shape[0] == len(products)
    assert np.allclose(vectors, expected, atol=1e-6)


def test_apply_changes_upsert_new_and_update(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path / "products.jsonl", base_products())
    products = base_products()
    vectors = encode_products(products, catalog, FakeEncoder(), image_weight=0.0)
    changes = [
        CatalogChange(op="upsert", product_id="p-new", product=product("p-new", "白色棒球帽")),
        CatalogChange(op="upsert", product_id="p-000", product=product("p-000", "防水外套", price=999)),
    ]
    updated_products, updated_vectors, stats = apply_changes(
        products, vectors, changes, FakeEncoder(), catalog, image_weight=0.0
    )
    assert stats["upsert_new"] == 1 and stats["upsert_update"] == 1
    assert len(updated_products) == 7
    assert updated_products[-1].id == "p-new"
    assert updated_products[0].price == 999
    assert_rows_aligned(updated_products, updated_vectors, catalog)


def test_apply_changes_delete_compacts_rows(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path / "products.jsonl", base_products())
    products = base_products()
    vectors = encode_products(products, catalog, FakeEncoder(), image_weight=0.0)
    changes = [
        CatalogChange(op="delete", product_id="p-001"),
        CatalogChange(op="delete", product_id="p-004"),
    ]
    updated_products, updated_vectors, stats = apply_changes(
        products, vectors, changes, FakeEncoder(), catalog, image_weight=0.0
    )
    assert stats["delete_hit"] == 2
    assert [item.id for item in updated_products] == ["p-000", "p-002", "p-003", "p-005"]
    assert_rows_aligned(updated_products, updated_vectors, catalog)


def test_apply_changes_delete_unknown_id_is_counted_not_fatal(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path / "products.jsonl", base_products(3))
    products = base_products(3)
    vectors = encode_products(products, catalog, FakeEncoder(), image_weight=0.0)
    _, _, stats = apply_changes(
        products, vectors, [CatalogChange(op="delete", product_id="ghost")], FakeEncoder(), catalog, 0.0
    )
    assert stats["delete_miss"] == 1
    assert stats["final_products"] == 3


def test_apply_changes_rejects_mismatched_shapes(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path / "products.jsonl", base_products(3))
    with pytest.raises(ValueError) as error:
        apply_changes(base_products(3), np.zeros((2, 4), dtype=np.float32), [], FakeEncoder(), catalog, 0.0)
    assert "不一致" in str(error.value)


def test_change_model_validates_product_id() -> None:
    with pytest.raises(ValueError) as error:
        CatalogChange(op="upsert", product_id="a", product=product("b", "运动鞋"))
    assert "不一致" in str(error.value)
    with pytest.raises(ValueError) as error:
        CatalogChange(op="upsert", product_id="a")
    assert "必须携带" in str(error.value)
    with pytest.raises(ValueError) as error:
        CatalogChange(op="delete", product_id="a", product=product("a", "运动鞋"))
    assert "不应携带" in str(error.value)


def test_change_log_append_and_read_since(tmp_path: Path) -> None:
    log = ChangeLog(tmp_path / "changes.jsonl")
    assert log.last_seq() == 0
    last = log.append(
        [
            CatalogChange(op="upsert", product_id="p-1", product=product("p-1", "运动鞋"), seq=1),
            CatalogChange(op="delete", product_id="p-2", seq=2),
        ]
    )
    assert last == 2
    assert len(log.read()) == 2
    assert [change.product_id for change in log.read_since(1)] == ["p-2"]
    assert log.read_since(2) == []


def build_version(tmp_path: Path, version: int, count: int) -> Path:
    catalog = write_catalog(tmp_path / f"catalog-v{version}.jsonl", base_products(count))
    target = tmp_path / "index_root" / f"v{version}"
    build_ann_index(
        catalog,
        target,
        FakeEncoder(),
        "fake-clip",
        AnnIndexConfig(kind="numpy", dimension=FakeEncoder.dimension),
        image_weight=0.0,
        catalog_snapshot=True,
    )
    return target


def test_version_manager_promote_and_rollback(tmp_path: Path) -> None:
    manager = IndexVersionManager(tmp_path / "index_root")
    assert manager.current_version() is None
    assert manager.next_version() == 1
    build_version(tmp_path, 1, 4)
    build_version(tmp_path, 2, 5)
    assert manager.versions() == [1, 2]

    manager.promote(1)
    assert manager.current_version() == 1
    assert manager.current_dir().name == "v1"

    manager.promote(2)
    assert manager.current_version() == 2

    assert manager.rollback() == 1
    assert manager.current_version() == 1
    # 回滚只改指针，数据文件原地未动——所以它是毫秒级的。
    assert (manager.version_dir(2) / "manifest.json").exists()


def test_promote_rejects_incomplete_version(tmp_path: Path) -> None:
    manager = IndexVersionManager(tmp_path / "index_root")
    target = build_version(tmp_path, 1, 3)
    (target / "vectors.npy").unlink()
    with pytest.raises(ValueError) as error:
        manager.promote(1)
    assert "缺少声明的文件" in str(error.value)
    assert manager.current_version() is None, "校验失败时不能改指针"


def test_promote_rejects_missing_version(tmp_path: Path) -> None:
    manager = IndexVersionManager(tmp_path / "index_root")
    with pytest.raises(FileNotFoundError):
        manager.promote(9)


def test_rollback_without_history_is_rejected(tmp_path: Path) -> None:
    manager = IndexVersionManager(tmp_path / "index_root")
    build_version(tmp_path, 1, 3)
    manager.promote(1)
    with pytest.raises(ValueError) as error:
        manager.rollback()
    assert "没有更早的版本" in str(error.value)


def test_prune_keeps_current_and_recent(tmp_path: Path) -> None:
    manager = IndexVersionManager(tmp_path / "index_root")
    for version in range(1, 5):
        build_version(tmp_path, version, 3)
    manager.promote(4)
    removed = manager.prune(keep=2)
    assert removed == [1, 2]
    assert manager.versions() == [3, 4]
    assert manager.current_version() == 4


def test_status_reports_pointer(tmp_path: Path) -> None:
    manager = IndexVersionManager(tmp_path / "index_root")
    build_version(tmp_path, 1, 3)
    manager.promote(1)
    status = manager.status()
    assert status["current"] == 1
    assert status["versions"] == [1]
    assert Path(status["pointer"]).name == CURRENT_POINTER


def test_build_incremental_version_end_to_end(tmp_path: Path) -> None:
    base_dir = build_version(tmp_path, 1, 6)
    catalog = tmp_path / "catalog-v1.jsonl"
    new_dir = tmp_path / "index_root" / "v2"
    changes = [
        CatalogChange(op="upsert", product_id="p-100", product=product("p-100", "白色棒球帽", 88.0), seq=1),
        CatalogChange(op="delete", product_id="p-001", seq=2),
    ]
    manifest, stats = build_incremental_version(base_dir, catalog, new_dir, changes, FakeEncoder(), image_weight=0.0)
    assert stats["upsert_new"] == 1 and stats["delete_hit"] == 1
    assert manifest.product_count == 6
    assert manifest.has_catalog_snapshot is True

    manager = IndexVersionManager(tmp_path / "index_root")
    manager.promote(2)
    assert manager.current_version() == 2
    # 新版本自带快照，因此不依赖外部目录也能恢复出正确的商品集合。
    snapshot = load_version_catalog(new_dir)
    assert "p-001" not in [item.id for item in snapshot]
    assert "p-100" in [item.id for item in snapshot]


def test_build_incremental_requires_base_vectors(tmp_path: Path) -> None:
    catalog = write_catalog(tmp_path / "catalog.jsonl", base_products(4))
    base = tmp_path / "index_root" / "v1"
    build_ann_index(
        catalog,
        base,
        FakeEncoder(),
        "fake-clip",
        AnnIndexConfig(kind="numpy", dimension=FakeEncoder.dimension),
        keep_vectors=False,
    )
    with pytest.raises(ValueError) as error:
        build_incremental_version(base, catalog, tmp_path / "v2", [], FakeEncoder())
    assert "vectors.npy" in str(error.value)
