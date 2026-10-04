"""增量索引与灰度切换：让索引跟得上商品变更，并且**出问题能秒回退**。

为什么不能"变了就重建"
----------------------
全量重建在 6 条商品时是零成本，在百万级时是"编码百万条 + 建图数小时"。
于是索引只能按天更新，而电商的商品、价格、库存是分钟级变化的。

但增量更新真正的难点不是"怎么加数据"，而是这三件事：

1. **删除**。向量矩阵的行是按商品顺序排的，删一个商品就产生空洞；
   空洞不压实，索引与商品元数据就会错位，而且这种错位不会报错，只会返回错误的商品。
2. **版本**。增量更新必须产出**新的完整版本**，而不是就地改写当前版本——
   就地改写意味着更新失败时没有可回退的状态，线上只能挂着半残索引。
3. **切换**。新版本建好之后，如何让检索端"瞬间"切换过去、且切换本身是原子的。
   如果切换是"先删旧的、再写新的"，中间必然有一段空窗。

本模块分别用**压实**、**版本目录**、**CURRENT 指针原子替换**解决这三点：

    index_root/
      CURRENT          <- 单行文本，内容是要服务的版本号；用 os.replace 原子替换
      v1/  v2/  v3/    <- 每个版本自包含：manifest + ann.index + vectors.npy + products.jsonl

检索端的 `current_dir()` 每次都读指针，所以"回滚"只是把指针改回上一个版本号，
耗时是毫秒级，且不需要重新加载任何数据文件——旧版本目录一直都在。

一个必要的成本声明
------------------
增量更新要求基础版本**保留 ``vectors.npy``**：新向量的矩阵要基于旧矩阵拼出来，
只留一个打包好的 ``ann.index`` 是无法追加的。这也是 ``build_ann_index``
默认 ``keep_vectors=True`` 的实际理由之一。省掉这份副本确实能省磁盘，
但代价是每次变更都得全量重建。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from .ann_index import AnnIndexConfig
from .encoders import MultimodalEncoder, normalize
from .indexing import (
    MANIFEST_FILENAME,
    VECTORS_FILENAME,
    IndexManifest,
    build_ann_index,
    encode_products,
    load_version_catalog,
    read_manifest,
)
from .models import Product

CURRENT_POINTER = "CURRENT"
CHANGE_LOG_FILENAME = "changes.jsonl"


class CatalogChange(BaseModel):
    """一条商品变更。``seq`` 单调递增，用于断点续传与幂等重放。"""

    model_config = {"extra": "forbid"}

    op: Literal["upsert", "delete"]
    product_id: str
    product: Product | None = None
    seq: int = Field(default=0, ge=0)
    source: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> "CatalogChange":
        if self.op == "upsert":
            if self.product is None:
                raise ValueError("upsert 变更必须携带 product")
            if self.product.id != self.product_id:
                raise ValueError(f"product.id({self.product.id}) 与 product_id({self.product_id}) 不一致")
        elif self.product is not None:
            raise ValueError("delete 变更不应携带 product")
        return self


class ChangeLog:
    """追加式变更日志（JSONL）。每行一条变更，天然支持顺序重放与断点续传。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def append(self, changes: list[CatalogChange]) -> int:
        if not changes:
            return self.last_seq()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            for change in changes:
                stream.write(change.model_dump_json() + "\n")
        return self.last_seq()

    def read(self) -> list[CatalogChange]:
        if not self.path.exists():
            return []
        return [
            CatalogChange.model_validate_json(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def read_since(self, seq: int) -> list[CatalogChange]:
        return [change for change in self.read() if change.seq > seq]

    def last_seq(self) -> int:
        changes = self.read()
        return max((change.seq for change in changes), default=0)


def apply_changes(
    products: list[Product],
    vectors: np.ndarray,
    changes: list[CatalogChange],
    encoder: MultimodalEncoder,
    catalog_path: Path,
    image_weight: float = 0.5,
) -> tuple[list[Product], np.ndarray, dict[str, int]]:
    """把变更作用到 (商品, 向量) 上，返回压实后的新集合。

    压实（compaction）是这里的核心：删除会产生空洞，而向量矩阵的行号与商品列表的
    下标必须严格一一对应。做法是先把所有 upsert 追加/覆盖完，最后一次性地按保留掩码
    重建矩阵——而不是每删一个就挪一次数组（那是 O(N²)）。
    """
    if vectors.shape[0] != len(products):
        raise ValueError(f"向量条数 {vectors.shape[0]} 与商品数 {len(products)} 不一致")
    rows = {product.id: index for index, product in enumerate(products)}
    current_products = list(products)
    current_vectors = normalize(vectors)
    appended_products: list[Product] = []
    appended_vectors: list[np.ndarray] = []
    deleted_rows: set[int] = set()
    stats = {"upsert_new": 0, "upsert_update": 0, "delete_hit": 0, "delete_miss": 0, "deleted_orphan": 0}

    for change in changes:
        if change.op == "delete":
            row = rows.pop(change.product_id, None)
            if row is None:
                stats["delete_miss"] += 1
            else:
                deleted_rows.add(row)
                stats["delete_hit"] += 1
            continue
        assert change.product is not None
        vector = encode_products([change.product], catalog_path, encoder, image_weight)[0]
        row = rows.get(change.product.id)
        if row is None:
            rows[change.product.id] = len(current_products) + len(appended_products)
            appended_products.append(change.product)
            appended_vectors.append(vector)
            stats["upsert_new"] += 1
        else:
            current_products[row] = change.product
            current_vectors[row] = vector
            stats["upsert_update"] += 1

    if appended_products:
        current_products = current_products + appended_products
        current_vectors = np.vstack([current_vectors, np.asarray(appended_vectors, dtype=np.float32)])

    if deleted_rows:
        keep = np.asarray(
            [index for index in range(len(current_products)) if index not in deleted_rows], dtype=np.int64
        )
        current_products = [current_products[index] for index in keep.tolist()]
        current_vectors = current_vectors[keep]

    stats["final_products"] = len(current_products)
    return current_products, normalize(current_vectors), stats


class IndexVersionManager:
    """管理"多版本索引 + 单个活跃指针"。切换与回滚都只改指针，不动数据文件。"""

    def __init__(self, root: str | Path, prefix: str = "v") -> None:
        self.root = Path(root).resolve()
        self.prefix = prefix

    def version_dir(self, version: int) -> Path:
        if version < 1:
            raise ValueError("版本号必须大于等于 1")
        return self.root / f"{self.prefix}{version}"

    @property
    def pointer(self) -> Path:
        return self.root / CURRENT_POINTER

    def versions(self) -> list[int]:
        if not self.root.exists():
            return []
        found: list[int] = []
        for path in self.root.iterdir():
            if not path.is_dir() or not path.name.startswith(self.prefix):
                continue
            suffix = path.name[len(self.prefix) :]
            if suffix.isdigit():
                found.append(int(suffix))
        return sorted(found)

    def next_version(self) -> int:
        return (max(self.versions()) + 1) if self.versions() else 1

    def current_version(self) -> int | None:
        if not self.pointer.exists():
            return None
        try:
            return int(self.pointer.read_text(encoding="utf-8").strip())
        except ValueError:
            return None

    def current_dir(self) -> Path:
        version = self.current_version()
        if version is None:
            raise FileNotFoundError(f"{self.pointer} 不存在或内容非法，索引尚未发布任何版本")
        directory = self.version_dir(version)
        if not directory.exists():
            raise FileNotFoundError(f"指针指向 {directory}，但该目录不存在")
        return directory

    def verify(self, version: int) -> dict[str, object]:
        """发布前的体检：清单可读、声明的文件都在、商品数与向量条数对得上。

        这一步不能省。索引切换是原子的，但"原子地切到一个坏版本"同样是原子操作。
        """
        directory = self.version_dir(version)
        if not directory.exists():
            raise FileNotFoundError(f"版本目录不存在：{directory}")
        manifest = read_manifest(directory)
        missing = [name for name in manifest.index_files if not (directory / name).exists()]
        if missing:
            raise ValueError(f"版本 {version} 缺少声明的文件：{missing}")
        if (directory / MANIFEST_FILENAME).exists() and (directory / VECTORS_FILENAME).exists():
            vectors = np.load(directory / VECTORS_FILENAME, allow_pickle=False)
            if vectors.shape[0] != manifest.product_count:
                raise ValueError(
                    f"版本 {version} 的向量条数 {vectors.shape[0]} 与清单 {manifest.product_count} 不一致"
                )
        return {
            "version": version,
            "directory": str(directory),
            "manifest": manifest.model_dump(mode="json"),
            "files": sorted(path.name for path in directory.iterdir() if path.is_file()),
        }

    def promote(self, version: int) -> Path:
        """原子发布：先校验，再改指针。

        指针用 `os.replace` 替换 —— 在同一个文件系统内这是原子操作，
        读取方要么看到旧版本号、要么看到新版本号，不会读到半个数字。
        """
        self.verify(version)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / f"{CURRENT_POINTER}.tmp"
        temporary.write_text(str(version), encoding="utf-8")
        os.replace(temporary, self.pointer)
        return self.version_dir(version)

    def rollback(self) -> int:
        current = self.current_version()
        candidates = [version for version in self.versions() if current is None or version < current]
        if not candidates:
            raise ValueError("没有更早的版本可以回滚")
        target = candidates[-1]
        self.promote(target)
        return target

    def prune(self, keep: int = 2) -> list[int]:
        """清理旧版本目录，保留当前版本与最近 ``keep - 1`` 个其他版本。

        只删除本管理器根目录下、命名匹配 ``prefix + 数字`` 的子目录；
        不匹配的一律跳过，避免误删用户放在这里的其他东西。
        """
        if keep < 1:
            raise ValueError("keep 必须大于等于 1")
        current = self.current_version()
        ordered = [version for version in self.versions() if version != current]
        removable = ordered[: max(0, len(ordered) - (keep - 1))]
        removed: list[int] = []
        for version in removable:
            directory = self.version_dir(version)
            if directory.parent != self.root or not directory.name.startswith(self.prefix):
                continue
            shutil.rmtree(directory, ignore_errors=True)
            removed.append(version)
        return removed

    def status(self) -> dict[str, object]:
        current = self.current_version()
        return {
            "root": str(self.root),
            "current": current,
            "current_dir": str(self.version_dir(current)) if current else None,
            "versions": self.versions(),
            "pointer": str(self.pointer),
        }


def build_incremental_version(
    base_index_dir: str | Path,
    catalog_path: str | Path,
    output_dir: str | Path,
    changes: list[CatalogChange],
    encoder: MultimodalEncoder,
    config: AnnIndexConfig | None = None,
    model_name: str | None = None,
    image_weight: float | None = None,
    keep_vectors: bool = True,
) -> tuple[IndexManifest, dict[str, int]]:
    """基于一个已有版本 + 一批变更，产出**新的完整版本目录**。

    注意这里不碰基础版本：新版本写到 ``output_dir``，基础版本保持可服务状态，
    直到 ``promote`` 把指针切过去。更新失败时线上完全无感。
    """
    base = Path(base_index_dir).resolve()
    manifest = read_manifest(base)
    vectors_path = base / VECTORS_FILENAME
    if not vectors_path.exists():
        raise ValueError(
            f"基础版本 {base} 没有 {VECTORS_FILENAME}，无法做增量更新；"
            "请执行一次保留向量的全量重建（build_ann_index 的 keep_vectors=True）"
        )
    products = load_version_catalog(base, fallback=catalog_path)
    vectors = np.load(vectors_path, allow_pickle=False)
    resolved_weight = manifest.image_weight if image_weight is None else image_weight
    updated_products, updated_vectors, stats = apply_changes(
        products, vectors, changes, encoder, Path(catalog_path).resolve(), resolved_weight
    )
    resolved_config = config or manifest.ann or AnnIndexConfig(
        kind="numpy", dimension=manifest.vector_dimension
    )
    if resolved_config.dimension != manifest.vector_dimension:
        resolved_config = resolved_config.model_copy(update={"dimension": manifest.vector_dimension})
    new_manifest, _ = build_ann_index(
        catalog_path,
        output_dir,
        encoder,
        model_name or manifest.model_name,
        resolved_config,
        image_weight=resolved_weight,
        keep_vectors=keep_vectors,
        catalog_snapshot=True,
        products=updated_products,
        vectors=updated_vectors,
    )
    stats["base_products"] = manifest.product_count
    stats["base_index_type"] = manifest.index_type
    return new_manifest, stats


def load_changes(path: str | Path) -> list[CatalogChange]:
    return ChangeLog(path).read()


__all__ = [
    "CURRENT_POINTER",
    "CatalogChange",
    "ChangeLog",
    "IndexVersionManager",
    "apply_changes",
    "build_incremental_version",
    "load_changes",
]
