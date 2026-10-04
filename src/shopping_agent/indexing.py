from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from .ann_index import ANN_INDEX_FILENAME, AnnIndex, AnnIndexConfig
from .encoders import MultimodalEncoder, normalize
from .models import Product

VECTORS_FILENAME = "vectors.npy"
MANIFEST_FILENAME = "manifest.json"
CATALOG_SNAPSHOT_FILENAME = "products.jsonl"


class IndexManifest(BaseModel):
    """索引清单。

    v1 只描述"一份精确向量矩阵"：模型名 + 目录哈希 + 形状。
    v2 在此之上补三样东西，都是为了规模化之后仍然可复现、可回退：

    - ``index_type`` / ``ann``：这份索引到底是哪种 ANN 后端、用什么参数建出来的。
      没有它，压测数字就无法归因——「HNSW 召回 0.93」必须能对上「efSearch 是多少」。
    - ``index_files``：本次索引真正落盘的文件清单。FAISS 索引可能存在 ``ann.index`` 里，
      也可能为了省磁盘而不写 ``vectors.npy``，加载端需要知道该找什么。
    - ``effective``：实际生效的参数。IVF 的桶数在数据量小时会被强制收缩，
      不记录这个就会得到"配了 1024 桶、其实只用了 25 桶"的假结论。

    v1 清单仍然可以正常加载：``ann`` 缺省即视为 numpy 精确后端。
    """

    version: int = 1
    model_name: str
    catalog_sha256: str
    product_count: int
    vector_dimension: int
    created_at: str
    image_weight: float = 0.5
    index_type: str = "numpy"
    backend: str = "numpy"
    ann: AnnIndexConfig | None = None
    index_files: list[str] = Field(default_factory=list)
    has_vectors: bool = True
    has_catalog_snapshot: bool = False
    # 分段索引（过滤下推）专用：记录按什么维度切、切成几段。
    # 详细的每段布局在 partitions.json 里，这里只留够定位用的字段——
    # 加载器要先能一眼看出"这不是单索引"，才能给出可执行的报错而不是索引类型不匹配。
    partition_by: str | None = None
    partition_count: int = 0
    effective: dict = Field(default_factory=dict)


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


# ---------- 清单与校验 ----------


def read_manifest(index_dir: str | Path) -> IndexManifest:
    index = Path(index_dir).resolve()
    path = index / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"未找到索引清单 {path}")
    return IndexManifest.model_validate_json(path.read_text(encoding="utf-8"))


def _verify_manifest(manifest: IndexManifest, catalog: Path, index: Path, expected_model: str | None) -> None:
    """校验索引与商品目录是否配对。

    若版本目录里带了商品快照，就以快照为准：索引版本必须能独立自证，
    否则多个版本并行存在时，"这个版本当时服务的是哪批商品"就无从判断。
    没有快照时才回退到外部目录（v1 行为）。
    """
    snapshot = index / CATALOG_SNAPSHOT_FILENAME
    reference = snapshot if snapshot.exists() else catalog
    if not reference.exists():
        raise ValueError("商品目录已变化，请重新构建向量索引")
    if manifest.catalog_sha256 != file_sha256(reference):
        raise ValueError("商品目录已变化，请重新构建向量索引")
    if expected_model and manifest.model_name != expected_model:
        raise ValueError(f"索引模型为 {manifest.model_name}，但当前配置为 {expected_model}")


def _read_vectors(index: Path, manifest: IndexManifest) -> np.ndarray:
    path = index / VECTORS_FILENAME
    if not path.exists():
        raise ValueError(
            f"索引目录缺少 {VECTORS_FILENAME}（清单标记 has_vectors={manifest.has_vectors}）；"
            "numpy/精确后端需要该文件，FAISS 后端请改用 load_ann_index"
        )
    vectors = np.load(path, allow_pickle=False)
    expected_shape = (manifest.product_count, manifest.vector_dimension)
    if vectors.shape != expected_shape:
        raise ValueError("索引文件维度或商品数量与清单不一致")
    return normalize(vectors)


def _resolve_config(manifest: IndexManifest, config: AnnIndexConfig | None) -> AnnIndexConfig:
    """清单里的 ANN 参数是权威：它可以不重建索引就被覆盖运行期旋钮，但不能被悄悄换掉维度。"""
    resolved = config or manifest.ann or AnnIndexConfig(kind="numpy", dimension=manifest.vector_dimension)
    if resolved.dimension != manifest.vector_dimension:
        resolved = resolved.model_copy(update={"dimension": manifest.vector_dimension})
    return resolved


# ---------- 构建 ----------


def build_index(
    catalog_path: str | Path,
    output_dir: str | Path,
    encoder: MultimodalEncoder,
    model_name: str,
    image_weight: float = 0.5,
) -> IndexManifest:
    """v1 路径：只存一份精确向量矩阵。小规模场景（数千条以内）这就够了。"""
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
        index_files=[VECTORS_FILENAME],
    )
    np.save(output / VECTORS_FILENAME, vectors, allow_pickle=False)
    (output / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def build_ann_index(
    catalog_path: str | Path,
    output_dir: str | Path,
    encoder: MultimodalEncoder,
    model_name: str,
    config: AnnIndexConfig,
    image_weight: float = 0.5,
    keep_vectors: bool = True,
    catalog_snapshot: bool = False,
    products: list[Product] | None = None,
    vectors: np.ndarray | None = None,
) -> tuple[IndexManifest, AnnIndex]:
    """v2 路径：按指定 ANN 后端建索引。

    ``keep_vectors=False`` 会**不写 ``vectors.npy``**。百万级 512 维向量裸矩阵是 2 GB，
    对 HNSW/IVF-PQ 这类自带存储的后端来说这份副本纯属浪费磁盘 —— 但代价是失去
    "随时降级回精确检索"的能力，所以默认仍然保留。

    ``catalog_snapshot=True`` 会把商品目录**快照**一份进版本目录（约几十 MB，
    远小于向量矩阵）。这不是冗余：多个索引版本并行存在时，商品目录可能已经变了，
    快照是"这个版本当时到底服务的是哪批商品"的唯一证据，也是回滚能自洽的前提。

    ``products`` / ``vectors`` 用于**增量路径**：调用方已经算好更新后的商品与向量，
    此时跳过编码。保持"只有这一处写清单"很重要——清单是索引的唯一真相来源，
    多一条写入路径就多一种不一致的可能。
    """
    catalog = Path(catalog_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if products is None:
        products = load_catalog(catalog)
        if not products:
            raise ValueError("商品目录不能为空")
    elif not products:
        raise ValueError("商品目录不能为空")
    if vectors is None:
        vectors = encode_products(products, catalog, encoder, image_weight)
    if vectors.shape[0] != len(products):
        raise ValueError(f"向量条数 {vectors.shape[0]} 与商品数 {len(products)} 不一致")
    # 维度由编码器决定，不是用户选项：以真实向量为准，避免手填参数造成的静默不一致。
    if config.dimension != int(vectors.shape[1]):
        config = config.model_copy(update={"dimension": int(vectors.shape[1])})

    ann = AnnIndex.build(vectors, config)
    files = ann.save(output)
    if keep_vectors:
        np.save(output / VECTORS_FILENAME, vectors, allow_pickle=False)
        files = [VECTORS_FILENAME, *files]
    else:
        # 丢掉旧副本，否则加载端可能读到与本次索引不匹配的陈年向量。
        (output / VECTORS_FILENAME).unlink(missing_ok=True)
    if catalog_snapshot:
        snapshot = output / CATALOG_SNAPSHOT_FILENAME
        snapshot.write_text(
            "\n".join(product.model_dump_json() for product in products) + "\n", encoding="utf-8"
        )
        files = [*files, CATALOG_SNAPSHOT_FILENAME]
        catalog_digest = file_sha256(snapshot)
    else:
        (output / CATALOG_SNAPSHOT_FILENAME).unlink(missing_ok=True)
        catalog_digest = file_sha256(catalog)

    manifest = IndexManifest(
        version=2,
        model_name=model_name,
        catalog_sha256=catalog_digest,
        product_count=len(products),
        vector_dimension=int(vectors.shape[1]),
        created_at=datetime.now(timezone.utc).isoformat(),
        image_weight=image_weight,
        index_type=config.kind,
        backend=ann.backend,
        ann=config,
        index_files=files,
        has_vectors=keep_vectors,
        has_catalog_snapshot=catalog_snapshot,
        effective=ann.effective,
    )
    (output / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest, ann


def load_version_catalog(index_dir: str | Path, fallback: str | Path | None = None) -> list[Product]:
    """优先读取版本目录内的商品快照；没有快照时才回退到外部目录。

    增量更新与回滚都依赖这个顺序：索引版本必须能独立回答"我当时服务的是哪批商品"。
    """
    directory = Path(index_dir).resolve()
    snapshot = directory / CATALOG_SNAPSHOT_FILENAME
    if snapshot.exists():
        return load_catalog(snapshot)
    if fallback is not None:
        return load_catalog(Path(fallback).resolve())
    raise FileNotFoundError(f"{directory} 既没有商品快照，也没有提供外部目录")


def load_vectors_if_present(index_dir: str | Path) -> np.ndarray | None:
    """存在精确向量矩阵就加载，没有则返回 None（不抛错）。

    调用方用它给 ANN 检索配一个"精确兜底"：图索引在极端过滤下无法返回完整邻居集时，
    这份矩阵是唯一还能保证正确性的东西。没有它也不该失败，只是能力降级。
    """
    path = Path(index_dir).resolve() / VECTORS_FILENAME
    if not path.exists():
        return None
    return normalize(np.load(path, allow_pickle=False))


# ---------- 加载 ----------


def load_index(
    catalog_path: str | Path,
    index_dir: str | Path,
    expected_model: str | None = None,
) -> tuple[list[Product], np.ndarray, IndexManifest]:
    """加载精确向量矩阵（v1 行为，v2 清单同样兼容——只要它保留了 vectors.npy）。"""
    catalog = Path(catalog_path).resolve()
    index = Path(index_dir).resolve()
    manifest = read_manifest(index)
    _verify_manifest(manifest, catalog, index, expected_model)
    products = load_catalog(catalog)
    vectors = _read_vectors(index, manifest)
    if len(products) != manifest.product_count:
        raise ValueError("索引文件维度或商品数量与清单不一致")
    return products, vectors, manifest


def load_ann_index(
    catalog_path: str | Path,
    index_dir: str | Path,
    expected_model: str | None = None,
    config: AnnIndexConfig | None = None,
) -> tuple[list[Product], IndexManifest, AnnIndex]:
    """加载 ANN 索引。

    ``config`` 只用于覆盖**运行期旋钮**（HNSW 的 efSearch、IVF 的 nprobe）：
    这些参数被手工写回索引对象，因此调召回率不需要重建索引，也不改动画像。
    索引类型与维度仍以清单为准，防止"拿 HNSW 清单配一个 flat 配置"这种静默错配。
    """
    catalog = Path(catalog_path).resolve()
    index = Path(index_dir).resolve()
    manifest = read_manifest(index)
    if manifest.partition_by is not None:
        # 分段索引的每个分段各有自己的 ann.index，按"一份索引"去加载必然错配。
        # 与其抛一个溯源不出来的维度错误，不如直接说清该换哪个入口。
        raise ValueError(
            f"{index} 是 {manifest.partition_by} 分段索引（共 {manifest.partition_count} 段），"
            "请改用 PartitionedRetriever.from_index 加载"
        )
    _verify_manifest(manifest, catalog, index, expected_model)
    # 有快照就用快照：并行存在多个版本时，外部目录可能已经不是这个版本的商品了。
    products = load_version_catalog(index, fallback=catalog)
    if len(products) != manifest.product_count:
        raise ValueError("索引文件维度或商品数量与清单不一致")
    resolved = _resolve_config(manifest, config)
    if config is not None and config.kind != (manifest.ann.kind if manifest.ann else manifest.index_type):
        raise ValueError(
            f"配置的索引类型 {config.kind} 与清单记录的 {manifest.index_type} 不一致，"
            "请用同一配置重建索引（运行期只能调 efSearch / nprobe 这类旋钮）"
        )
    vectors = _read_vectors(index, manifest) if resolved.kind == "numpy" else None
    ann = AnnIndex.load(index, resolved, vectors=vectors, expected_count=manifest.product_count)
    return products, manifest, ann


__all__ = [
    "ANN_INDEX_FILENAME",
    "CATALOG_SNAPSHOT_FILENAME",
    "MANIFEST_FILENAME",
    "VECTORS_FILENAME",
    "IndexManifest",
    "build_ann_index",
    "build_index",
    "encode_products",
    "file_sha256",
    "load_ann_index",
    "load_catalog",
    "load_index",
    "load_vectors_if_present",
    "load_version_catalog",
    "read_manifest",
]
