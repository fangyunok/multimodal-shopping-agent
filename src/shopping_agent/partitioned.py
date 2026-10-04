"""过滤下推：把过滤维度做进索引结构，而不是在检索结果上再筛一遍。

问题：超采样解决不了"过滤强度高"这一类查询
------------------------------------------
``FaissRetriever`` 用自适应超采样缓解过滤，但**缓解不等于解决**。实测 10 万条 512 维、
价格放行率仅 1% 时，与精确后端的结果集一致率是：IVF 0.78、IVF-PQ 0.30。

根因不在参数上：价格过滤与向量相似度**本来就无关**，符合条件的前 10 个商品在向量空间里
位置是随机的。而任何只扫描部分子空间的近似索引（IVF 扫若干桶、HNSW 走若干层）都可能
**压根没访问到它们**——候选池里没有的东西，再怎么重排也变不出来。调 nprobe 只是把
"扫多少桶"往上推，推到全扫就是精确检索，加速比归零。

所以正解不是继续调参，而是**让过滤发生在检索之前**：

    按价位分段建索引 → 查询时先用 max_price 选段 → 只在选中的段内做 ANN 检索

过滤掉的分段**根本不会被查询**。ANN 的候选预算全部花在合格商品上，
"价格过滤吃掉 99% 候选"这件事从源头消失了。

契约：分段必须是一个划分
------------------------
分段必须两两不交、并集为全集。否则要么漏商品（有商品不属于任何段），要么重复
（同一商品在两个段里被返回两次）。构造时校验，不靠约定。

路由判断必须"宁滥勿缺"
----------------------
``PartitionSpec.may_match`` 判 True 最多多扫一个分段（慢一点），判 False 会**永久漏掉商品**。
所以价格用段内**最低价**作下界——这是精确的：段内最便宜的商品都超预算，这个段才真没戏。

代价：选择性越低收益越小
------------------------
``max_price`` 高过全库最高价时所有分段都会被选中，此时比单索引还慢（多了一层合并）。
所以这是**过滤越狠收益越大**的手段，不是无脑开。注意它解决的是"过滤与向量无关"，
如果过滤条件本身与向量相关（例如"用户偏好的价位带"），分段边界就能与语义对齐，收益更大。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .ann_index import AnnIndex, AnnIndexConfig
from .encoders import MultimodalEncoder, normalize
from .faiss_retrieval import FaissRetriever, encode_request
from .indexing import (
    CATALOG_SNAPSHOT_FILENAME,
    MANIFEST_FILENAME,
    VECTORS_FILENAME,
    IndexManifest,
    file_sha256,
    load_version_catalog,
    read_manifest,
)
from .models import Product, SearchHit, SearchRequest

PartitionBy = Literal["price", "category", "category_price"]

PARTITION_MANIFEST_FILENAME = "partitions.json"
PARTITION_MEMBERS_FILENAME = "members.npy"
PARTITION_DIR_PREFIX = "part-"

#: 区分"调用方没传 exact_vectors"与"调用方明确传了 None"。
#: 前者默认保留向量，后者是明确放弃精确兜底——两者必须走不同的分支。
_UNSET: object = object()


class PartitionSpec(BaseModel):
    """一个分段的元数据。够用来做路由判断，也够用来在加载时重建分段。"""

    model_config = {"extra": "forbid"}

    name: str
    directory: str
    size: int = Field(ge=0)
    price_low: float
    price_high: float
    categories: tuple[str, ...] = ()

    def may_match(self, request: SearchRequest) -> bool:
        """这个分段是否**可能**含有满足过滤条件的商品。

        宁滥勿缺：判 True 只是多扫一个分段，判 False 会永久漏掉商品。
        """
        if request.max_price is not None and self.price_low > request.max_price:
            return False
        if request.category:
            needle = request.category.lower()
            if not any(needle in category for category in self.categories):
                return False
        return True


class PartitionLayout(BaseModel):
    """分段索引的布局清单，存成 ``partitions.json``。

    与 ``manifest.json`` 的分工：``manifest.json`` 回答"这是什么索引、怎么加载"，
    布局清单回答"切成几段、每段是什么、商品怎么映射回全局下标"。
    """

    model_config = {"extra": "forbid"}

    version: int = 1
    by: PartitionBy
    buckets: int = Field(ge=1)
    dimension: int = Field(gt=0)
    product_count: int = Field(ge=0)
    image_weight: float = 0.5
    model_name: str = "unnamed"
    ann: AnnIndexConfig
    has_vectors: bool = True
    partitions: list[PartitionSpec] = Field(default_factory=list)


@dataclass
class PartitionEntry:
    """运行期的一个分段：元数据 + 它在全局目录中的下标 + 它自己的检索器。"""

    spec: PartitionSpec
    members: np.ndarray
    retriever: FaissRetriever


# ---------- 分段规划 ----------


def _price_groups(prices: np.ndarray, indices: np.ndarray, buckets: int) -> list[np.ndarray]:
    """按**分位数**切桶，不是等宽切。

    商品价格是长尾的：等宽切会让绝大多数商品挤进第一个桶，而后面几个桶近乎空置——
    分段就失去了分摊的意义。分位数保证每段商品数相近。
    """
    if buckets <= 1 or indices.size <= 1:
        return [indices]
    values = prices[indices]
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, buckets + 1)))
    internal = edges[1:-1]
    if internal.size == 0:  # 价格全相同，切不动
        return [indices]
    assignment = np.searchsorted(internal, values, side="right")
    return [indices[assignment == bucket] for bucket in np.unique(assignment)]


def _category_groups(products: list[Product]) -> list[tuple[str, np.ndarray]]:
    rows: dict[str, list[int]] = {}
    for position, product in enumerate(products):
        # 统一小写：路由匹配与 FaissRetriever 的过滤语义（同样是 lower + 子串）必须一致。
        rows.setdefault(product.category.lower(), []).append(position)
    return [(name, np.asarray(positions, dtype=np.int64)) for name, positions in sorted(rows.items())]


def plan_partitions(
    products: list[Product], by: PartitionBy = "price", buckets: int = 16
) -> list[tuple[PartitionSpec, list[int]]]:
    """把目录切成若干分段。返回 ``(元数据, 全局下标列表)``，且保证是一个划分。"""
    if not products:
        raise ValueError("商品目录不能为空")
    if buckets < 1:
        raise ValueError("buckets 必须大于等于 1")
    prices = np.asarray([product.price for product in products], dtype=np.float64)
    everything = np.arange(len(products), dtype=np.int64)

    if by == "price":
        groups: list[tuple[str, np.ndarray]] = [("price", rows) for rows in _price_groups(prices, everything, buckets)]
    elif by == "category":
        groups = _category_groups(products)
    else:
        groups = []
        for name, rows in _category_groups(products):
            for sub in _price_groups(prices, rows, buckets):
                groups.append((name, sub))

    planned: list[tuple[PartitionSpec, list[int]]] = []
    for order, (label, rows) in enumerate(groups):
        if rows.size == 0:
            continue
        members = [int(row) for row in rows]
        spec = PartitionSpec(
            name=f"{label}#{order:04d}",
            directory=f"{PARTITION_DIR_PREFIX}{order:04d}",
            size=len(members),
            # 段内最低价是路由下界：它精确，所以"跳过这个段"永远不会是错的。
            price_low=float(prices[rows].min()),
            price_high=float(prices[rows].max()),
            categories=tuple(sorted({products[row].category.lower() for row in members})),
        )
        planned.append((spec, members))
    return planned


# ---------- 检索器 ----------


class PartitionedRetriever:
    """多索引路由：一份全局索引换成 N 份分段索引，查询时先按过滤条件选段。

    对外接口与 ``FaissRetriever`` 一致（``search(request)`` / ``describe()``），
    所以 ``ShoppingTools``、MCP 层、FastAPI 层都不需要知道底下是分段索引。
    """

    def __init__(
        self,
        products: list[Product],
        catalog_path: Path,
        encoder: MultimodalEncoder,
        layout: PartitionLayout,
        entries: list[PartitionEntry],
        image_weight: float = 0.5,
    ) -> None:
        if not entries:
            raise ValueError("分段索引至少要有一个分段")
        covered = np.sort(np.concatenate([entry.members for entry in entries]))
        if covered.shape[0] != len(products) or not np.array_equal(covered, np.arange(len(products))):
            raise ValueError("分段必须构成商品目录的一个划分（两两不交、并集为全集）")
        self.products = products
        self.catalog_path = catalog_path
        self.encoder = encoder
        self.layout = layout
        self.entries = entries
        self.image_weight = image_weight
        self.last_stats: dict[str, float | int | str | bool] = {}

    # ---------- 构造 ----------

    @classmethod
    def from_vectors(
        cls,
        products: list[Product],
        catalog_path: str | Path,
        encoder: MultimodalEncoder,
        vectors: np.ndarray,
        config: AnnIndexConfig,
        by: PartitionBy = "price",
        buckets: int = 16,
        image_weight: float = 0.5,
        model_name: str = "unnamed",
        **kwargs,
    ) -> "PartitionedRetriever":
        """用同一批向量建出 N 份分段索引。

        每个分段都是一份**独立**的 ANN 索引，因此它的 ``nlist`` 会按段内数据量自行收缩
        （``AnnIndex`` 会记录实际生效值）。这正是分段的隐藏代价：段切得越细，
        每段的训练样本越少，IVF/PQ 这类需要训练的后端就越难建——IVF-PQ 有个硬门槛
        （每个 PQ 码本中心 39 条样本），段太小会被直接拒绝。

        ``exact_vectors`` 在这里按**全库**语义理解：传全库向量就自动切片，传 ``None``
        就明确放弃精确兜底，不传则默认每段保留自己的向量。分段对使用方是透明的，
        没有理由要求调用方自己先把向量切好——那既多余，又极易切错段。
        """
        catalog = Path(catalog_path).resolve()
        normalized = normalize(np.asarray(vectors, dtype=np.float32))
        if normalized.shape[0] != len(products):
            raise ValueError(f"向量条数 {normalized.shape[0]} 与商品数 {len(products)} 不一致")
        if config.dimension != int(normalized.shape[1]):
            config = config.model_copy(update={"dimension": int(normalized.shape[1])})

        fallback = kwargs.pop("exact_vectors", _UNSET)
        if fallback is not _UNSET and fallback is not None:
            supplied = np.asarray(fallback, dtype=np.float32)
            if supplied.shape[0] != normalized.shape[0]:
                raise ValueError(
                    f"exact_vectors 有 {supplied.shape[0]} 条，但商品有 {len(products)} 条；"
                    "请按全库向量传入，分段会在内部自行切片"
                )

        entries: list[PartitionEntry] = []
        plan = plan_partitions(products, by, buckets)
        for spec, members in plan:
            rows = np.asarray(members, dtype=np.int64)
            subset_vectors = normalized[rows]
            # 每轮都要拷贝一份 kwargs：直接复用同一个 dict 会让后续分段拿到别的分段的参数。
            extra = dict(kwargs)
            if fallback is _UNSET or fallback is None:
                # 不传 → 默认保留本段向量；显式传 None → 尊重"不要兜底"的意图。
                extra["exact_vectors"] = subset_vectors if fallback is _UNSET else None
            else:
                extra["exact_vectors"] = np.asarray(fallback, dtype=np.float32)[rows]
            retriever = FaissRetriever(
                [products[row] for row in members],
                catalog,
                encoder,
                AnnIndex.build(subset_vectors, config),
                image_weight=image_weight,
                **extra,
            )
            entries.append(PartitionEntry(spec=spec, members=rows, retriever=retriever))

        layout = PartitionLayout(
            by=by,
            buckets=buckets,
            dimension=int(normalized.shape[1]),
            product_count=len(products),
            image_weight=image_weight,
            model_name=model_name,
            ann=config,
            has_vectors=fallback is _UNSET or fallback is not None,
            partitions=[spec for spec, _ in plan],
        )
        return cls(products, catalog, encoder, layout, entries, image_weight=image_weight)

    @classmethod
    def from_index(
        cls,
        catalog_path: str | Path,
        index_dir: str | Path,
        encoder: MultimodalEncoder,
        config: AnnIndexConfig | None = None,
        **kwargs,
    ) -> "PartitionedRetriever":
        """从磁盘加载分段索引。

        ``config`` 只能覆盖运行期旋钮（efSearch / nprobe），与单索引路径的规则一致：
        索引用什么建的，加载时不能悄悄换掉。同理，分段保不保留精确向量由构建时的
        ``has_vectors`` 决定，加载时不能另给一份——否则"这个索引能不能重排"就有了
        两个来源，早晚会对不上。
        """
        if kwargs.pop("exact_vectors", None) is not None:
            raise ValueError("exact_vectors 是构建期参数，加载分段索引时不能另传（由布局的 has_vectors 决定）")
        directory = Path(index_dir).resolve()
        catalog = Path(catalog_path).resolve()
        manifest = read_manifest(directory)
        if manifest.partition_by is None:
            raise ValueError(f"{directory} 不是分段索引（manifest 未记录 partition_by），请改用 FaissRetriever.from_index")
        layout_path = directory / PARTITION_MANIFEST_FILENAME
        if not layout_path.exists():
            raise FileNotFoundError(f"分段索引缺少布局清单 {layout_path}")
        layout = PartitionLayout.model_validate_json(layout_path.read_text(encoding="utf-8"))
        # 与单索引路径同样的配对校验：索引和商品目录必须是一起构建的那一对，
        # 否则分段下标会指向另一批商品——分段索引的下标是"全局目录下标"，错配会静默返回错商品。
        snapshot = directory / CATALOG_SNAPSHOT_FILENAME
        reference = snapshot if snapshot.exists() else catalog
        if reference.exists() and manifest.catalog_sha256 != file_sha256(reference):
            raise ValueError("商品目录已变化，请重新构建向量索引")
        products = load_version_catalog(directory, fallback=catalog)
        if len(products) != layout.product_count:
            raise ValueError(f"索引服务 {layout.product_count} 条商品，但目录有 {len(products)} 条")

        resolved = config or layout.ann
        if resolved.dimension != layout.dimension:
            resolved = resolved.model_copy(update={"dimension": layout.dimension})
        if resolved.kind != layout.ann.kind:
            raise ValueError(
                f"配置的索引类型 {resolved.kind} 与布局记录的 {layout.ann.kind} 不一致，"
                "运行期只能调 efSearch / nprobe 这类旋钮"
            )

        entries: list[PartitionEntry] = []
        for spec in layout.partitions:
            part_dir = directory / spec.directory
            if not part_dir.exists():
                raise FileNotFoundError(f"分段目录缺失：{part_dir}")
            members = np.load(part_dir / PARTITION_MEMBERS_FILENAME, allow_pickle=False).astype(np.int64)
            if members.shape[0] != spec.size:
                raise ValueError(f"分段 {spec.name} 的下标数 {members.shape[0]} 与布局记录 {spec.size} 不一致")
            subset_vectors = None
            if layout.has_vectors:
                subset_vectors = normalize(np.load(part_dir / VECTORS_FILENAME, allow_pickle=False))
            ann = AnnIndex.load(
                part_dir,
                resolved,
                vectors=subset_vectors if resolved.kind == "numpy" else None,
                expected_count=spec.size,
            )
            extra = dict(kwargs)
            # 显式赋值而非 setdefault：向量由磁盘上的布局决定，不该被外部 kwargs 覆盖。
            extra["exact_vectors"] = subset_vectors
            retriever = FaissRetriever(
                [products[int(row)] for row in members],
                catalog,
                encoder,
                ann,
                image_weight=layout.image_weight,
                **extra,
            )
            entries.append(PartitionEntry(spec=spec, members=members, retriever=retriever))
        return cls(products, catalog, encoder, layout, entries, image_weight=layout.image_weight)

    # ---------- 检索 ----------

    def search(self, request: SearchRequest) -> list[SearchHit]:
        # 只编码一次：分段数一多，重复的模型前向会比检索本身还贵。
        query_vector = encode_request(self.encoder, request, self.layout.dimension)
        selected = [entry for entry in self.entries if entry.spec.may_match(request)]

        if query_vector is None:
            # 没有语义条件时不必绕道向量：各段内部按评分排序即可。
            batches = [entry.retriever.search(request) for entry in selected]
        else:
            batches = [entry.retriever.search_vector(query_vector, request) for entry in selected]

        merged = [hit for hits in batches for hit in hits]
        # 每段各返回自己的 top_k，取并集后再截断——全局 top_k 必然落在并集里，所以这是精确的。
        merged.sort(key=lambda hit: (hit.score, hit.product.rating), reverse=True)

        scanned = sum(entry.spec.size for entry in selected)
        self.last_stats = {
            "mode": "partitioned",
            "by": self.layout.by,
            "partitions_total": len(self.entries),
            "partitions_scanned": len(selected),
            "partitions_pruned": len(self.entries) - len(selected),
            "scanned_products": scanned,
            # 这个比例就是"过滤下推省掉了多少检索量"的直接读数。
            "scanned_fraction": round(scanned / max(self.layout.product_count, 1), 4),
            "candidates": len(merged),
            # 段内仍会各自超采样，所以轮数取各段的最大值：它才是这次查询的真实最坏开销。
            "rounds": max((int(entry.retriever.last_stats.get("rounds", 0)) for entry in selected), default=0),
            "examined": sum(int(entry.retriever.last_stats.get("examined", 0)) for entry in selected),
        }
        return merged[: request.top_k]

    # ---------- 落盘 ----------

    def save(self, index_dir: str | Path, catalog_snapshot: bool = False) -> list[str]:
        """写出分段索引，同时写一份标准 ``manifest.json``。

        为什么要写标准清单：索引的发布/回滚靠 ``IndexVersionManager``，而它发布前会
        校验 ``manifest.json`` 与声明的文件是否齐备。分段索引如果不写这份清单，
        就会被排除在原子发布链路之外——那等于把一个只能手工替换目录的索引交出去。
        """
        directory = Path(index_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        for entry in self.entries:
            part_dir = directory / entry.spec.directory
            part_dir.mkdir(parents=True, exist_ok=True)
            entry.retriever.ann.save(part_dir)
            np.save(part_dir / PARTITION_MEMBERS_FILENAME, entry.members.astype(np.int64), allow_pickle=False)
            if self.layout.has_vectors:
                vectors = entry.retriever.exact_vectors
                if vectors is None:
                    raise ValueError(f"布局声明保留向量，但分段 {entry.spec.name} 没有精确向量")
                np.save(part_dir / VECTORS_FILENAME, vectors, allow_pickle=False)
            else:
                (part_dir / VECTORS_FILENAME).unlink(missing_ok=True)
        (directory / PARTITION_MANIFEST_FILENAME).write_text(
            self.layout.model_dump_json(indent=2), encoding="utf-8"
        )

        files = [PARTITION_MANIFEST_FILENAME, *[entry.spec.directory for entry in self.entries]]
        snapshot = directory / CATALOG_SNAPSHOT_FILENAME
        if catalog_snapshot:
            snapshot.write_text(
                "\n".join(product.model_dump_json() for product in self.products) + "\n", encoding="utf-8"
            )
            files.append(CATALOG_SNAPSHOT_FILENAME)
            catalog_digest = file_sha256(snapshot)
        else:
            snapshot.unlink(missing_ok=True)
            catalog_digest = file_sha256(self.catalog_path)

        manifest = IndexManifest(
            version=2,
            model_name=self.layout.model_name,
            catalog_sha256=catalog_digest,
            product_count=self.layout.product_count,
            vector_dimension=self.layout.dimension,
            created_at=datetime.now(timezone.utc).isoformat(),
            image_weight=self.layout.image_weight,
            index_type="partitioned",
            backend="faiss" if self.layout.ann.requires_faiss else "numpy",
            ann=self.layout.ann,
            index_files=files,
            has_vectors=self.layout.has_vectors,
            has_catalog_snapshot=catalog_snapshot,
            partition_by=self.layout.by,
            partition_count=len(self.entries),
            # 每段的实际生效参数都要留下：段大小不同，nlist 的收缩程度也不同，
            # 只记一份"配置值"会让复现时对不上号。
            effective={
                "partitions": len(self.entries),
                "sizes": [entry.spec.size for entry in self.entries],
                "ann_effective": [entry.retriever.ann.effective for entry in self.entries],
            },
        )
        (directory / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return files

    # ---------- 观测 ----------

    def describe(self) -> dict[str, object]:
        sizes = [entry.spec.size for entry in self.entries]
        return {
            "kind": "partitioned",
            "backend": "faiss" if self.layout.ann.requires_faiss else "numpy",
            "ann": self.layout.ann.describe(),
            "by": self.layout.by,
            "buckets": self.layout.buckets,
            "partitions": len(self.entries),
            "count": self.layout.product_count,
            "dimension": self.layout.dimension,
            "partition_min": min(sizes),
            "partition_max": max(sizes),
            "size_mb": round(sum(entry.retriever.ann.index_bytes for entry in self.entries) / (1024 * 1024), 3),
            "scanned_fraction": self.last_stats.get("scanned_fraction"),
            "last_stats": dict(self.last_stats),
        }


__all__ = [
    "PARTITION_DIR_PREFIX",
    "PARTITION_MANIFEST_FILENAME",
    "PARTITION_MEMBERS_FILENAME",
    "PartitionEntry",
    "PartitionLayout",
    "PartitionSpec",
    "PartitionedRetriever",
    "plan_partitions",
]
