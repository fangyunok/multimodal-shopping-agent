"""近似最近邻（ANN）索引层：把"向量怎么存、怎么查"从检索逻辑里彻底剥离。

为什么单独成层
--------------
项目早期只有 6 条商品，`product_vectors @ query_vector` 一次矩阵乘法就结束了，
既精确又足够快，FAISS 是净负担。但规模一旦上去，**精确检索的 O(N·D) 会先撞墙**，
此时需要换成近似索引；而"换索引"这件事如果渗透进检索逻辑，就会变成不可回退的改造。

所以这一层的契约只有一个：

    build(vectors, config) -> AnnIndex
    AnnIndex.search(queries, top_k) -> (scores, indices)

五种后端实现同一个契约，调用方（``FaissRetriever``）完全不知道底下是什么：

- ``numpy``    ：纯 NumPy 精确检索。零依赖、无近似损失，正确性基准 + 降级路径。
- ``flat``     ：FAISS ``IndexFlatIP``。仍然精确，只是 C++/SIMD/多线程实现。
- ``hnsw``     ：分层可导航小世界图。查询复杂度接近对数级，召回可调（efSearch）。
- ``ivf``      ：倒排文件 + 精确残差。按桶剪枝，内存不降但扫描量大幅下降。
- ``ivfpq``    ：倒排 + 乘积量化。**唯一能真正压缩内存的选项**（本例约 1/10），代价是召回损失。

设计约束
--------
1. ``faiss`` 是可选依赖，**只在真正构建/加载 FAISS 索引时才导入**。
   这样纯 CPU、不装 faiss 的环境仍然能跑通 numpy 精确路径与全部单元测试。
2. 向量一律先 L2 归一化，因此内积等价于余弦相似度——这正是选择 ``METRIC_INNER_PRODUCT`` 的原因。
3. 有效参数（effective）会被记录下来。IVF 桶数在数据量小时必须向小收缩，
   如果不把"实际用了多少桶"写进清单，压测结论就会被"配了 1024 桶其实只用了 25 桶"这种事实污染。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

IndexKind = Literal["numpy", "flat", "hnsw", "ivf", "ivfpq"]

ANN_INDEX_FILENAME = "ann.index"
FAISS_KINDS: frozenset[str] = frozenset({"flat", "hnsw", "ivf", "ivfpq"})
FAISS_MISSING_HINT = '未安装 faiss，请执行 pip install -e ".[ann]"（或 pip install faiss-cpu）'

# IVF 的桶数经验值：每桶至少要有 39 条训练样本，否则 k-means 退化成"每个点一个簇"。
MIN_POINTS_PER_CENTROID = 39


def faiss_module() -> Any:
    """惰性导入 faiss；缺失时给出可执行的安装提示而不是 ImportError 堆栈。"""
    try:
        import faiss  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - 取决于环境是否装了 faiss
        raise RuntimeError(FAISS_MISSING_HINT) from error
    return faiss


def faiss_available() -> bool:
    try:
        faiss_module()
    except RuntimeError:
        return False
    return True


def auto_pq_segments(dimension: int) -> int:
    """PQ 把向量切成 ``M`` 段，``M`` 必须整除维度。取不超过 64 的最大可行值。"""
    for candidate in (64, 32, 16, 8, 4, 2, 1):
        if dimension % candidate == 0:
            return candidate
    return 1


class AnnIndexConfig(BaseModel):
    """索引构建与运行期参数。会被完整写进索引清单，保证可复现。"""

    model_config = {"extra": "forbid"}

    kind: IndexKind = "numpy"
    dimension: int = Field(gt=0)
    metric: Literal["ip", "l2"] = "ip"
    m: int = Field(default=32, ge=2, le=128, description="HNSW 每层连接度：越大越准、越占内存")
    ef_construction: int = Field(default=200, ge=8, description="HNSW 建图宽度：越大建得越慢但图质量越好")
    ef_search: int = Field(default=64, ge=1, description="HNSW 查询宽度：直接决定召回率与延迟的取舍")
    nlist: int = Field(default=1024, ge=1, description="IVF 桶数（会在数据量小时自动收缩）")
    nprobe: int = Field(default=16, ge=1, description="查询时扫描的桶数")
    pq_segments: int | None = Field(default=None, ge=1, description="PQ 子空间数，None 表示按维度自动选择")
    pq_bits: int = Field(default=8, ge=2, le=16, description="每个子空间的编码位数")
    seed: int = 20261004

    @model_validator(mode="after")
    def _validate(self) -> "AnnIndexConfig":
        if self.metric == "l2" and self.kind != "numpy":
            raise ValueError("FAISS 后端当前只支持 ip 度量（向量已归一化，内积即余弦）")
        if self.kind == "ivfpq" and self.dimension % self.segments:
            raise ValueError(f"pq_segments({self.segments}) 必须整除向量维度({self.dimension})")
        return self

    @property
    def segments(self) -> int:
        return self.pq_segments or auto_pq_segments(self.dimension)

    @property
    def requires_faiss(self) -> bool:
        return self.kind in FAISS_KINDS

    def describe(self) -> str:
        """一行摘要，用于日志、清单和压测报告。"""
        if self.kind == "numpy":
            return "numpy 精确检索"
        if self.kind == "flat":
            return "faiss IndexFlatIP 精确检索"
        if self.kind == "hnsw":
            return f"faiss HNSW(M={self.m}, efC={self.ef_construction}, efS={self.ef_search})"
        if self.kind == "ivf":
            return f"faiss IVF(nlist={self.nlist}, nprobe={self.nprobe})"
        return f"faiss IVF-PQ(nlist={self.nlist}, nprobe={self.nprobe}, M={self.segments}, nbits={self.pq_bits})"


class _NumpyHandle:
    """纯 NumPy 精确检索。既是正确性基准，也是 faiss 不可用时的降级路径。"""

    kind = "numpy"

    def __init__(self, vectors: np.ndarray, metric: str = "ip") -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.metric = metric
        self._squared_norms = (self.vectors * self.vectors).sum(axis=1)

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if self.metric == "ip":
            scores = queries @ self.vectors.T
        else:
            # 展开 ||q - v||^2，取负号让"分数越大越近"的统一约定成立。
            scores = -(
                (queries * queries).sum(axis=1)[:, None]
                - 2.0 * (queries @ self.vectors.T)
                + self._squared_norms[None, :]
            )
        count = self.vectors.shape[0]
        k = int(min(top_k, count))
        if k <= 0:
            empty = np.empty((queries.shape[0], 0), dtype=np.float32)
            return empty, np.empty((queries.shape[0], 0), dtype=np.int64)
        # argpartition 是 O(N) 的部分排序；只有入选的 k 条才需要真正排序。
        partitioned = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        taken = np.take_along_axis(scores, partitioned, axis=1)
        order = np.argsort(-taken, axis=1, kind="stable")
        indices = np.take_along_axis(partitioned, order, axis=1)
        values = np.take_along_axis(taken, order, axis=1)
        return values.astype(np.float32), indices.astype(np.int64)

    @property
    def index_bytes(self) -> int:
        return int(self.vectors.nbytes)


class _FaissHandle:
    """FAISS 后端薄封装：只负责建索引、查索引、存盘，不掺杂任何业务语义。"""

    def __init__(self, index: Any, config: AnnIndexConfig, effective: dict[str, Any]) -> None:
        self.index = index
        self.config = config
        self.effective = effective

    @property
    def kind(self) -> str:
        return self.config.kind

    @classmethod
    def build(cls, vectors: np.ndarray, config: AnnIndexConfig) -> "_FaissHandle":
        faiss = faiss_module()
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        dimension = int(vectors.shape[1])
        count = int(vectors.shape[0])
        if dimension != config.dimension:
            raise ValueError(f"向量维度 {dimension} 与配置 {config.dimension} 不一致")

        if config.kind == "flat":
            index = faiss.IndexFlatIP(dimension)
            index.add(vectors)
            return cls(index, config, {"ntotal": count})

        if config.kind == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, config.m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = config.ef_construction
            index.hnsw.efSearch = config.ef_search
            index.add(vectors)
            return cls(
                index,
                config,
                {"m": config.m, "ef_construction": config.ef_construction, "ef_search": config.ef_search},
            )

        # ---- IVF 家族：先训练粗量化器，再按残差建桶 ----
        if config.kind == "ivfpq":
            # PQ 的每个子空间要训 2^bits 个码本中心，faiss 的经验下限是每中心 39 条样本。
            # 样本不足时码本接近随机，recall 会被严重低估——那不是"索引不行"，
            # 而是"用错了规模"，所以这里直接拒绝而不是让它跑出一个假的低分。
            recommended = MIN_POINTS_PER_CENTROID * (2**config.pq_bits)
            if count < recommended:
                raise ValueError(
                    f"IVF-PQ 的 PQ 码本训练需要约 {recommended} 条向量"
                    f"（39 × 2^{config.pq_bits}），当前只有 {count} 条。"
                    "样本不足时召回率会被严重低估，请改用 ivf / hnsw，或先把训练集做大。"
                )
        nlist = max(1, min(config.nlist, count // MIN_POINTS_PER_CENTROID))
        quantizer = faiss.IndexFlatIP(dimension)
        if config.kind == "ivf":
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexIVFPQ(
                quantizer, dimension, nlist, config.segments, config.pq_bits, faiss.METRIC_INNER_PRODUCT
            )
        index.train(vectors)
        index.add(vectors)
        nprobe = max(1, min(config.nprobe, nlist))
        index.nprobe = nprobe
        effective: dict[str, Any] = {"nlist": nlist, "nprobe": nprobe, "nlist_configured": config.nlist}
        if config.kind == "ivfpq":
            effective.update({"pq_segments": config.segments, "pq_bits": config.pq_bits})
        return cls(index, config, effective)

    def apply_runtime_params(self) -> None:
        """把配置里的运行期旋钮重新压回索引对象。

        nprobe 与 efSearch 都被 faiss 序列化，所以它们可以**不重建索引就调整**——
        这正是"召回率不够时先调 efSearch，而不是重新建库"的依据。
        """
        if self.config.kind == "hnsw":
            self.index.hnsw.efSearch = self.config.ef_search
        elif self.config.kind in {"ivf", "ivfpq"}:
            self.index.nprobe = max(1, min(self.config.nprobe, int(self.index.nlist)))

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        scores, indices = self.index.search(queries, int(top_k))
        return scores.astype(np.float32), indices.astype(np.int64)

    @property
    def index_bytes(self) -> int:
        faiss = faiss_module()
        return int(len(faiss.serialize_index(self.index)))

    def save(self, directory: Path) -> list[str]:
        faiss = faiss_module()
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / ANN_INDEX_FILENAME))
        return [ANN_INDEX_FILENAME]


class AnnIndex:
    """统一门面：按 ``config.kind`` 选择 NumPy 精确实现或 FAISS 实现。"""

    def __init__(self, config: AnnIndexConfig, handle: Any, count: int, effective: dict[str, Any]) -> None:
        self.config = config
        self.handle = handle
        self.count = int(count)
        self.effective = effective

    @property
    def backend(self) -> str:
        return "numpy" if self.config.kind == "numpy" else "faiss"

    @property
    def index_bytes(self) -> int:
        return self.handle.index_bytes

    @property
    def size_mb(self) -> float:
        return round(self.index_bytes / (1024 * 1024), 3)

    @classmethod
    def build(cls, vectors: np.ndarray, config: AnnIndexConfig) -> "AnnIndex":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors 必须是二维数组 (N, D)")
        if vectors.shape[1] != config.dimension:
            raise ValueError(f"向量维度 {vectors.shape[1]} 与配置 {config.dimension} 不一致")
        if vectors.shape[0] == 0:
            raise ValueError("不能对空向量集合建索引")
        if config.kind == "numpy":
            handle: Any = _NumpyHandle(vectors, config.metric)
            return cls(config, handle, vectors.shape[0], {"ntotal": int(vectors.shape[0])})
        handle = _FaissHandle.build(vectors, config)
        return cls(config, handle, int(handle.index.ntotal), dict(handle.effective))

    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        config: AnnIndexConfig,
        vectors: np.ndarray | None = None,
        expected_count: int | None = None,
    ) -> "AnnIndex":
        directory = Path(index_dir).resolve()
        if config.kind == "numpy":
            if vectors is None:
                raise ValueError("numpy 后端需要 vectors.npy 中的向量，请确认索引目录包含该文件")
            handle: Any = _NumpyHandle(np.asarray(vectors, dtype=np.float32), config.metric)
            return cls(config, handle, int(handle.vectors.shape[0]), {"ntotal": int(handle.vectors.shape[0])})

        path = directory / ANN_INDEX_FILENAME
        if not path.exists():
            raise FileNotFoundError(f"未找到 ANN 索引文件 {path}，请先用 build_ann_index 构建")
        faiss = faiss_module()
        index = faiss.read_index(str(path))
        if int(index.d) != config.dimension:
            raise ValueError(f"索引维度 {index.d} 与配置 {config.dimension} 不一致")
        if expected_count is not None and int(index.ntotal) != expected_count:
            raise ValueError(f"索引条目数 {index.ntotal} 与清单记录 {expected_count} 不一致")
        handle = _FaissHandle(index, config, {})
        handle.apply_runtime_params()
        return cls(config, handle, int(index.ntotal), dict(handle.effective))

    def search(self, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.handle.search(queries, top_k)

    def search_one(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores, indices = self.search(np.asarray([query]), top_k)
        return scores[0], indices[0]

    def save(self, index_dir: str | Path) -> list[str]:
        """写盘并返回本次真正新增的文件名列表。

        ``numpy`` 后端返回空列表：向量本来就存在 ``vectors.npy`` 里，重复写一份没有意义。
        """
        if self.config.kind == "numpy":
            return []
        return self.handle.save(Path(index_dir).resolve())

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.config.kind,
            "backend": self.backend,
            "count": self.count,
            "dimension": self.config.dimension,
            "size_mb": self.size_mb,
            "bytes_per_vector": round(self.index_bytes / max(self.count, 1), 1),
            "effective": self.effective,
        }
