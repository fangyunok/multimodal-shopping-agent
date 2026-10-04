"""合成语料生成器：让"规模化"从一句话变成一组可复现的实测数字。

为什么需要它
------------
仓库里的真实目录只有个位数商品。这带来一个结构性盲区：**所有关于"规模上去了会怎样"的
判断都无法被验证**，只能停留在"理论上应该换 ANN"。而 ANN 改造最危险的恰恰是那些
只在大规模下才暴露的问题——近似召回损失、过滤吃掉候选、内存超限、桶数配置失真。

本模块提供两样东西：

- ``generate_corpus``：确定性（同 seed 同结果）生成 N 条带簇结构的归一化向量，
  外加 Q 条查询向量与它们的**精确 top-k 真值**。有了真值，ANN 的召回率才能被量化，
  而不是只能报"延迟降低了"。
- ``SyntheticEncoder``：把 ``syn0001234`` 这类 token 直接映射回对应行。
  这样整条检索链路（编码 → ANN → 过滤 → 排序）可以在**没有 GPU、没有模型**的情况下
  被端到端压测，而且每条查询都有确定性的标准答案。

为什么向量要带簇结构
--------------------
均匀随机的 512 维向量在近邻意义上近似"彼此等距"，所有索引都表现得一样好，
压测结果会虚高。真实 embedding 是各向异性且成簇的（同类商品挤在一起），
HNSW 的 efSearch 与 IVF 的 nprobe 才会体现出真实取舍。所以这里用高斯混合：
先取 ``clusters`` 个簇心，再在簇心附近加噪声。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ann_index import AnnIndex, AnnIndexConfig
from .encoders import normalize
from .models import Product

PRODUCT_TOKEN_PATTERN = re.compile(r"syn(\d{7,})")

CATEGORIES = ("运动鞋", "箱包", "连衣裙", "耳机", "手表", "水杯", "台灯", "键盘", "外套", "背包")
COLORS = ("白色", "黑色", "灰色", "米色", "藏青", "酒红", "雾蓝", "橄榄绿")
STYLES = ("简约", "复古", "通勤", "户外", "轻奢", "运动")
MATERIALS = ("牛皮", "帆布", "网面", "棉麻", "铝合金", "羊毛")
SCENES = ("通勤", "旅行", "办公", "健身", "居家", "约会")


@dataclass
class SyntheticCorpus:
    """一批合成语料：向量、查询、精确真值、以及配套的商品元数据。"""

    vectors: np.ndarray
    queries: np.ndarray
    ground_truth: np.ndarray
    products: list[Product] = field(default_factory=list)
    seed: int = 0
    clusters: int = 0
    spread: float = 0.0

    @property
    def count(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def recall(self, indices: np.ndarray, top_k: int) -> float:
        """ANN 返回的 ``indices`` 相对精确真值的 recall@k。"""
        if self.ground_truth.size == 0:
            return float("nan")
        hits = 0
        total = 0
        for row, truth in zip(indices, self.ground_truth):
            truth_set = {int(item) for item in truth[:top_k] if item >= 0}
            if not truth_set:
                continue
            total += 1
            hits += len(truth_set & {int(item) for item in row[:top_k] if item >= 0}) / len(truth_set)
        return hits / total if total else float("nan")

    def summarize(self) -> dict[str, object]:
        return {
            "vectors": self.count,
            "dimension": self.dimension,
            "queries": int(self.queries.shape[0]),
            "clusters": self.clusters,
            "spread": self.spread,
            "seed": self.seed,
            "raw_vectors_mb": round(self.vectors.nbytes / (1024 * 1024), 3),
        }


def _chunked(count: int, size: int):
    start = 0
    while start < count:
        yield start, min(start + size, count)
        start += size


def _draw(rows: int, dimension: int, centers: np.ndarray, spread: float, rng: np.random.Generator) -> np.ndarray:
    """从高斯混合里抽 ``rows`` 条向量。分块生成，避免百万级时出现数 GB 的临时数组。

    ``spread`` 是**簇内散开的相对幅度**，语义是 ``point = normalize(center + spread × 随机方向)``。

    这里刻意不用「高斯噪声 × spread」：那样噪声范数会随维度增长（512 维下是 √512 × spread），
    稍微一调就把簇心冲没了，数据退化成"彼此等距"——所有索引看起来一样好或一样坏，
    压测结果失去意义。除以噪声自身范数后，``spread`` 就与维度无关了：
    spread=0（全重合）、spread≈0.5（簇内相似度约 0.8、跨簇接近 0）、spread→∞（纯随机）。
    """
    out = np.empty((rows, dimension), dtype=np.float32)
    cluster_count = centers.shape[0]
    for start, end in _chunked(rows, 100_000):
        block = end - start
        assignment = rng.integers(0, cluster_count, size=block)
        noise = rng.normal(size=(block, dimension)).astype(np.float32)
        noise /= np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
        out[start:end] = centers[assignment] + spread * noise
    return normalize(out)


def generate_corpus(
    count: int,
    dimension: int = 512,
    queries: int = 200,
    top_k: int = 10,
    clusters: int = 128,
    spread: float = 0.5,
    seed: int = 20261004,
    with_products: bool = True,
) -> SyntheticCorpus:
    """生成合成语料与精确 top-k 真值。

    ``spread`` 越大簇内越松散（近邻越难分），是调节"检索难度"的旋钮。
    """
    if count < 1:
        raise ValueError("count 必须大于 0")
    if dimension < 1:
        raise ValueError("dimension 必须大于 0")
    if clusters < 1:
        raise ValueError("clusters 必须大于 0")
    if spread < 0:
        raise ValueError("spread 不能为负")
    rng = np.random.default_rng(seed)
    centers = normalize(rng.normal(size=(clusters, dimension)).astype(np.float32))
    vectors = _draw(count, dimension, centers, spread, rng)
    query_vectors = _draw(queries, dimension, centers, spread * 0.6, rng)

    # 精确真值：逐块算，避免 (Q, N) 的分数矩阵在百万级直接吃掉内存。
    truth_rows: list[np.ndarray] = []
    exact = AnnIndex.build(vectors, AnnIndexConfig(kind="numpy", dimension=dimension))
    for start, end in _chunked(queries, 8):
        _, indices = exact.search(query_vectors[start:end], min(top_k, count))
        truth_rows.append(indices)
    ground_truth = np.concatenate(truth_rows, axis=0) if truth_rows else np.empty((0, 0), dtype=np.int64)

    products = _build_products(count, rng) if with_products else []
    return SyntheticCorpus(
        vectors=vectors,
        queries=query_vectors,
        ground_truth=ground_truth,
        products=products,
        seed=seed,
        clusters=clusters,
        spread=spread,
    )


def _build_products(count: int, rng: np.random.Generator) -> list[Product]:
    categories = np.asarray(CATEGORIES)
    colors = np.asarray(COLORS)
    styles = np.asarray(STYLES)
    materials = np.asarray(MATERIALS)
    scenes = np.asarray(SCENES)
    # 价格取对数正态：真实电商价格分布长尾，用均匀分布会让"按预算过滤"的压力失真。
    prices = np.round(np.exp(rng.normal(6.0, 0.8, size=count)), 2)
    ratings = np.round(rng.uniform(3.5, 5.0, size=count), 1)
    stocks = rng.integers(0, 200, size=count)
    picks = {
        "category": rng.integers(0, len(categories), size=count),
        "color": rng.integers(0, len(colors), size=count),
        "style": rng.integers(0, len(styles), size=count),
        "material": rng.integers(0, len(materials), size=count),
        "scene": rng.integers(0, len(scenes), size=count),
    }
    products: list[Product] = []
    for row in range(count):
        token = f"syn{row:07d}"
        category = str(categories[picks["category"][row]])
        color = str(colors[picks["color"][row]])
        style = str(styles[picks["style"][row]])
        material = str(materials[picks["material"][row]])
        scene = str(scenes[picks["scene"][row]])
        products.append(
            Product(
                id=token,
                # token 出现在标题里，SyntheticEncoder 才能把文本查询映射回同一行向量。
                title=f"{color}{style}{category} {token}",
                category=category,
                description=f"{style}设计，{material}材质，适合{scene}场景的{category}",
                price=float(prices[row]),
                attributes={"颜色": color, "风格": style, "材质": material, "场景": scene},
                rating=float(ratings[row]),
                stock=int(stocks[row]),
                source="synthetic",
                license="CC0-1.0",
            )
        )
    return products


def write_catalog(products: list[Product], path: str | Path) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(product.model_dump_json() for product in products) + "\n", encoding="utf-8")
    return output


class SyntheticEncoder:
    """无模型编码器：把 ``syn0001234`` 解析回对应行，其余文本走确定性哈希。

    作用是把整条检索链路（编码 → ANN → 过滤 → 排序）在没有 GPU 的情况下跑起来，
    并且每条查询都有确定性的标准答案。它**不能**用来评估真实的图文语义检索质量，
    只能用来评估**检索管线与索引结构**——这个边界必须写清楚。
    """

    def __init__(self, corpus: SyntheticCorpus, fallback_dimension: int | None = None) -> None:
        self.corpus = corpus
        self.dimension = int(corpus.dimension if corpus.count else (fallback_dimension or 512))

    def _vector_for_token(self, index: int) -> np.ndarray:
        row = index % self.corpus.count
        return self.corpus.vectors[row]

    def _hash_vector(self, text: str) -> np.ndarray:
        # 用 sha256 而不是内置 hash()：后者受 PYTHONHASHSEED 影响，跨进程不可复现。
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**32)
        rng = np.random.default_rng(seed)
        return normalize(rng.normal(size=(1, self.dimension)).astype(np.float32))[0]

    def encode_texts(self, texts) -> np.ndarray:
        rows = []
        for text in texts:
            match = PRODUCT_TOKEN_PATTERN.search(text or "")
            rows.append(self._vector_for_token(int(match.group(1))) if match else self._hash_vector(text or ""))
        return normalize(np.asarray(rows, dtype=np.float32))

    def encode_images(self, paths) -> np.ndarray:
        rows = []
        for path in paths:
            match = PRODUCT_TOKEN_PATTERN.search(str(path))
            rows.append(self._vector_for_token(int(match.group(1))) if match else self._hash_vector(str(path)))
        return normalize(np.asarray(rows, dtype=np.float32))


def query_for(product_id: str) -> str:
    """构造一条"点名查询"，用于端到端链路验证。"""
    return f"找一款 {product_id} 同款商品"


__all__ = [
    "CATEGORIES",
    "COLORS",
    "MATERIALS",
    "PRODUCT_TOKEN_PATTERN",
    "SCENES",
    "STYLES",
    "SyntheticCorpus",
    "SyntheticEncoder",
    "generate_corpus",
    "query_for",
    "write_catalog",
]
