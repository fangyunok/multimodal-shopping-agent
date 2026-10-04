"""ANN 检索后端：向量只存索引，检索走近似最近邻。

为什么不能简单地把 ``product_vectors @ q`` 换成 ``index.search(q, top_k)``
--------------------------------------------------------------------------
这是规模化改造里最容易踩的坑。检索请求除了"语义相近"，还带两类**业务过滤**：
``max_price`` 与 ``category``。全量扫描时代码是"先算分、再逐条过滤"，过滤掉谁都不影响别人。

但 ANN 只认向量，不认价格。如果直接 `index.search(q, top_k)`，那 top_k 个近似最近邻里
可能有一大半因为超预算被丢掉，最终返回不足 top_k 条——**表现是"结果变少了"，
而根因是过滤发生在检索之后**。项目早先对这点没有压力，因为 6 条商品怎么过滤都够。

本模块的做法是**自适应超采样（adaptive over-fetch）**：

1. 先按 ``top_k × oversample`` 且不少于 ``min_candidates`` 的规模向 ANN 要候选；
2. 在候选上做向量化过滤，若剩下的不足 ``top_k`` 条，把请求规模翻 4 倍重来；
3. 直到凑够 ``top_k``、或已覆盖全库、或触到 ``max_rounds`` 上限。

超采样倍数被记录在 ``last_stats`` 里，因此"过滤越狠、超采样越大"这件事是可观测的，
而不是一个藏在代码里的魔法数字。

超采样的天花板，以及为什么需要精确兜底
--------------------------------------
一个自然的想法是"那就把 k 一路撑到全库，总能凑齐"。**这个想法在 HNSW 上不成立。**

实测：300 条向量、查询 300 个近邻，faiss HNSW 只返回 250 条（其余补 -1），
另一簇的 50 条只找到了 5 条；把 efSearch 从 32 提到 1024 也没有改善。
原因是 HNSW 返回的是"束搜索实际访问到的节点"，而不是"数学上最近的 k 个"，
k 变大并不会让图遍历自动覆盖全库。

于是"过滤掉 99% 候选"这类查询在纯 ANN 下有硬边界。本模块的处理是**精确兜底**：
当 ANN 已经把请求规模撑到全库、却仍凑不满 ``top_k`` 时，如果调用方提供了精确向量矩阵
（``exact_vectors``），就退化成一次全量精确扫描。

这条路径极少触发，但触发时它保证正确性 —— 这也正是索引默认保留 ``vectors.npy``
的实际价值：那份 2 GB 的副本不是冗余，它是**正确性的最后一道防线**。
没有它时，``last_stats["exhausted"]`` 会置为 True，把"结果不足"这个事实显式暴露出来，
而不是悄悄少返回几条。

最终方案是三类手段的组合，而不是单靠一种：
预过滤/分区（把过滤维度做成索引的一部分）→ 自适应超采样 → 精确兜底。

接口与 ``SemanticRetriever`` 完全一致（``products`` / ``catalog_path`` / ``search``），
所以 ``ScoreFusionRetriever``、``ShoppingTools``、MCP 层、FastAPI 层都不需要任何改动。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .ann_index import AnnIndex, AnnIndexConfig
from .encoders import MultimodalEncoder, normalize
from .indexing import encode_products, load_ann_index, load_catalog, load_vectors_if_present
from .models import Product, SearchHit, SearchRequest


class FaissRetriever:
    """近似最近邻检索后端；对外行为与精确后端保持一致，差异只体现在召回率与延迟上。"""

    def __init__(
        self,
        products: list[Product],
        catalog_path: Path,
        encoder: MultimodalEncoder,
        ann: AnnIndex,
        image_weight: float = 0.5,
        min_candidates: int = 128,
        oversample: float = 4.0,
        max_rounds: int = 6,
        exact_vectors: np.ndarray | None = None,
    ) -> None:
        if oversample < 1:
            raise ValueError("oversample 必须大于等于 1")
        if min_candidates < 1:
            raise ValueError("min_candidates 必须大于等于 1")
        if max_rounds < 1:
            raise ValueError("max_rounds 必须大于等于 1")
        if len(products) != ann.count:
            raise ValueError(f"商品数 {len(products)} 与索引条目数 {ann.count} 不一致")
        self.products = products
        self.catalog_path = catalog_path
        self.encoder = encoder
        self.ann = ann
        self.image_weight = image_weight
        self.min_candidates = min_candidates
        self.oversample = oversample
        self.max_rounds = max_rounds
        if exact_vectors is not None and exact_vectors.shape[0] != ann.count:
            raise ValueError(
                f"精确向量条数 {exact_vectors.shape[0]} 与索引条目数 {ann.count} 不一致"
            )
        self.exact_vectors = normalize(exact_vectors) if exact_vectors is not None else None
        # 过滤用的元数据一次性抽成 numpy 数组：过滤发生在候选集上，而不是全库。
        self._prices = np.asarray([product.price for product in products], dtype=np.float64)
        self._ratings = np.asarray([product.rating for product in products], dtype=np.float64)
        self._categories = [product.category.lower() for product in products]
        self.last_stats: dict[str, float | int | str | bool] = {}

    # ---------- 构造 ----------

    @classmethod
    def from_index(
        cls,
        catalog_path: str | Path,
        index_dir: str | Path,
        encoder: MultimodalEncoder,
        expected_model: str | None = None,
        config: AnnIndexConfig | None = None,
        **kwargs,
    ) -> "FaissRetriever":
        catalog = Path(catalog_path).resolve()
        products, manifest, ann = load_ann_index(catalog, index_dir, expected_model, config)
        # 有 vectors.npy 就带上：它是极端过滤下的精确兜底，不是冗余副本。
        vectors = load_vectors_if_present(index_dir)
        return cls(
            products,
            catalog,
            encoder,
            ann,
            image_weight=manifest.image_weight,
            exact_vectors=vectors,
            **kwargs,
        )

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        encoder: MultimodalEncoder,
        config: AnnIndexConfig | None = None,
        image_weight: float = 0.5,
        **kwargs,
    ) -> "FaissRetriever":
        """不落盘直接建索引，便于测试与单机压测。"""
        catalog = Path(path).resolve()
        products = load_catalog(catalog)
        if not products:
            raise ValueError("商品目录不能为空")
        vectors = encode_products(products, catalog, encoder, image_weight)
        resolved = config or AnnIndexConfig(kind="numpy", dimension=int(vectors.shape[1]))
        ann = AnnIndex.build(vectors, resolved)
        return cls(products, catalog, encoder, ann, image_weight=image_weight, exact_vectors=vectors, **kwargs)

    @classmethod
    def from_vectors(
        cls,
        products: list[Product],
        catalog_path: str | Path,
        encoder: MultimodalEncoder,
        vectors: np.ndarray,
        config: AnnIndexConfig,
        image_weight: float = 0.5,
        **kwargs,
    ) -> "FaissRetriever":
        """复用同一批向量构建不同后端的检索器。

        压测要做的正是"同一批向量、换五种索引"，没有这个入口就得把商品目录和编码
        重复做 N 遍——在十万级规模下这部分开销会超过索引本身。
        """
        # 调用方若显式传了 exact_vectors（例如要模拟"没有精确兜底"），以调用方为准。
        kwargs.setdefault("exact_vectors", vectors)
        return cls(
            products,
            Path(catalog_path).resolve(),
            encoder,
            AnnIndex.build(vectors, config),
            image_weight=image_weight,
            **kwargs,
        )

    # ---------- 检索 ----------

    def search(self, request: SearchRequest) -> list[SearchHit]:
        parts: list[np.ndarray] = []
        if request.query:
            parts.append(self.encoder.encode_texts([request.query])[0])
        if request.image_path:
            parts.append(self.encoder.encode_images([Path(request.image_path)])[0])
        if not parts:
            self.last_stats = {"mode": "rating_fallback", "rounds": 0, "examined": 0}
            return self._rank_by_rating(request)
        query_vector = normalize(np.asarray([sum(parts)]))[0]
        if query_vector.shape[0] != self.ann.config.dimension:
            raise ValueError(
                f"查询向量维度 {query_vector.shape[0]} 与索引维度 {self.ann.config.dimension} 不一致；"
                "请确认索引与查询用的是同一个编码器"
            )
        return self._ann_search(
            query_vector, request, has_text=bool(request.query), has_image=bool(request.image_path)
        )

    def _ann_search(
        self, query_vector: np.ndarray, request: SearchRequest, has_text: bool, has_image: bool
    ) -> list[SearchHit]:
        total = self.ann.count
        wanted = int(request.top_k)
        k = min(total, max(wanted, int(wanted * self.oversample), self.min_candidates))
        collected_indices = np.empty(0, dtype=np.int64)
        collected_scores = np.empty(0, dtype=np.float32)
        rounds = 0
        examined = 0
        requested = 0
        under_delivered = False

        while True:
            rounds += 1
            requested = k
            scores, indices = self.ann.search_one(query_vector, k)
            # faiss 会用 -1 把结果**补齐**到 k，所以数组长度恒等于 k，
            # 必须数有效条数才知道索引到底交付了多少。
            delivered = int((indices >= 0).sum())
            examined += delivered
            # 图索引（HNSW）在 k 变大时会"交付不足"：它返回的是束搜索**访问到的节点**，
            # 而不是距离最近的 k 个，所以可达规模由 efSearch 决定（实测 efSearch=64 时
            # 大约只能交付 ~700 条，k=512 以内才足额）。这时继续放大 k 是徒劳的：
            # 候选集已经被"访问到哪些"而非"离得多近"污染了，再翻倍只会重复同一批节点。
            if delivered < min(k, total):
                under_delivered = True
            mask = self._filter_mask(indices, request)
            keep_indices = indices[mask]
            keep_scores = scores[mask]
            if rounds == 1:
                collected_indices, collected_scores = keep_indices, keep_scores
            else:
                collected_indices = np.concatenate([collected_indices, keep_indices])
                collected_scores = np.concatenate([collected_scores, keep_scores])
            if keep_indices.shape[0] >= wanted or k >= total or rounds >= self.max_rounds or under_delivered:
                break
            # 过滤太狠，候选被吃光了：把检索规模翻四倍重来。
            k = min(total, k * 4)

        if rounds > 1 and collected_indices.size:
            # 多轮结果取并集：按分数降序后去重，保证同一商品只保留最高分那一次。
            order = np.argsort(-collected_scores, kind="stable")
            collected_indices = collected_indices[order]
            collected_scores = collected_scores[order]
            _, first = np.unique(collected_indices, return_index=True)
            collected_indices = collected_indices[first]
            collected_scores = collected_scores[first]

        fallback: str | None = None
        needs_exact = under_delivered or (collected_indices.size < wanted and requested >= total)
        if needs_exact and self.exact_vectors is not None:
            # 两种触发条件：① 索引交付不足，候选集不完整；② 已把规模撑到全库仍凑不满。
            # 两者都意味着"继续问 ANN 已经问不出更多了"，只有全量精确扫描能给出正确答案。
            # 代价是这一次查询退化成 O(N)，换的是"不会因为过滤而漏掉本该返回的商品"。
            collected_indices, collected_scores = self._exact_scan(query_vector, request)
            fallback = "exact"
        exhausted = collected_indices.size < wanted

        self.last_stats = {
            "mode": "ann",
            "kind": self.ann.config.kind,
            "rounds": rounds,
            "requested_k": requested,
            "examined": examined,
            "matched": int(collected_indices.size),
            "oversample": self.oversample,
            "under_delivered": under_delivered,
            "fallback": fallback or "none",
            "exhausted": exhausted,
        }

        # 按商品在目录中的原始顺序整理，再稳定排序——这样同分同评分的并列顺序与全量扫描后端一致。
        by_catalog_order = np.argsort(collected_indices, kind="stable")
        collected_indices = collected_indices[by_catalog_order]
        collected_scores = collected_scores[by_catalog_order]

        hits: list[SearchHit] = []
        for product_id, score in zip(collected_indices.tolist(), collected_scores.tolist()):
            product = self.products[product_id]
            reasons = ["CLIP 图文语义相似"]
            if request.max_price is not None:
                reasons.append(f"价格不超过 ¥{request.max_price:g}")
            hits.append(
                SearchHit(
                    product=product,
                    score=round(float(score), 4),
                    text_score=round(float(score), 4) if has_text else 0,
                    image_score=round(float(score), 4) if has_image else 0,
                    reasons=reasons,
                )
            )
        hits.sort(key=lambda hit: (hit.score, hit.product.rating), reverse=True)
        return hits[:wanted]

    def _rank_by_rating(self, request: SearchRequest) -> list[SearchHit]:
        """无文字无图片时按评分排序：这不是向量检索，没必要绕道 ANN。"""
        indices = np.arange(self.ann.count)
        kept = indices[self._filter_mask(indices, request)]
        kept = kept[np.argsort(-self._ratings[kept], kind="stable")]
        hits: list[SearchHit] = []
        for product_id in kept[: request.top_k].tolist():
            product = self.products[product_id]
            reasons = ["按商品评分排序"]
            if request.max_price is not None:
                reasons.append(f"价格不超过 ¥{request.max_price:g}")
            hits.append(
                SearchHit(
                    product=product,
                    score=round(product.rating / 5, 4),
                    text_score=0,
                    image_score=0,
                    reasons=reasons,
                )
            )
        return hits

    def _exact_scan(self, query_vector: np.ndarray, request: SearchRequest) -> tuple[np.ndarray, np.ndarray]:
        """对全库做一次精确余弦扫描并施加过滤。只在 ANN 已穷尽时触发。"""
        assert self.exact_vectors is not None
        scores = self.exact_vectors @ query_vector
        mask = self._filter_mask(np.arange(self.ann.count), request)
        kept = np.flatnonzero(mask)
        order = np.argsort(-scores[kept], kind="stable")
        return kept[order], scores[kept][order]

    def _filter_mask(self, indices: np.ndarray, request: SearchRequest) -> np.ndarray:
        """在候选索引上做向量化过滤；越界与 -1（faiss 的填充值）一律判为不通过。"""
        if indices.size == 0:
            return np.zeros(0, dtype=bool)
        mask = (indices >= 0) & (indices < self.ann.count)
        safe = np.where(mask, indices, 0)
        if request.max_price is not None:
            mask &= self._prices[safe] <= request.max_price
        if request.category:
            needle = request.category.lower()
            mask &= np.asarray([needle in self._categories[int(i)] for i in safe], dtype=bool)
        return mask

    # ---------- 观测 ----------

    def describe(self) -> dict[str, object]:
        return {
            **self.ann.summary(),
            "image_weight": self.image_weight,
            "oversample": self.oversample,
            "min_candidates": self.min_candidates,
            "exact_fallback": self.exact_vectors is not None,
            "last_stats": dict(self.last_stats),
        }
