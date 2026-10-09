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
**过滤下推**（把过滤维度做成索引的一部分，见 ``partitioned.py``）→ 自适应超采样 → 精确兜底。

粗召回 + 精确重排（``rerank_candidates``）
--------------------------------------------
乘积量化（IVF-PQ）把索引压到 1/21，代价是重建出来的分数有偏，所以直接用 recall@10 只有 0.097。
但**候选集本身是好的**：取 500 条候选、用原始向量精确重排，recall@10 回到 0.9375。
这是个业界通用的组合——先做一次便宜且召回尚可的粗筛，再对粗筛结果做一次昂贵但精确的重排。

本类把它做成 ``rerank_candidates`` 一档开关：设了之后，检索目标从"凑够 top_k"抬高到
"凑够 rerank_candidates"，凑齐后用 ``exact_vectors`` 重新算分再截断。
代价是重排必须拿到原始向量，所以开启它等于承认 ``vectors.npy`` 不是冗余而是**必要组件**：
压缩掉的 180 MB 内存，是用这 2 GB 副本换回来的。这两笔账必须一起看。

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


def encode_request(
    encoder: MultimodalEncoder, request: SearchRequest, dimension: int | None = None
) -> np.ndarray | None:
    """把请求里的文字与图片编码成单个 L2 归一化查询向量。

    两者都为空时返回 ``None``——调用方据此走"按评分排序"的兜底路径，
    而不是硬造一个零向量（零向量与任何向量的内积都是 0，会得到一批看似正常的垃圾结果）。

    之所以是模块级函数而不是私有方法，是为了让 ``PartitionedRetriever`` 能**只编码一次**
    再分发给各分段：分段一多，重复编码会比检索本身更贵。
    """
    parts: list[np.ndarray] = []
    if request.query:
        parts.append(encoder.encode_texts([request.query])[0])
    if request.image_path:
        parts.append(encoder.encode_images([Path(request.image_path)])[0])
    if not parts:
        return None
    vector = normalize(np.asarray([sum(parts)]))[0]
    if dimension is not None and vector.shape[0] != dimension:
        raise ValueError(
            f"查询向量维度 {vector.shape[0]} 与索引维度 {dimension} 不一致；"
            "请确认索引与查询用的是同一个编码器"
        )
    return vector


class FaissRetriever:
    """近似最近邻检索后端；对外行为与精确后端保持一致，差异只体现在召回率与延迟上。"""

    @property
    def route_name(self) -> str:
        """融合层里的稳定标识：把后端类型带进统计，避免五路检索在报告里都叫同一个名字。"""
        return f"ann-{self.ann.config.kind}"

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
        rerank_candidates: int = 0,
    ) -> None:
        if oversample < 1:
            raise ValueError("oversample 必须大于等于 1")
        if min_candidates < 1:
            raise ValueError("min_candidates 必须大于等于 1")
        if max_rounds < 1:
            raise ValueError("max_rounds 必须大于等于 1")
        if rerank_candidates < 0:
            raise ValueError("rerank_candidates 不能为负")
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
        if rerank_candidates and self.exact_vectors is None:
            # 不能"配了重排却静默不生效"：那会让人以为重排没用，而实际是它根本没跑。
            raise ValueError(
                "rerank_candidates 需要原始向量才能精确重排，但索引里没有 vectors.npy。"
                "请用 keep_vectors=True 重建索引，或把 rerank_candidates 设为 0。"
            )
        self.rerank_candidates = int(rerank_candidates)

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
        query_vector = self.encode_request(request)
        if query_vector is None:
            self.last_stats = {"mode": "rating_fallback", "rounds": 0, "examined": 0}
            return self._rank_by_rating(request)
        return self.search_vector(query_vector, request)

    def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
        """返回最多 ``count`` 条候选，不做 top_k 截断。

        融合与重排需要比 ``top_k`` 宽得多的候选集：``SearchRequest.top_k`` 的上限是 20
        （对外接口契约），而两阶段检索的候选规模是上百条级。所以这条入口与 ``search``
        共用同一段检索逻辑，只在最后一步放宽截断，避免两条路径给出不一致的结果。
        """
        if count < 1:
            raise ValueError("count 必须大于等于 1")
        wanted = max(int(request.top_k), int(count))
        query_vector = self.encode_request(request)
        if query_vector is None:
            self.last_stats = {"mode": "rating_fallback", "rounds": 0, "examined": 0}
            return self._rank_by_rating(request.model_copy(update={"top_k": wanted}, deep=False))
        return self._search_wide(query_vector, request, wanted)

    def encode_request(self, request: SearchRequest) -> np.ndarray | None:
        """把请求编码成查询向量；文字与图片都没有时返回 None。"""
        return encode_request(self.encoder, request, self.ann.config.dimension)

    def search_vector(self, query_vector: np.ndarray, request: SearchRequest) -> list[SearchHit]:
        """给定已编码的查询向量直接检索。

        ``PartitionedRetriever`` 靠它把"编码一次、分发到 N 个分段"变成可能：
        否则分段数一多，模型前向的次数就等于分段数，比检索本身还贵。
        """
        return self._ann_search(
            query_vector, request, has_text=bool(request.query), has_image=bool(request.image_path)
        )

    def _search_wide(self, query_vector: np.ndarray, request: SearchRequest, limit: int) -> list[SearchHit]:
        return self._ann_search(
            query_vector, request, has_text=bool(request.query), has_image=bool(request.image_path), limit=limit
        )

    def _ann_search(
        self,
        query_vector: np.ndarray,
        request: SearchRequest,
        has_text: bool,
        has_image: bool,
        limit: int | None = None,
    ) -> list[SearchHit]:
        total = self.ann.count
        wanted = int(request.top_k) if limit is None else max(int(request.top_k), int(limit))
        # 开启重排时，检索目标从"凑够 top_k"抬高到"凑够 rerank_candidates"——
        # 粗筛的价值全在候选集够大，候选集不够大时重排只能把一堆无关项排出个先后。
        target = max(wanted, self.rerank_candidates) if self.rerank_candidates else wanted
        k = min(total, max(target, int(wanted * self.oversample), self.min_candidates))
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
            if keep_indices.shape[0] >= target or k >= total or rounds >= self.max_rounds or under_delivered:
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

        # 精确重排：粗筛只负责"别漏"，排序交回原始向量。
        # fallback 时结果本来就是精确的，再重排一次是白做。
        reranked = 0
        if fallback is None and self.rerank_candidates and collected_indices.size:
            assert self.exact_vectors is not None
            exact_scores = (self.exact_vectors[collected_indices] @ query_vector).astype(np.float32)
            order = np.argsort(-exact_scores, kind="stable")
            collected_indices = collected_indices[order]
            collected_scores = exact_scores[order]
            reranked = int(collected_indices.size)

        exhausted = collected_indices.size < wanted

        self.last_stats = {
            "mode": "ann",
            "kind": self.ann.config.kind,
            "rounds": rounds,
            "requested_k": requested,
            "target": target,
            "examined": examined,
            "matched": int(collected_indices.size),
            "oversample": self.oversample,
            "under_delivered": under_delivered,
            "fallback": fallback or "none",
            "rerank": "exact" if reranked else "off",
            "reranked": reranked,
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
        if request.excluded_product_ids:
            excluded = set(request.excluded_product_ids)
            mask &= np.asarray([self.products[int(i)].id not in excluded for i in safe], dtype=bool)
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
            "rerank_candidates": self.rerank_candidates,
            "exact_fallback": self.exact_vectors is not None,
            "last_stats": dict(self.last_stats),
        }

