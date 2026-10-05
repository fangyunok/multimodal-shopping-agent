"""融合层：把多路检索结果合成一个有序列表。

两种融合方式，为什么默认用 RRF
---------------------------
``ScoreFusionRetriever``（原有）做**分数线性加权**：``w·lexical + (1-w)·semantic``。
它直观，但要求两路分数**量纲可比**，而 BM25 分数无上界（取决于语料与词项分布）、
CLIP 余弦落在 [-1, 1]。两者直接相加，等于让 ``lexical_weight`` 承担全部量纲对齐的责任，
而这个值随语料、随查询长度漂移——换一批数据就要重调，调不好时排名的变化其实来自分数尺度
而不是相关性。

``ReciprocalRankFusion``（本模块新增）只吃**名次**，不看分数：

    RRF(d) = Σ_retriever weight / (k + rank_retriever(d))

``k``（默认 60）是唯一的平滑常数，作用是压制头部：rank 1 与 rank 2 的贡献差是
``1/61 - 1/62 ≈ 2.6e-4``，也就是说"两路都把某商品排进前几名"会自然压过"一路第一、一路五十"。
这正是混合检索想要的行为——**两路独立证据共同认可**的商品应该浮上来。

为什么融合必须先放宽候选集
------------------------
两路各自只返回 top-20 时，融合只能在 40 条里挑，而稠密路的真值常常落在 rank 30~200
（[QUERY_DISTRIBUTION.md](../docs/QUERY_DISTRIBUTION.md) 测出的 33pt 缺口正落在这个区间）。
因此融合器一律先向各路要 ``top_k × oversample`` 条再融合，
并且把每路实际交回多少条写进 ``last_stats``——某一路候选不足时会静默变权，
这种"权重被数据悄悄改了"的情况必须能被观测到。
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import SearchHit, SearchRequest

DEFAULT_RRF_K = 60


class ReciprocalRankFusionRetriever:
    """RRF 融合：只按名次合并多路结果，与分数量纲无关。

    构造时接收任意个「检索器 + 权重」，每个检索器只需满足 ``search(SearchRequest) -> list[SearchHit]``
    （``FaissRetriever`` / ``BM25Retriever`` / ``SemanticRetriever`` 都满足）。
    """

    def __init__(self, retrievers: Sequence[tuple[object, float]], rrf_k: int = DEFAULT_RRF_K, oversample: float = 5.0):
        if not retrievers:
            raise ValueError("RRF 至少需要一路检索器")
        if rrf_k < 1:
            raise ValueError("rrf_k 必须大于等于 1")
        if oversample < 1:
            raise ValueError("oversample 必须大于等于 1")
        for _, weight in retrievers:
            if weight <= 0:
                raise ValueError("每路检索器的权重必须大于 0")
        self.retrievers = list(retrievers)
        self.rrf_k = int(rrf_k)
        self.oversample = float(oversample)
        head = self.retrievers[0][0]
        self.products = head.products  # type: ignore[attr-defined]
        self.catalog_path = head.catalog_path  # type: ignore[attr-defined]
        self.last_stats: dict[str, object] = {}

    def _route_name(self, retriever) -> str:
        """每路检索器的稳定标识，进入 ``last_stats`` 与命中理由，跨进程可比较。"""
        name = getattr(retriever, "route_name", None)
        if isinstance(name, str):
            return name
        return type(retriever).__name__

    def search(self, request: SearchRequest) -> list[SearchHit]:
        return self.search_candidates(request, int(request.top_k))[: int(request.top_k)]

    def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
        """返回最多 ``count`` 条融合结果。

        融合器自己也是别人的粗排来源（叠 cross-encoder 时就是），
        所以它必须和底层检索器一样提供候选级入口；否则「融合 → 重排」这条链路
        会因为缺这个方法而退化成 top_k 截断，重排只能在一小撮结果里换顺序。
        """
        if count < 1:
            raise ValueError("count 必须大于等于 1")
        wanted = int(count)
        # 候选窗口要够宽：稠密路的真值常落在 rank 30~200，而 SearchRequest.top_k 的
        # 对外上限是 20。所以优先走检索器的候选级入口（不受该上限约束），
        # 只有不具备该入口的后端才退化为改写 top_k。
        pool = max(wanted, int(wanted * self.oversample))

        fused: dict[str, float] = {}
        hits_by_id: dict[str, SearchHit] = {}
        contributions: dict[str, dict[str, int]] = {}
        per_route: dict[str, int] = {}

        for retriever, weight in self.retrievers:
            route = self._route_name(retriever)
            candidates_of = getattr(retriever, "search_candidates", None)
            if callable(candidates_of):
                hits = candidates_of(request, pool)
            else:
                hits = retriever.search(request.model_copy(update={"top_k": min(pool, 20)}, deep=False))
            per_route[route] = len(hits)
            for rank, hit in enumerate(hits, start=1):
                product_id = hit.product.id
                fused[product_id] = fused.get(product_id, 0.0) + weight / (self.rrf_k + rank)
                # 保留原始命中对象：融合只决定顺序，不该改写各路给出的分数与理由。
                hits_by_id.setdefault(product_id, hit)
                contributions.setdefault(product_id, {})[route] = rank

        # 排序键里带上首次出现顺序作为最后的破同分依据，保证同分结果的确定性。
        order = {product_id: position for position, product_id in enumerate(hits_by_id)}
        ranked = sorted(fused.items(), key=lambda item: (-item[1], order[item[0]]))
        results: list[SearchHit] = []
        for product_id, score in ranked[:wanted]:
            hit = hits_by_id[product_id]
            routes = contributions[product_id]
            reasons = list(hit.reasons) + [
                f"RRF 融合 {route} 第 {rank} 名" for route, rank in routes.items() if len(self.retrievers) > 1
            ]
            results.append(
                SearchHit(
                    product=hit.product,
                    score=round(float(score), 6),
                    text_score=hit.text_score,
                    image_score=hit.image_score,
                    reasons=reasons,
                )
            )
        self.last_stats = {
            "mode": "rrf",
            "rrf_k": self.rrf_k,
            "oversample": self.oversample,
            "candidate_window": pool,
            "routes": len(self.retrievers),
            "fused_candidates": len(fused),
            "per_route_hits": per_route,
        }
        return results

    def describe(self) -> dict[str, object]:
        described = []
        for retriever, weight in self.retrievers:
            detail = getattr(retriever, "describe", None)
            described.append(
                {
                    "name": self._route_name(retriever),
                    "weight": weight,
                    "detail": detail() if callable(detail) else {},
                }
            )
        return {
            "mode": "rrf",
            "rrf_k": self.rrf_k,
            "oversample": self.oversample,
            "routes": described,
            "last_stats": dict(self.last_stats),
        }
