"""重排层：粗排之后用更贵的信号决定最终顺序。

为什么"多取候选"本身就能涨召回
--------------------------------
[REAL_CLIP_BENCHMARK.md](../docs/REAL_CLIP_BENCHMARK.md) 里的 IVF-PQ 是这个现象最干净的例子：
粗排 recall@10 只有 0.097（PQ 重建的分数有偏），但取 500 条候选、用原始 fp32 向量精确重排，
recall@10 回到 0.9375。**候选集是好的，坏的是排序信号**——所以本层的第一件事不是换模型，
而是把"排序"从粗排信号交回给一个更贵的信号。

两级重排
--------
- ``ExactVectorReranker``：用 fp32 原始向量重算余弦。零额外依赖、零模型加载，代价是必须留着
  ``vectors.npy``（也就是 ``FaissRetriever`` 里那份"看起来冗余"的副本）。它同时是
  cross-encoder 不可用时的**降级路径**——重排层绝不能因为缺一个可选依赖就整个失效。
- ``CrossEncoderReranker``：query 与 doc 拼在一起送进 cross-encoder 联合打分。
  这是唯一能修正"向量语义近邻压过真值"这类失败的手段：双塔编码器分别编码 query 和 doc，
  两者之间**没有交互**，所以它对词序、规格数字、局部否定天生不敏感；cross-encoder 有交互。

依赖策略
--------
cross-encoder 需要 torch + transformers，是**可选依赖**。本模块与项目其他可选依赖
（faiss、Redis、Chinese-CLIP）保持同一套约定：惰性导入、缺失时给可执行的安装提示，
并让上层能显式地"配了但用不上"报错而不是静默降级。推理耗时写进 ``last_stats``，
让"重排带来的延迟"可观测，而不是藏在代码里的常数。
"""

from __future__ import annotations

import time
from typing import Protocol, Sequence

import numpy as np

from .models import Product, SearchHit, SearchRequest

RERANK_MODEL_HINT = 'cross-encoder 依赖未安装，请执行 pip install -e ".[clip]"（torch + transformers）'


class Reranker(Protocol):
    """重排器契约：给定查询与候选商品，返回与候选一一对应的分数。

    分数方向统一为**越大越相关**——上层只做稳定排序，不关心分数量纲。
    长度必须与候选数一致；不一致时上层会拒绝重排（见 ``RerankRetriever``）。
    """

    name: str

    def score(self, query: str, candidates: Sequence[Product]) -> np.ndarray: ...

    @property
    def last_stats(self) -> dict[str, float | int | str | bool]: ...


def product_document(product: Product) -> str:
    """把商品拼成一段供重排器阅读的文本。

    标题放最前且独立成句：cross-encoder 的输入是一整段串，
    字段顺序会改变它读到的"主语"，而标题才是这次排序最该依据的信号。
    """
    attributes = " ".join(f"{key} {value}" for key, value in product.attributes.items())
    body = " ".join(part for part in (product.description, attributes) if part)
    return f"{product.title}。{product.category}。{body}".strip()


class ExactVectorReranker:
    """用原始稠密向量重算余弦。零依赖、零模型加载，是 cross-encoder 的降级基线。

    它需要**查询向量**而不是查询文本：检索链路上查询早已编码过一次，
    重排阶段再编码一遍是纯浪费（同 ``FaissRetriever`` 里"编码一次、分发多处"的原则）。
    """

    name = "exact-vector"

    def __init__(self, encoder, product_vectors: np.ndarray) -> None:
        vectors = np.asarray(product_vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("product_vectors 必须是二维数组 (N, D)")
        # 商品向量必须在构造时就归一化：这里算的是内积，
        # 若向量未归一化，内积就不再等于余弦，"精确重排"会静默给出错误的排序。
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-12)
        self.encoder = encoder
        self.index = {product.id: row for row, product in enumerate(getattr(encoder, "products", []) or [])}
        self.last_stats: dict[str, float | int | str | bool] = {}

    def bind_products(self, products: Sequence[Product]) -> None:
        """把候选商品 id 映射到向量行号。构造时不接收商品列表，便于与检索器解耦。"""
        self.index = {product.id: row for row, product in enumerate(products)}

    def score(self, query: str, candidates: Sequence[Product]) -> np.ndarray:
        if not candidates:
            self.last_stats = {"reranker": self.name, "candidates": 0, "model_inference_ms": 0.0}
            return np.empty(0, dtype=np.float32)
        query_vector = self.encoder.encode_texts([query])[0]
        rows = [self.index[product.id] for product in candidates]
        self.last_stats = {"reranker": self.name, "candidates": len(rows), "model_inference_ms": 0.0}
        return (self.vectors[rows] @ np.asarray(query_vector, dtype=np.float32)).astype(np.float32)

    def score_vectors(self, query_vector: np.ndarray, rows: Sequence[int]) -> np.ndarray:
        """检索链路的快捷路径：查询向量已算好时免去一次文本编码。"""
        indices = np.asarray(rows, dtype=np.int64)
        self.last_stats = {"reranker": self.name, "candidates": int(indices.size), "model_inference_ms": 0.0}
        if not indices.size:
            return np.empty(0, dtype=np.float32)
        return (self.vectors[indices] @ np.asarray(query_vector, dtype=np.float32)).astype(np.float32)


class CrossEncoderReranker:
    """cross-encoder 联合打分（bge-reranker 等）。需要 torch + transformers，为可选依赖。"""

    name = "cross-encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 128,
        dtype: str = "float32",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(RERANK_MODEL_HINT) from error
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtype_map:
            raise ValueError(f"不支持的 dtype：{dtype}（可选 {sorted(dtype_map)}）")
        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.dtype = dtype_map[dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device, dtype=self.dtype).eval()
        self.last_stats: dict[str, float | int | str | bool] = {}

    def score(self, query: str, candidates: Sequence[Product]) -> np.ndarray:
        if not candidates:
            self.last_stats = {"reranker": self.name, "candidates": 0, "model_inference_ms": 0.0}
            return np.empty(0, dtype=np.float32)
        documents = [product_document(product) for product in candidates]
        scores: list[np.ndarray] = []
        start = time.perf_counter()
        for begin in range(0, len(documents), self.batch_size):
            chunk = documents[begin : begin + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(chunk),
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
                for key, value in inputs.items()
            }
            with self.torch.inference_mode():
                logits = self.model(**inputs).logits.view(-1)
            scores.append(logits.float().cpu().numpy())
        elapsed = (time.perf_counter() - start) * 1000.0
        merged = np.concatenate(scores).astype(np.float32)
        self.last_stats = {
            "reranker": self.name,
            "model": self.model_name,
            "device": self.device,
            "candidates": len(documents),
            "batch_size": self.batch_size,
            "model_inference_ms": round(elapsed, 2),
            "per_candidate_ms": round(elapsed / max(len(documents), 1), 3),
        }
        return merged


class RerankRetriever:
    """把"粗排 → 重排"包成一个检索器，让 ``search`` 的调用方无需感知两阶段。

    粗排候选不足 ``candidates`` 时不报错、也不重排：
    重排是增益手段，不是正确性前提——候选不够时它做不了更多，也不该让整条链路失败。
    但"配了重排却一个候选都没进来"必须显式暴露在 ``last_stats`` 里，
    否则会让人以为重排没用，而实际是它根本没跑。
    """

    def __init__(
        self,
        base,
        reranker: Reranker | None = None,
        candidates: int = 100,
        keep: int | None = None,
    ) -> None:
        if candidates < 1:
            raise ValueError("candidates 必须大于等于 1")
        if keep is not None and keep < 1:
            raise ValueError("keep 必须大于等于 1")
        self.base = base
        self.reranker = reranker
        self.candidates = int(candidates)
        self.keep = int(keep) if keep else None
        self.products = base.products
        self.catalog_path = base.catalog_path
        self.last_stats: dict[str, float | int | str | bool] = {}

    def _coarse(self, request: SearchRequest) -> list[SearchHit]:
        """粗排阶段放宽到 ``candidates`` 条。

        不能用 ``model_copy`` 硬改 ``top_k``：``SearchRequest.top_k`` 上限是 20（对外接口契约），
        而重排候选需要上百条。因此走检索器提供的候选级入口，
        没有该入口的后端才退化为改写 ``top_k``。
        """
        wanted = max(self.candidates, int(request.top_k))
        candidates_of = getattr(self.base, "search_candidates", None)
        if callable(candidates_of):
            return candidates_of(request, wanted)
        if wanted > 20:
            return self.base.search(request.model_copy(update={"top_k": 20}, deep=False))
        return self.base.search(request.model_copy(update={"top_k": wanted}, deep=False))

    def search(self, request: SearchRequest) -> list[SearchHit]:
        wanted = int(request.top_k)
        coarse_hits = self._coarse(request)
        base_stats = dict(getattr(self.base, "last_stats", {}) or {})

        if self.reranker is None or not coarse_hits:
            self.last_stats = {**base_stats, "rerank": "off", "reranked": 0}
            return coarse_hits[:wanted]

        scores = self.reranker.score(request.query, [hit.product for hit in coarse_hits])
        if scores.shape[0] != len(coarse_hits):
            # 打分数与候选数不一致说明重排器内部截断了；宁可不做重排，也不要用错位的分数排序。
            self.last_stats = {**base_stats, "rerank": "skipped_size_mismatch", "reranked": 0}
            return coarse_hits[:wanted]

        reranked = [
            SearchHit(
                product=hit.product,
                score=round(float(score), 4),
                text_score=hit.text_score,
                image_score=hit.image_score,
                reasons=hit.reasons + [f"{self.reranker.name} 重排"],
            )
            for hit, score in zip(coarse_hits, scores)
        ]
        # Python 的 sort 是稳定的：不把 id 写进 key，交叉阶段就天然按粗排名次破同分。
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        self.last_stats = {
            **base_stats,
            "rerank": self.reranker.name,
            "reranked": len(reranked),
            **self.reranker.last_stats,
        }
        return reranked[: self.keep or wanted]

    def describe(self) -> dict[str, object]:
        described = getattr(self.base, "describe", None)
        return {
            "candidates": self.candidates,
            "keep": self.keep,
            "reranker": getattr(self.reranker, "name", "none"),
            "base": described() if callable(described) else {},
            "last_stats": dict(self.last_stats),
        }
