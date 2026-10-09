"""稀疏检索层：BM25 倒排索引。

为什么在已有稠密向量检索之后还要 BM25
------------------------------------
[QUERY_DISTRIBUTION.md](../docs/QUERY_DISTRIBUTION.md) 用 500 条真实中文查询量出一个 33 个百分点的
缺口：同分布切片查询 recall@10 = 0.9982，用户打字描述只有 0.668。把 166 条未命中拆开看，
**没有一条是"查询在问另一件商品"**，绝大多数是同品类的近邻商品压过了真值。

这说明缺的不是"更强的向量"，而是**一路完全独立的证据**：

- 稠密检索擅长"意思相近"（语义近邻、同品类不同款式），但对**精确词项**是糊的——
  「5W-30」和「5W-30 机油」在向量空间里几乎一样，销量高的相邻规格很容易挤到前面；
- BM25 恰好相反：它只认字面。「132x168」「5升」「heart-shaped」这类**规格、型号、尺寸**
  越精确，词项匹配的区分度就越高，而这正是那 166 条失败样本里最容易丢的信息。

两路证据的**错误模式不同**，这才是混合检索能补上缺口的前提——如果两路都靠语义，
加权平均只会把同一批近邻重排一遍。融合方式因此也不能沿用现有的线性加权：
BM25 分数无上界、CLIP 余弦在 [-1, 1]，两者量纲不可比，直接加权等于让 `lexical_weight`
变成一个必须靠语料调的经验值。``fusion_retrieval.py`` 里的 RRF 只看**名次**不看分数，
天然免疫量纲问题。

分词：中文用「单字 + 相邻二元组」
--------------------------------
中文没有空格，jieba 之类词典分词会引入一个额外依赖和多一层失败模式。这里用无词典方案：
单字保证召回（任何字都能命中），二元组保证精度（「机油」比「机」+「油」更能定位）。
拉丁串与数字整体保留，它们承载型号与规格（``5W-30``、``132x168``、``J7``），
拆成单字符会让这些最有区分度的词项消失。

实现：稀疏打分而不是稠密矩阵
----------------------------
倒排表按词项存 ``(doc_ids, term_freqs)`` 两列。查询时只对**命中的 posting** 累加分数，
复杂度是 O(命中 posting 数) 而不是 O(语料条数)。这一点决定了它能不能作为向量检索的并列路径：
稠密路径无论语料多大都是全量矩阵乘（O(N·D)），稀疏路径的成本只随查询词的实际分布度增长，
在百万级语料上仍可接受——这正是"用 BM25 兜住长尾召回"的前提。
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .models import Product, SearchHit, SearchRequest

BM25_INDEX_FILENAME = "bm25.npz"
BM25_META_FILENAME = "bm25_meta.json"

# 拉丁字母数字串（含小数点、连字符）：型号与规格的主要载体。
LATIN_PATTERN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def analyze(text: str) -> list[str]:
    """把一段文本切成 BM25 词项：拉丁串整体保留，中文出单字与相邻二元组。"""
    lowered = text.lower()
    terms: list[str] = [match.group(0) for match in LATIN_PATTERN.finditer(lowered)]
    cjk = CJK_PATTERN.findall(lowered)
    terms.extend(cjk)
    terms.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return terms


def bm25_idf(doc_count: int, doc_freq: int) -> float:
    """BM25 的 IDF 用 ``log(1 + (N-df+0.5)/(df+0.5))``。

    相比经典 ``log((N-df+0.5)/(df+0.5))``，外面的 ``log(1+x)`` 保证**恒为正**。
    这不是为了好看：一个在 90% 文档里都出现的词（如类目名）若拿到负 IDF，
    反而会把包含它的文档**推后**——一个出现得越普遍就越该被惩罚的项，
    在纯 BM25 里属于符号错误，而不是"负相关"。
    """
    return math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


class BM25Index:
    """倒排索引 + BM25 打分。契约与 ``AnnIndex`` 对齐：build / search / save / load。"""

    def __init__(
        self,
        postings: dict[str, tuple[np.ndarray, np.ndarray]],
        idf: dict[str, float],
        doc_lengths: np.ndarray,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self.postings = postings
        self.idf = idf
        self.doc_lengths = doc_lengths.astype(np.float32)
        self.k1 = float(k1)
        self.b = float(b)
        self.avg_doc_length = float(self.doc_lengths.mean()) if self.doc_lengths.size else 0.0

    @property
    def count(self) -> int:
        return int(self.doc_lengths.shape[0])

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    @property
    def index_bytes(self) -> int:
        """倒排表实际占用：posting 的 doc_id(int32) + term_freq(float32)。"""
        total = 0
        for doc_ids, freqs in self.postings.values():
            total += doc_ids.nbytes + freqs.nbytes
        return total + int(self.doc_lengths.nbytes)

    @property
    def size_mb(self) -> float:
        return round(self.index_bytes / (1024 * 1024), 3)

    @classmethod
    def build(cls, documents: list[str], k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> "BM25Index":
        if k1 < 0:
            raise ValueError("k1 不能为负")
        if not 0 <= b <= 1:
            raise ValueError("b 必须在 0 到 1 之间")
        if not documents:
            raise ValueError("不能对空文档集合建 BM25 索引")

        # 先整份统计一遍：posting 按词项聚合需要"这个词出现在哪些文档、出现几次"，
        # 因此 doc_freq 与 doc_lengths 必须在建 posting 之前就算齐。
        raw: dict[str, list[tuple[int, int]]] = {}
        doc_lengths = np.zeros(len(documents), dtype=np.float32)
        for row, document in enumerate(documents):
            counts = Counter(analyze(document))
            doc_lengths[row] = sum(counts.values())
            for term, freq in counts.items():
                raw.setdefault(term, []).append((row, freq))

        postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        idf: dict[str, float] = {}
        count = len(documents)
        for term, pairs in raw.items():
            doc_ids = np.fromiter((row for row, _ in pairs), dtype=np.int32, count=len(pairs))
            freqs = np.fromiter((freq for _, freq in pairs), dtype=np.float32, count=len(pairs))
            postings[term] = (doc_ids, freqs)
            idf[term] = bm25_idf(count, len(pairs))
        return cls(postings, idf, doc_lengths, k1=k1, b=b)

    def search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(scores, indices)``；分数为 BM25 原始分（无上界，非负）。"""
        scores = np.zeros(self.count, dtype=np.float32)
        avg_length = self.avg_doc_length if self.avg_doc_length > 0 else 1.0
        # 查询里的重复词项按出现次数加权（等价于把查询当作 bag-of-terms 的加权求和）。
        for term, query_freq in Counter(analyze(query)).items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            doc_ids, freqs = posting
            lengths = self.doc_lengths[doc_ids]
            # k1 控制词频饱和：tf 很大时增益递减，避免长文档靠堆词赢。
            # b 控制长度归一化：b=0 完全不管长度，b=1 完全按平均长度拉平。
            numerator = freqs * (self.k1 + 1.0)
            denominator = freqs + self.k1 * (1.0 - self.b + self.b * lengths / avg_length)
            contribution = self.idf[term] * numerator / np.maximum(denominator, 1e-12)
            # 同一文档可能在多个词项上重复累加，np.add.at 保证不丢叠加。
            np.add.at(scores, doc_ids, (contribution * query_freq).astype(np.float32))

        k = int(min(top_k, self.count))
        if k <= 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
        partitioned = np.argpartition(-scores, k - 1)[:k]
        order = np.argsort(-scores[partitioned], kind="stable")
        indices = partitioned[order]
        return scores[indices], indices.astype(np.int64)

    def save(self, directory: str | Path) -> list[str]:
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        terms = sorted(self.postings)
        # 词项表本身要保序（load 端按同样的顺序读回），因此单独存一份词表，
        # 让 npz 里的扁平数组可以被向量化重建，而不必依赖 pickle。
        doc_id_chunks: list[np.ndarray] = []
        freq_chunks: list[np.ndarray] = []
        offsets = [0]
        for term in terms:
            doc_ids, freqs = self.postings[term]
            doc_id_chunks.append(doc_ids)
            freq_chunks.append(freqs)
            offsets.append(offsets[-1] + doc_ids.shape[0])
        np.savez_compressed(
            directory / BM25_INDEX_FILENAME,
            terms=np.asarray(terms, dtype=object),
            doc_ids=np.concatenate(doc_id_chunks) if doc_id_chunks else np.empty(0, dtype=np.int32),
            freqs=np.concatenate(freq_chunks) if freq_chunks else np.empty(0, dtype=np.float32),
            offsets=np.asarray(offsets, dtype=np.int64),
            idf=np.asarray([self.idf[term] for term in terms], dtype=np.float32),
            doc_lengths=self.doc_lengths,
        )
        (directory / BM25_META_FILENAME).write_text(
            json.dumps(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "count": self.count,
                    "vocabulary_size": len(terms),
                    "avg_doc_length": round(self.avg_doc_length, 4),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return [BM25_INDEX_FILENAME, BM25_META_FILENAME]

    @classmethod
    def load(cls, directory: str | Path) -> "BM25Index":
        directory = Path(directory).resolve()
        meta = json.loads((directory / BM25_META_FILENAME).read_text(encoding="utf-8"))
        payload = np.load(directory / BM25_INDEX_FILENAME, allow_pickle=True)
        terms = [str(term) for term in payload["terms"].tolist()]
        doc_ids = payload["doc_ids"]
        freqs = payload["freqs"]
        offsets = payload["offsets"]
        idf_values = payload["idf"]
        postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        idf: dict[str, float] = {}
        for position, term in enumerate(terms):
            start, end = int(offsets[position]), int(offsets[position + 1])
            postings[term] = (doc_ids[start:end], freqs[start:end])
            idf[term] = float(idf_values[position])
        return cls(
            postings,
            idf,
            payload["doc_lengths"],
            k1=float(meta["k1"]),
            b=float(meta["b"]),
        )

    def summary(self) -> dict[str, object]:
        return {
            "count": self.count,
            "vocabulary_size": self.vocabulary_size,
            "avg_doc_length": round(self.avg_doc_length, 2),
            "k1": self.k1,
            "b": self.b,
            "size_mb": self.size_mb,
        }


class BM25Retriever:
    """稀疏检索后端；    对外行为与 ``SemanticRetriever`` / ``FaissRetriever`` 一致。

    统一 ``products`` / ``catalog_path`` / ``search(request)`` 这三个成员，
    是为了让 ``ScoreFusionRetriever`` 与 RRF 融合器能把它和稠密后端**并列**接进来，
    而不需要知道底下是倒排表还是向量索引。
    """

    route_name = "bm25"

    def __init__(
        self,
        products: list[Product],
        catalog_path: Path,
        index: BM25Index,
        max_candidates: int = 0,
    ) -> None:
        if len(products) != index.count:
            raise ValueError(f"商品数 {len(products)} 与 BM25 索引条目数 {index.count} 不一致")
        if max_candidates < 0:
            raise ValueError("max_candidates 不能为负")
        self.products = products
        self.catalog_path = catalog_path
        self.index = index
        # 融合阶段需要"比 top_k 更宽的候选集"才能让两路证据互相补充，
        # 因此默认超采样到 4 倍；不足时退化为全库（等价于 BM25 精确排序）。
        self.max_candidates = int(max_candidates) if max_candidates else max(index.count, 1)
        self._prices = np.asarray([product.price for product in products], dtype=np.float64)
        self._categories = [product.category.lower() for product in products]
        self.last_stats: dict[str, float | int | str | bool] = {}

    @classmethod
    def from_products(
        cls,
        products: list[Product],
        catalog_path: str | Path,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        **kwargs,
    ) -> "BM25Retriever":
        documents = [product.searchable_text() for product in products]
        return cls(products, Path(catalog_path).resolve(), BM25Index.build(documents, k1=k1, b=b), **kwargs)

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs) -> "BM25Retriever":
        from .indexing import load_catalog

        catalog = Path(path).resolve()
        return cls.from_products(load_catalog(catalog), catalog, **kwargs)

    def search(self, request: SearchRequest) -> list[SearchHit]:
        wanted = int(request.top_k)
        return self.search_candidates(request, wanted)[:wanted]

    def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
        """返回最多 ``count`` 条候选，不做 top_k 截断——融合与重排需要更宽的候选集。"""
        if count < 1:
            raise ValueError("count 必须大于等于 1")
        pool = min(self.max_candidates, self.index.count, count)
        # 无查询词时倒排表无处可打分，走"按评分排序"兜底而不是返回空列表。
        if not request.query:
            self.last_stats = {"mode": "rating_fallback", "candidates": 0}
            return self._rank_by_rating(count, request)

        ceiling = min(self.max_candidates, self.index.count)
        while True:
            scores, indices = self.index.search(request.query, pool)
            mask = self._filter_mask(indices, request)
            kept_indices = indices[mask]
            kept_scores = scores[mask]
            if len(kept_indices) >= count or pool >= ceiling or len(indices) < pool:
                break
            pool = min(ceiling, max(pool + 1, pool * 2))

        self.last_stats = {
            "mode": "bm25",
            "candidates": int(kept_indices.shape[0]),
            "scanned": int(indices.shape[0]),
            "fraction": round(float(indices.shape[0]) / max(self.index.count, 1), 4),
        }

        hits: list[SearchHit] = []
        for row, score in zip(kept_indices.tolist(), kept_scores.tolist()):
            product = self.products[row]
            reasons = ["BM25 词项匹配"]
            if request.max_price is not None:
                reasons.append(f"价格不超过 ¥{request.max_price:g}")
            hits.append(
                SearchHit(
                    product=product,
                    score=round(float(score), 4),
                    text_score=round(float(score), 4),
                    image_score=0,
                    reasons=reasons,
                )
            )
        hits.sort(key=lambda hit: (hit.score, hit.product.rating), reverse=True)
        return hits[:count]

    def _rank_by_rating(self, wanted: int, request: SearchRequest) -> list[SearchHit]:
        indices = np.arange(self.index.count, dtype=np.int64)
        kept = indices[self._filter_mask(indices, request)]
        ratings = np.asarray([product.rating for product in self.products], dtype=np.float64)
        kept = kept[np.argsort(-ratings[kept], kind="stable")]
        hits: list[SearchHit] = []
        for row in kept[:wanted].tolist():
            product = self.products[row]
            reasons = ["按商品评分排序"]
            if request.max_price is not None:
                reasons.append(f"价格不超过 ¥{request.max_price:g}")
            hits.append(SearchHit(product=product, score=round(product.rating / 5, 4), text_score=0, image_score=0, reasons=reasons))
        return hits

    def _filter_mask(self, indices: np.ndarray, request: SearchRequest) -> np.ndarray:
        if indices.size == 0:
            return np.zeros(0, dtype=bool)
        mask = np.ones(indices.shape[0], dtype=bool)
        if request.excluded_product_ids:
            excluded = set(request.excluded_product_ids)
            mask &= np.asarray([self.products[int(i)].id not in excluded for i in indices], dtype=bool)
        if request.max_price is not None:
            mask &= self._prices[indices] <= request.max_price
        if request.category:
            needle = request.category.lower()
            mask &= np.asarray([needle in self._categories[int(i)] for i in indices], dtype=bool)
        return mask

    def describe(self) -> dict[str, object]:
        return {**self.index.summary(), "max_candidates": self.max_candidates, "last_stats": dict(self.last_stats)}

