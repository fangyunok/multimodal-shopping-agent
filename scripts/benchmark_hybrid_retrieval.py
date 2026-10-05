"""混合检索对照实验：把"33pt 缺口该由谁补上"从推断变成实测。

对照什么
--------
同一批语料、同一批查询、同一套真值，只改**检索与排序策略**：

1. ``dense_only``  ：仅 Chinese-CLIP 稠密向量（等价于 ``benchmark_ann_scaling.py`` 的 numpy 档）
2. ``bm25_only``   ：仅 BM25 倒排
3. ``rrf``         ：RRF 融合两路
4. ``rrf_rerank``  ：RRF 候选再交给重排器排前 k（exact-vector 或 cross-encoder）

为什么必须在**同一份语料**上比
----------------------------
A1（[QUERY_DISTRIBUTION.md](../docs/QUERY_DISTRIBUTION.md)）测出真实文本查询 hit@10 = 0.668，
同分布切片查询 0.9982。本脚本复用**完全相同**的语料、查询与真值装配，只换检索策略，
因此每一行的差值都可以直接归因到策略本身，而不是语料或样本的变化。

真值口径的诚实说明
----------------
真值 = 生成该查询的原商品（单元素集合）。这个口径偏严——A1 已证明未命中样本里
66.3% 的 top-1 与真值同 category，在用户视角下并不是错答案。
所以本脚本**同时**报告宽松口径 ``hit@k_category``：真值商品是否出现在 top-k，
**或** top-k 里是否有同 category 的商品。两个数字一起读，才能区分
"排序没排对"与"标注本身不成立"。

查询向量为什么可以预先注入
-------------------------
A1 已把 500 条查询编码成 ``queries.npy``（与线上查询端口径一致）。重跑一次编码既慢，
又会引入"两次编码结果不一致"的噪声。所以本脚本用 ``FixedQueryEncoder`` 把查询向量
按行号喂给检索器：检索链路的代码路径（``search(SearchRequest)``）完全不变，
只是"文本 → 向量"这一步换成了查表，模型前向从实验里彻底移除。

用法
----
    python scripts/benchmark_hybrid_retrieval.py \\
        --vectors corpus.npy --catalog corpus_catalog.jsonl \\
        --queries-file queries.npy --queries-catalog queries.jsonl \\
        --output-dir results/real_text_query

    # 叠一层 cross-encoder 重排（需要 torch + transformers）
    python scripts/benchmark_hybrid_retrieval.py ... \\
        --reranker cross-encoder --rerank-candidates 100 --save-scores results/ce_scores.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.bm25 import BM25Retriever  # noqa: E402
from shopping_agent.encoders import normalize  # noqa: E402
from shopping_agent.models import Product, SearchHit, SearchRequest  # noqa: E402
from shopping_agent.rerank import CrossEncoderReranker  # noqa: E402
from shopping_agent.rrf import ReciprocalRankFusionRetriever  # noqa: E402

ALL_KINDS = ("dense_only", "bm25_only", "rrf", "rrf_rerank")


class FixedQueryEncoder:
    """把预先算好的查询向量按当前行号返回，让检索链路的"编码"这一步变成查表。

    它满足 ``MultimodalEncoder`` 协议（``encode_texts``），因此 ``FaissRetriever`` /
    ``SemanticRetriever`` 这类后端可以原样接进来；而 BM25 根本不调用编码器。
    每次评一条查询前调一次 ``use()`` 切换到对应行——**漏掉这一步会让整轮指标静默错位**，
    所以所有检索器共用同一个实例，由主循环统一驱动。
    """

    def __init__(self, query_vectors: np.ndarray, products: list[Product]) -> None:
        self.vectors = normalize(np.asarray(query_vectors, dtype=np.float32))
        self.products = products
        self.position = 0
        self.calls = 0

    def use(self, position: int) -> None:
        self.position = int(position)

    def encode_texts(self, texts):
        self.calls += 1
        if len(texts) != 1:
            raise ValueError("FixedQueryEncoder 只支持单条查询：一次检索只对应一条查询向量")
        return self.vectors[self.position : self.position + 1]

    def encode_images(self, paths):
        raise RuntimeError("本实验不涉及图片查询")

    @property
    def current(self) -> np.ndarray:
        return self.vectors[self.position]


class DenseExactRetriever:
    """稠密基线：查询向量与语料矩阵做一次内积（= numpy 精确档，已归一化故内积即余弦）。"""

    route_name = "dense"

    def __init__(self, vectors: np.ndarray, products: list[Product], catalog_path: Path, encoder: FixedQueryEncoder):
        self.vectors = normalize(np.asarray(vectors, dtype=np.float32))
        self.products = products
        self.catalog_path = catalog_path
        self.encoder = encoder
        self.last_stats: dict[str, object] = {}

    def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
        start = time.perf_counter()
        scores = self.vectors @ self.encoder.current
        take = int(min(count, self.vectors.shape[0]))
        partitioned = np.argpartition(-scores, take - 1)[:take]
        order = np.argsort(-scores[partitioned], kind="stable")
        rows = partitioned[order]
        self.last_stats = {
            "mode": "dense_exact",
            "candidates": int(rows.size),
            "query_ms": round((time.perf_counter() - start) * 1000.0, 3),
        }
        hits = [
            SearchHit(
                product=self.products[row],
                score=round(float(scores[row]), 4),
                text_score=round(float(scores[row]), 4),
                image_score=0,
                reasons=["CLIP 图文语义相似"],
            )
            for row in rows.tolist()
        ]
        hits.sort(key=lambda hit: (hit.score, hit.product.rating), reverse=True)
        return hits

    def search(self, request: SearchRequest) -> list[SearchHit]:
        return self.search_candidates(request, int(request.top_k))

    def describe(self) -> dict[str, object]:
        return {"count": int(self.vectors.shape[0]), "dimension": int(self.vectors.shape[1])}


class ExactVectorTableReranker:
    """精确向量重排：候选商品 → 语料行号 → 与查询向量重算余弦。

    与 ``rerank.ExactVectorReranker`` 的唯一差别是查询向量按行号取而非现编码，
    因此这一档也能在没有模型的机器上复现。重排上限：它和稠密粗排用的是同一份向量，
    所以**不会改变顺序**——它的价值在于把"粗排信号换成排序信号"这件事的机制跑通，
    真正能提召回的是 cross-encoder。
    """

    def __init__(self, products: list[Product], row_of: dict[str, int], encoder: FixedQueryEncoder) -> None:
        self.name = "exact-vector"
        self.products = products
        self.row_of = row_of
        self.encoder = encoder
        self.last_stats: dict[str, object] = {}
        self.vectors: np.ndarray | None = None

    def bind_vectors(self, vectors: np.ndarray) -> None:
        self.vectors = normalize(np.asarray(vectors, dtype=np.float32))

    def score(self, query: str, candidates):
        if self.vectors is None:
            raise RuntimeError("ExactVectorTableReranker 需要先 bind_vectors")
        rows = np.asarray([self.row_of[product.id] for product in candidates], dtype=np.int64)
        self.last_stats = {"reranker": self.name, "candidates": int(rows.size), "model_inference_ms": 0.0}
        return (self.vectors[rows] @ self.encoder.current).astype(np.float32)


class CrossEncoderTableReranker:
    """cross-encoder 重排，分数从落盘表里查（模型推理解耦，见 ``--rerank-scores``）。"""

    def __init__(self, table: dict[str, dict[str, float]], meta: dict[str, object]) -> None:
        self.name = "cross-encoder"
        self.table = table
        self.meta = meta
        self.last_stats: dict[str, object] = {}

    def score(self, query: str, candidates):
        row = self.table.get(str(self.encoder_position), {})
        # 表里没有的商品给极低分而不是 0：0 会被排到"有真实分数但为负"的前面，
        # 用 -inf 才能保证"没被打过分"的候选一定排在被打过分的候选之后。
        scores = np.asarray(
            [row.get(product.id, float("-inf")) for product in candidates],
            dtype=np.float32,
        )
        self.last_stats = {
            "reranker": self.name,
            "model": self.meta.get("model", ""),
            "candidates": len(candidates),
            "model_inference_ms": 0.0,
        }
        return scores


class TableRerankRetriever:
    """``rerank.RerankRetriever`` 的表格版：粗排 → 查表打分 → 稳定排序。

    与线上一致的关键点：**粗排放宽到 ``candidates`` 再排序**。
    若在粗排阶段就截到 top_k，重排只能在这 k 条里换顺序，永远换不出更好的东西。
    """

    def __init__(self, base, reranker, candidates: int) -> None:
        self.base = base
        self.reranker = reranker
        self.candidates = int(candidates)
        self.products = base.products
        self.catalog_path = base.catalog_path
        self.last_stats: dict[str, object] = {}

    def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
        """返回重排后的前 ``count`` 条——候选级入口必须返回**本层真正的输出**。

        这里刻意不做"透传粗排候选"的实现：透传会让调用方拿到一份没重排过的列表，
        而它以为拿到的是重排结果。评测脚本正是按"有候选级入口就自己切片"的逻辑
        分支的，透传会让重排被静默绕过——指标看起来合理，重排其实没跑。
        重排档的天花板应该看粗排那档的 candidate_recall，不在本档重复量一次。
        """
        return self.search(request.model_copy(update={"top_k": min(int(count), 20)}, deep=False))

    def search(self, request: SearchRequest) -> list[SearchHit]:
        coarse = self.base.search_candidates(request, self.candidates)
        if not coarse:
            self.last_stats = {"rerank": "off", "reranked": 0}
            return coarse
        scores = self.reranker.score(request.query, [hit.product for hit in coarse])
        if scores.shape[0] != len(coarse):
            self.last_stats = {"rerank": "skipped_size_mismatch", "reranked": 0}
            return coarse[: request.top_k]
        reranked = [
            SearchHit(
                product=hit.product,
                score=round(float(score), 4),
                text_score=hit.text_score,
                image_score=hit.image_score,
                reasons=hit.reasons + [f"{self.reranker.name} 重排"],
            )
            for hit, score in zip(coarse, scores)
        ]
        # sort 稳定：不把 id 写进 key，同分就沿用粗排名次。
        reranked.sort(key=lambda hit: hit.score, reverse=True)
        self.last_stats = {"rerank": self.reranker.name, "reranked": len(reranked), **self.reranker.last_stats}
        return reranked[: request.top_k]

    def describe(self) -> dict[str, object]:
        described = getattr(self.base, "describe", None)
        return {
            "candidates": self.candidates,
            "reranker": self.reranker.name,
            "base": described() if callable(described) else {},
        }


class Accumulator:
    """指标累加器。把累加与"算最终数字"分开，避免边跑边除导致的中间态混淆。"""

    def __init__(self, kind: str, top_k: int) -> None:
        self.kind = kind
        self.top_k = top_k
        self.queries = 0
        self.strict = 0
        self.loose = 0
        self.reciprocal = 0.0
        self.dcg = 0.0
        self.latencies: list[float] = []
        self.last_stats: dict[str, object] = {}
        # 候选集召回：真值是否落在粗排给出的候选里。
        # 这是重排的**天花板**——重排只能在候选集内换顺序，进不到候选集的商品永远出不来。
        # 不量它就会把"重排没用"错判成模型不行，而实际可能是候选集本身就没带上真值。
        self.candidate_total = 0
        self.candidate_hits = 0

    def add(
        self,
        strict: int,
        loose: int,
        reciprocal: float,
        dcg: float,
        latency_ms: float,
        stats: dict,
        candidate_hit: bool | None = None,
        candidate_total: int | None = None,
    ) -> None:
        self.queries += 1
        self.strict += strict
        self.loose += loose
        self.reciprocal += reciprocal
        self.dcg += dcg
        self.latencies.append(latency_ms)
        self.last_stats = stats
        if candidate_hit is not None:
            self.candidate_hits += int(candidate_hit)
        if candidate_total is not None:
            self.candidate_total = max(self.candidate_total, int(candidate_total))

    def report(self) -> dict[str, object]:
        total = max(self.queries, 1)
        row = {
            "kind": self.kind,
            "queries": self.queries,
            f"hit@{self.top_k}": round(self.strict / total, 4),
            f"hit@{self.top_k}_category": round(self.loose / total, 4),
            "mrr": round(self.reciprocal / total, 4),
            f"ndcg@{self.top_k}": round(self.dcg / total, 4),
            "latency_p50_ms": round(float(np.percentile(self.latencies, 50)), 3) if self.latencies else 0.0,
            "latency_p95_ms": round(float(np.percentile(self.latencies, 95)), 3) if self.latencies else 0.0,
            "latency_mean_ms": round(statistics.fmean(self.latencies), 3) if self.latencies else 0.0,
            "last_stats": self.last_stats,
        }
        if self.candidate_total:
            row["candidate_window"] = self.candidate_total
            row["candidate_recall"] = round(self.candidate_hits / total, 4)
        return row


def score_query(
    retriever,
    encoder: FixedQueryEncoder,
    position: int,
    query_text: str,
    truth_id: str,
    truth_row: int,
    categories: list[str],
    row_of: dict[str, int],
    top_k: int,
) -> tuple[int, int, float, float, float, bool | None, int | None]:
    """评一条查询：返回 ``(strict, loose, reciprocal, dcg, latency_ms, candidate_hit, candidate_total)``。

    ``candidate_hit`` / ``candidate_total`` 只有在检索器暴露了候选级入口时才有值——
    它们量的是"真值是否落在粗排候选里"，也就是重排能达到的上限。
    """
    encoder.use(position)
    request = SearchRequest(query=query_text, top_k=top_k)
    candidates_of = getattr(retriever, "search_candidates", None)
    start = time.perf_counter()
    if callable(candidates_of):
        coarse = candidates_of(request, 10 * top_k)
        coarse_ids = [hit.product.id for hit in coarse]
        hits = coarse[:top_k]
    else:
        coarse_ids = []
        hits = retriever.search(request)
    latency = (time.perf_counter() - start) * 1000.0

    ids = [hit.product.id for hit in hits]
    rank = next((index for index, product_id in enumerate(ids, start=1) if product_id == truth_id), None)
    truth_category = categories[truth_row]
    same_category = any(
        categories[row_of[product_id]] == truth_category for product_id in ids if product_id in row_of
    )
    strict = 1 if rank is not None else 0
    reciprocal = 1.0 / rank if rank else 0.0
    dcg = 1.0 / float(np.log2(rank + 1)) if rank else 0.0
    loose = 1 if (rank is not None or same_category) else 0
    candidate_hit = truth_id in coarse_ids if coarse_ids else None
    return strict, loose, reciprocal, dcg, latency, candidate_hit, len(coarse_ids) if coarse_ids else None


def sweep_candidate_window(
    retriever,
    encoder: FixedQueryEncoder,
    queries: list[dict],
    windows: list[int],
    top_k: int,
) -> list[dict[str, object]]:
    """扫粗排候选窗口对 hit@k 的影响，回答"窗口该开多大"。

    这条曲线是 A2 期间踩过的坑的正面解法：融合策略用 ``search()`` 评出来的
    hit@10 会随候选窗口变化（窗口=10 时 0.744，放宽到 50+ 后饱和在 0.770），
    同一份代码因此能给出两个都"看起来合理"的数字。把窗口做成显式参数后，
    报告里每个策略的候选窗口都是已知的，比较才有意义。
    """
    candidates_of = getattr(retriever, "search_candidates", None)
    if not callable(candidates_of):
        return []
    rows: list[dict[str, object]] = []
    for window in windows:
        hits = 0
        ceiling = 0
        for position, query_row in enumerate(queries):
            encoder.use(position)
            request = SearchRequest(query=query_row["query"], top_k=top_k)
            candidates = candidates_of(request, window)
            ids = [hit.product.id for hit in candidates]
            if query_row["truth_id"] in ids:
                ceiling += 1
            if query_row["truth_id"] in ids[:top_k]:
                hits += 1
        total = max(len(queries), 1)
        rows.append(
            {
                "window": int(window),
                f"hit@{top_k}": round(hits / total, 4),
                f"candidate_recall@{window}": round(ceiling / total, 4),
            }
        )
    return rows


def collect_rerank_candidates(retriever, encoder, queries: list[dict], count: int) -> list[list[Product]]:
    """按粗排顺序取每条查询的前 ``count`` 条候选——这就是线上重排器的输入。

    这里必须走候选级入口而不是 ``search()``：``search`` 会按 ``request.top_k`` 截断，
    而请求里的 ``top_k`` 是最终要返回的条数（10）。若用它取候选，重排器拿到的就是
    已经被截到 10 条的集合——**重排只能在这 10 条里换顺序，永远换不出更好的东西**，
    而且指标上看起来"重排没效果"，实际是重排的输入本来就是错的。
    """
    candidates_of = getattr(retriever, "search_candidates", None)
    candidates: list[list[Product]] = []
    for position, query_row in enumerate(queries):
        encoder.use(position)
        request = SearchRequest(query=query_row["query"], top_k=query_row["top_k"])
        hits = (
            candidates_of(request, count)
            if callable(candidates_of)
            else retriever.search(request.model_copy(update={"top_k": min(count, 20)}, deep=False))
        )
        candidates.append([hit.product for hit in hits[:count]])
    return candidates


def run_cross_encoder(
    queries: list[dict],
    candidate_lists: list[list[Product]],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
    checkpoint: Path | None = None,
    resume: bool = True,
) -> dict[str, object]:
    """对每条查询的候选集跑 cross-encoder，落盘 ``{query_index: {product_id: score}}``。

    以 product_id 为键而不是下标：候选集的组装顺序会随策略调整，
    用 id 做键才能让这张表在不同实验之间复用。

    **每 25 条 checkpoint 一次**：100 候选 × 500 查询 = 5 万次前向，CPU 上是小时级，
    而这正是"长耗时产物必须先落盘"那条教训的第二次应验（第一次在查询生成阶段，
    91 分钟生成成果因下游编码失败全部清零）。已打分的查询直接跳过，可断点续跑。
    """
    reranker = CrossEncoderReranker(model_name=model_name, device=device, batch_size=batch_size, max_length=max_length)
    scores_by_query: dict[str, dict[str, float]] = {}
    latencies: list[float] = []

    if checkpoint and resume and checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        scores_by_query = payload.get("scores", {})
        print(f"断点续跑：已加载 {len(scores_by_query)} 条查询的打分（{checkpoint.name}）")

    meta: dict[str, object] = {
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
    }

    def flush() -> None:
        if not checkpoint:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps({**meta, "scores": scores_by_query}, ensure_ascii=False), encoding="utf-8"
        )

    for position, (query_row, candidates) in enumerate(zip(queries, candidate_lists)):
        if str(position) in scores_by_query or not candidates:
            continue
        scores = reranker.score(query_row["query"], candidates)
        scores_by_query[str(position)] = {
            product.id: float(score) for product, score in zip(candidates, scores.tolist())
        }
        latencies.append(float(reranker.last_stats.get("model_inference_ms", 0.0)))
        if (position + 1) % 25 == 0:
            flush()
            print(f"  已打分 {position + 1}/{len(queries)}（本条均耗时 {statistics.fmean(latencies):.0f} ms）")
    flush()
    return {
        "meta": {
            **meta,
            "queries_scored": len(scores_by_query),
            "latency_mean_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
            "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        },
        "scores": scores_by_query,
    }


def render_markdown(report: dict[str, object]) -> str:
    config = report["config"]
    top_k = config["top_k"]
    rows = report["results"]
    baseline = next((row for row in rows if row["kind"] == "dense_only"), None)
    lines = [
        "# 混合检索对照实验报告",
        "",
        f"生成时间：{config['generated_at']}",
        "",
        "## 结论先行",
        "",
        f"| 策略 | hit@{top_k} | hit@{top_k}（同 category） | MRR | nDCG@{top_k} | 候选召回 | P50 延迟 | 相对稠密基线 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        if baseline and row["kind"] != "dense_only":
            delta = f"{(row[f'hit@{top_k}'] - baseline[f'hit@{top_k}']) * 100:+.1f}pt"
        else:
            delta = "基线"
        ceiling = f"{row['candidate_recall']:.4f}" if "candidate_recall" in row else "—"
        lines.append(
            f"| `{row['kind']}` | {row[f'hit@{top_k}']:.4f} | {row[f'hit@{top_k}_category']:.4f} | "
            f"{row['mrr']:.4f} | {row[f'ndcg@{top_k}']:.4f} | {ceiling} | "
            f"{row['latency_p50_ms']:.2f} ms | {delta} |"
        )
    lines += [
        "",
        f"语料：**真实向量**（{config['corpus_name']}，{config['dimension']} 维，{config['corpus']} 条），"
        f"查询：**真实中文文本查询** {config['queries']} 条，top_k={top_k}。",
        "",
        "两个 hit 口径必须一起读：严口径（真值 = 生成该查询的原商品）与 A1 的 0.668 直接可比；",
        "宽口径把「top-k 里有同品类商品」也算命中，回答的是「用户想找的东西在不在结果里」。",
        "",
        "**候选召回**是重排能达到的天花板：真值落在粗排候选集里的比例。重排只能在候选集内换顺序，",
        "进不到候选集的商品无论用多强的重排都出不来——所以它和 hit@k 必须一起读。",
    ]
    rerank_row = next((row for row in rows if row["kind"] == "rrf_rerank"), None)
    coarse_row = next((row for row in rows if row["kind"] == "rrf"), None)
    # 天花板取粗排那档：重排档自己量的是"重排后前 N 条里有没有真值"，
    # 那个数被重排结果本身决定，拿它当上限会把"排序失败"算成"召回失败"。
    if rerank_row and coarse_row and "candidate_recall" in coarse_row:
        ceiling = coarse_row["candidate_recall"]
        reached = rerank_row[f"hit@{top_k}"]
        missed = ceiling - reached
        lines += [
            "",
            "## 重排吃到了多少天花板",
            "",
            f"- 候选集含真值：**{ceiling:.4f}**（天花板）",
            f"- 重排后 hit@{top_k}：**{reached:.4f}**",
            f"- 仍未排进 top-{top_k}：**{missed:.4f}**（{missed * 100:.1f}pt）",
            "",
            f"重排把候选集里已召回的命中捞出了 {reached / ceiling * 100:.1f}%。"
            f"剩下的 {missed * 100:.1f}pt 属于两类，处置方式完全不同：",
            "",
            "| 剩余损失的类型 | 含义 | 该动什么 |",
            "|---|---|---|",
            f"| 粗排窗口太窄（真值在 100 名开外） | 召回问题 | 放宽候选窗口、加一路召回 |",
            f"| 真值在候选集里但重排没排进前十 | 排序问题 | 换更强的 cross-encoder、hard-negative 训练 |",
            "",
            "两阶段检索的分工由此明确：**粗排决定上限，重排决定逼近上限的程度**。"
            "重排再强也补不回没进候选集的商品，所以先确认候选集够大再看重排收益。",
        ]
    sweep = config.get("window_sweep") or []
    if sweep:
        lines += [
            "",
            "## 粗排候选窗口扫描",
            "",
            f"| 候选窗口 | hit@{top_k} | 候选召回 |",
            "|---|---|---|",
        ]
        for row in sweep:
            # 键名里嵌 f-string 在 Python 3.12+ 才合法（PEP 701），而 CI 跑 3.11——
            # 嵌套引号必须先算出来，否则本地（3.13）全绿、CI 语法错误。
            window = row["window"]
            recall_key = f"candidate_recall@{window}"
            lines.append(f"| {window} | {row[f'hit@{top_k}']:.4f} | {row[recall_key]:.4f} |")
        lines += [
            "",
            "融合策略的 hit@k **不是策略本身的固定属性**，它随粗排窗口变化：窗口太窄时融合只能看到各路前几名，",
            "融合出的 top-10 自然更差。所以引用融合策略的数字时必须连候选窗口一起给出，",
            "否则同一份代码能支撑两个都看似合理的结论。窗口超过 100 后 hit@k 基本饱和，",
            "继续放宽只增加重排成本、不再增加召回。",
        ]
    lines += [
        "",
        "## 延迟分布",
        "",
        "| 策略 | P50 | P95 | 均值 |",
        "|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['kind']}` | {row['latency_p50_ms']:.2f} ms | {row['latency_p95_ms']:.2f} ms | "
            f"{row['latency_mean_ms']:.2f} ms |"
        )
    lines += ["", "## 配置", "", "```json", json.dumps(config, ensure_ascii=False, indent=2), "```", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="BM25 + 向量 RRF 融合 + 重排的对照实验")
    parser.add_argument("--vectors", required=True, help="语料向量 .npy (N, D)")
    parser.add_argument("--catalog", required=True, help="与向量逐行对齐的商品目录 jsonl")
    parser.add_argument("--queries-file", required=True, help="查询向量 .npy (Q, D)")
    parser.add_argument("--queries-catalog", required=True, help="与查询向量逐行对齐的 jsonl（含 product_id 与 query）")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF 平滑常数，默认 60（RRF 论文与 Elasticsearch 的取值）")
    parser.add_argument("--oversample", type=float, default=5.0, help="融合时的候选窗口倍数")
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--rerank-candidates", type=int, default=100)
    parser.add_argument(
        "--reranker",
        choices=("none", "exact-vector", "cross-encoder"),
        default="none",
        help="叠在 RRF 之上的重排器；cross-encoder 需要 torch + transformers",
    )
    parser.add_argument("--rerank-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--rerank-device", default="cpu")
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=128)
    parser.add_argument("--rerank-scores", default=None, help="已落盘的 cross-encoder 分数表 json（复现用）")
    parser.add_argument("--rerank-score-queries", type=int, default=None, help="只对前 N 条查询跑 cross-encoder（冒烟）")
    parser.add_argument("--save-scores", default=None, help="把本次 cross-encoder 分数写到该路径")
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--kinds", default=",".join(ALL_KINDS), help="要跑的策略，逗号分隔")
    parser.add_argument(
        "--sweep-windows",
        default="",
        help="扫描 RRF 粗排候选窗口对 hit@k 的影响，逗号分隔（如 10,50,100,300）",
    )
    parser.add_argument("--output-dir", default="results/real_text_query")
    args = parser.parse_args()

    paths = {
        "vectors": Path(args.vectors).resolve(),
        "catalog": Path(args.catalog).resolve(),
        "queries": Path(args.queries_file).resolve(),
        "queries_catalog": Path(args.queries_catalog).resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"--{name.replace('_', '-')} 指向的文件不存在：{path}")

    vectors = normalize(np.load(paths["vectors"]))
    query_vectors = normalize(np.load(paths["queries"]))
    if vectors.shape[1] != query_vectors.shape[1]:
        raise SystemExit(f"语料维度 {vectors.shape[1]} 与查询维度 {query_vectors.shape[1]} 不一致")

    products = [Product.model_validate(row) for row in read_jsonl(paths["catalog"])]
    if len(products) != vectors.shape[0]:
        raise SystemExit(f"商品目录 {len(products)} 条与向量 {vectors.shape[0]} 条不一致")

    query_rows = read_jsonl(paths["queries_catalog"])
    if len(query_rows) != query_vectors.shape[0]:
        raise SystemExit(f"查询目录 {len(query_rows)} 条与查询向量 {query_vectors.shape[0]} 条不一致")
    missing_field = [key for key in ("product_id", "query") if key not in query_rows[0]]
    if missing_field:
        raise SystemExit(f"查询目录缺少字段：{missing_field}")

    row_of = {product.id: index for index, product in enumerate(products)}
    categories = [product.category for product in products]
    queries: list[dict] = []
    skipped = 0
    for row in query_rows:
        target = row_of.get(row["product_id"])
        if target is None:
            skipped += 1
            continue
        queries.append({**row, "truth_id": row["product_id"], "truth_row": target, "top_k": args.top_k})
    if not queries:
        raise SystemExit("没有任何一条查询能映射到语料，检查 product_id 是否与 --catalog 对齐")
    print(
        f"语料 {len(products)} 条，查询 {len(queries)} 条"
        f"（真值不在语料内而跳过 {skipped} 条），top_k={args.top_k}"
    )

    encoder = FixedQueryEncoder(query_vectors, products)
    dense = DenseExactRetriever(vectors, products, paths["catalog"], encoder)

    build_start = time.perf_counter()
    bm25 = BM25Retriever.from_products(products, paths["catalog"], k1=args.bm25_k1, b=args.bm25_b)
    bm25_seconds = time.perf_counter() - build_start
    print(f"BM25 倒排构建完成：{bm25.index.summary()}（{bm25_seconds:.2f}s）")

    rrf = ReciprocalRankFusionRetriever(
        [(dense, args.dense_weight), (bm25, args.bm25_weight)],
        rrf_k=args.rrf_k,
        oversample=args.oversample,
    )

    # ---- 重排器：分数来源三选一（现算 / 复用落盘表 / 精确向量表） ----
    rerank_payload: dict[str, object] | None = None
    if args.reranker == "cross-encoder":
        if args.rerank_scores and Path(args.rerank_scores).exists():
            rerank_payload = json.loads(Path(args.rerank_scores).read_text(encoding="utf-8"))
            print(
                f"复用已落盘的 cross-encoder 分数：{len(rerank_payload['scores'])} 条查询"
                f"（{rerank_payload['meta']['model']}）"
            )
        else:
            subset = queries[: args.rerank_score_queries] if args.rerank_score_queries else queries
            print(f"跑 cross-encoder 重排：{args.rerank_model}（{args.rerank_device}），候选 {args.rerank_candidates} 条…")
            staged = ReciprocalRankFusionRetriever(
                [(dense, args.dense_weight), (bm25, args.bm25_weight)],
                rrf_k=args.rrf_k,
                oversample=args.oversample,
            )
            candidate_lists = collect_rerank_candidates(staged, encoder, subset, args.rerank_candidates)
            rerank_payload = run_cross_encoder(
                subset,
                candidate_lists,
                args.rerank_model,
                args.rerank_device,
                args.rerank_batch_size,
                args.rerank_max_length,
                checkpoint=Path(args.save_scores) if args.save_scores else None,
            )
            meta = rerank_payload["meta"]
            print(f"cross-encoder 完成：{meta['queries_scored']} 条查询，均值 {meta['latency_mean_ms']} ms/查询")
            if args.save_scores:
                save_path = Path(args.save_scores)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(json.dumps(rerank_payload, ensure_ascii=False), encoding="utf-8")
                print(f"分数表已写入 {save_path}")

    exact_reranker = ExactVectorTableReranker(products, row_of, encoder)
    exact_reranker.bind_vectors(vectors)

    def cross_table_reranker() -> CrossEncoderTableReranker:
        assert rerank_payload is not None
        reranker = CrossEncoderTableReranker(rerank_payload["scores"], rerank_payload["meta"])
        # encoder_position 是个显式游标而不是从 encoder 内部读，
        # 这样"分数表的下标"与"查询下标"对齐这件事在代码里是可见的、可测的。
        reranker.encoder_position = 0
        return reranker

    requested_kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
    for kind in requested_kinds:
        if kind not in ALL_KINDS:
            raise SystemExit(f"未知策略：{kind}（可选 {', '.join(ALL_KINDS)}）")

    reports: list[dict[str, object]] = []
    cross_reranker = cross_table_reranker() if rerank_payload else None

    for kind in requested_kinds:
        if kind == "dense_only":
            retriever = dense
        elif kind == "bm25_only":
            retriever = bm25
        elif kind == "rrf":
            retriever = rrf
        else:
            if args.reranker == "cross-encoder" and cross_reranker is not None:
                staged = ReciprocalRankFusionRetriever(
                    [(dense, args.dense_weight), (bm25, args.bm25_weight)],
                    rrf_k=args.rrf_k,
                    oversample=args.oversample,
                )
                retriever = TableRerankRetriever(staged, cross_reranker, args.rerank_candidates)
            elif args.reranker == "exact-vector":
                retriever = TableRerankRetriever(rrf, exact_reranker, args.rerank_candidates)
            else:
                print(f"跳过 {kind}：未指定可用的 --reranker")
                continue

        accumulator = Accumulator(kind, args.top_k)
        # cross-encoder 的分数表可能只覆盖部分查询（冒烟跑）。
        # 只评有分数的查询：没被打过分的候选会拿到 -inf，排序退化成粗排原序，
        # 把它算进重排档的指标等于把粗排成绩算成重排的成绩——那是个假数字。
        positions = list(range(len(queries)))
        if kind == "rrf_rerank" and cross_reranker is not None:
            available = {int(key) for key in rerank_payload["scores"]}
            positions = [position for position in positions if position in available]
            if len(positions) < len(queries):
                print(
                    f"  注意：cross-encoder 分数只覆盖 {len(positions)}/{len(queries)} 条查询，"
                    f"该档指标只在这 {len(positions)} 条上统计"
                )
        for position in positions:
            if cross_reranker is not None and kind == "rrf_rerank":
                cross_reranker.encoder_position = position
            strict, loose, reciprocal, dcg, latency, candidate_hit, candidate_total = score_query(
                retriever,
                encoder,
                position,
                queries[position]["query"],
                queries[position]["truth_id"],
                queries[position]["truth_row"],
                categories,
                row_of,
                args.top_k,
            )
            accumulator.add(
                strict,
                loose,
                reciprocal,
                dcg,
                latency,
                dict(getattr(retriever, "last_stats", {}) or {}),
                candidate_hit,
                candidate_total,
            )
        row = accumulator.report()
        reports.append(row)
        ceiling = f" cand_recall={row['candidate_recall']:.4f}" if "candidate_recall" in row else ""
        print(
            f"{kind:12s} n={row['queries']:<4d} hit@{args.top_k}={row[f'hit@{args.top_k}']:.4f} "
            f"cat={row[f'hit@{args.top_k}_category']:.4f} mrr={row['mrr']:.4f} "
            f"p50={row['latency_p50_ms']:.2f}ms{ceiling}"
        )

    window_sweep: list[dict[str, object]] = []
    if args.sweep_windows:
        windows = [int(value) for value in args.sweep_windows.split(",") if value.strip()]
        window_sweep = sweep_candidate_window(rrf, encoder, queries, windows, args.top_k)
        for row in window_sweep:
            recall_key = f"candidate_recall@{row['window']}"
            print(
                f"  window={row['window']:<4d} hit@{args.top_k}={row[f'hit@{args.top_k}']:.4f} "
                f"cand_recall={row[recall_key]:.4f}"
            )

    report = {
        "config": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus": len(products),
            "corpus_name": paths["vectors"].name,
            "dimension": int(vectors.shape[1]),
            "queries": len(queries),
            "queries_skipped_unmapped": skipped,
            "queries_file": paths["queries"].name,
            "top_k": args.top_k,
            "rrf_k": args.rrf_k,
            "oversample": args.oversample,
            "dense_weight": args.dense_weight,
            "bm25_weight": args.bm25_weight,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "bm25_build_seconds": round(bm25_seconds, 3),
            "bm25_summary": bm25.index.summary(),
            "reranker": args.reranker,
            "rerank_candidates": args.rerank_candidates,
            "cross_encoder": rerank_payload["meta"] if rerank_payload else None,
            "kinds": [row["kind"] for row in reports],
            "window_sweep": window_sweep,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "results": reports,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "hybrid_retrieval_benchmark.json"
    md_path = output_dir / "hybrid_retrieval_benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"\n报告已写入：\n  {md_path}\n  {json_path}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
