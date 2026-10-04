"""规模化压测：把"该换哪种 ANN 索引"从直觉变成一组实测数字。

测四件事
--------
1. **索引层**：同一批向量分别用 numpy / flat / hnsw / ivf / ivfpq 建索引，
   量构建耗时、索引体积、单条查询 P50/P95/P99、批量吞吐，以及相对精确解的 recall@k。
   没有 recall 这一列，延迟对比毫无意义——把候选砍掉一半当然更快。
2. **检索层**：把 ``FaissRetriever`` 接上 ``SyntheticEncoder`` 跑端到端点名查询，
   验证换后端之后排序结果没有劣化。
3. **过滤压力**：把预算卡到只通过 1% 商品，观察自适应超采样的轮数与扫描量。
   这是"ANN + 业务过滤"最容易出错的地方，必须单独量化。
4. **重排与下推**：给每个索引补三列——"多取候选再精确重排"能把 recall 拉回多少
   （对应 ``rerank_candidates``），以及按价位分段后过滤能省掉多少检索量
   （对应 ``PartitionedRetriever``）。

用法
----
    python scripts/benchmark_ann_scaling.py                        # 默认 100 ~ 10 万，CPU 即可
    python scripts/benchmark_ann_scaling.py --scales 1000000       # 百万级（内存 ≥ 16 GB）
    python scripts/benchmark_ann_scaling.py --scales 10000 --kinds numpy,hnsw,ivfpq
    python scripts/benchmark_ann_scaling.py --partition-kinds ivf,ivfpq --partition-buckets 4
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.ann_index import AnnIndex, AnnIndexConfig, faiss_available  # noqa: E402
from shopping_agent.encoders import normalize  # noqa: E402
from shopping_agent.faiss_retrieval import FaissRetriever  # noqa: E402
from shopping_agent.indexing import load_catalog  # noqa: E402
from shopping_agent.models import SearchRequest  # noqa: E402
from shopping_agent.partitioned import PartitionedRetriever  # noqa: E402
from shopping_agent.synthetic import (  # noqa: E402
    SyntheticCorpus,
    SyntheticEncoder,
    generate_corpus,
    query_for,
    write_catalog,
)

ALL_KINDS = ("numpy", "flat", "hnsw", "ivf", "ivfpq")


def optional_psutil():
    try:
        import psutil  # noqa: PLC0415

        return psutil
    except ImportError:  # pragma: no cover - 取决于环境
        return None


def rss_mb() -> float | None:
    psutil = optional_psutil()
    if psutil is None:
        return None
    return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)


def make_config(kind: str, dimension: int, count: int, args: argparse.Namespace) -> AnnIndexConfig:
    """按规模推导参数：桶数、efSearch 这些旋钮在小数据上按原值配是没意义的。"""
    effective_nlist = max(1, count // 39)
    return AnnIndexConfig(
        kind=kind,  # type: ignore[arg-type]
        dimension=dimension,
        m=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction,
        ef_search=args.hnsw_ef_search,
        nlist=min(args.ivf_nlist, max(1, effective_nlist)),
        nprobe=args.ivf_nprobe,
        pq_bits=args.pq_bits,
        seed=args.seed,
    )


def percentile(values: list[float], point: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), point))


def measure_latency(ann: AnnIndex, queries: np.ndarray, top_k: int, warmup: int = 3) -> dict[str, float]:
    """单条查询延迟：这是真实请求路径（一次请求一条 query），不是批量离线任务。"""
    for row in range(min(warmup, queries.shape[0])):
        ann.search_one(queries[row], top_k)
    samples: list[float] = []
    for row in range(queries.shape[0]):
        start = time.perf_counter()
        ann.search_one(queries[row], top_k)
        samples.append((time.perf_counter() - start) * 1000.0)
    mean = statistics.fmean(samples)
    return {
        "p50_ms": round(percentile(samples, 50), 3),
        "p95_ms": round(percentile(samples, 95), 3),
        "p99_ms": round(percentile(samples, 99), 3),
        "mean_ms": round(mean, 3),
        "max_ms": round(max(samples), 3),
        "qps_single": round(1000.0 / mean, 1) if mean else float("nan"),
    }


def measure_throughput(ann: AnnIndex, queries: np.ndarray, top_k: int, batch: int = 32) -> float:
    """批量吞吐：反映离线补数或服务端合并请求时的真实能力。"""
    if queries.shape[0] < batch:
        return float("nan")
    batches = [queries[start : start + batch] for start in range(0, queries.shape[0] - batch + 1, batch)]
    chunks = batches[: max(1, len(batches) // 4)]  # 取前 1/4 批次做预热
    for chunk in chunks:
        ann.search(chunk, top_k)
    start = time.perf_counter()
    for chunk in batches:
        ann.search(chunk, top_k)
    elapsed = time.perf_counter() - start
    return round(len(batches) * batch / elapsed, 1) if elapsed else float("nan")


def rerank_exact(vectors: np.ndarray, queries: np.ndarray, candidates: np.ndarray, top_k: int) -> np.ndarray:
    """用原始向量对候选集精确重算分数——"粗召回 + 精确重排"的后半段。

    PQ 压掉的正是这一步需要的精度，所以重排必须拿得到原始向量。
    这也说明索引体积和 ``vectors.npy`` 是一对必须一起算的账：省下的 180 MB，
    是用这 2 GB 副本换来的。
    """
    output = np.full((queries.shape[0], top_k), -1, dtype=np.int64)
    for row in range(queries.shape[0]):
        kept = candidates[row][candidates[row] >= 0]
        if kept.size == 0:
            continue
        scores = vectors[kept] @ queries[row]
        order = np.argsort(-scores, kind="stable")[:top_k]
        output[row, : order.size] = kept[order]
    return output


def exact_neighbors(vectors: np.ndarray, queries: np.ndarray, top_k: int) -> np.ndarray:
    """精确 top-k 真值。向量已 L2 归一化，所以矩阵内积就是余弦相似度。"""
    scores = queries @ vectors.T
    k = min(top_k, int(vectors.shape[0]))
    if k <= 0:
        return np.empty((queries.shape[0], 0), dtype=np.int64)
    # argpartition 只保证"前 k 大都在这一批里"，不保证有序；要拿到真正的 top-k 顺序得再排一次。
    coarse = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, coarse], axis=1)
    return np.asarray(coarse[rows, order], dtype=np.int64)


def load_real_corpus(
    vectors_path: Path, catalog_path: Path | None, count: int, args: argparse.Namespace
) -> tuple[SyntheticCorpus, dict]:
    """把外部真实向量装进与合成语料**完全相同**的容器——判据一致，结果才可比。

    切分规则：语料取前缀 ``[:count]``，查询取紧接其后的 ``args.queries`` 条，两者不重叠。
    真实商品向量自带强自相关，一旦让查询命中它自己，recall 会被抬到一个没有意义的高位；
    这和仓库里批判过的"原图搜原图"是同一个坑。
    """
    raw = np.load(vectors_path)
    if raw.ndim != 2:
        raise ValueError(f"向量文件应为二维 (N, D)，实际是 {raw.shape}")
    vectors = normalize(np.asarray(raw, dtype=np.float32))
    needed = count + args.queries
    if vectors.shape[0] < needed:
        raise ValueError(
            f"向量文件只有 {vectors.shape[0]} 条，切不出 {count} 条语料 + {args.queries} 条查询"
            f"（需要 {needed} 条）。导出时多留查询余量，或调小 --scales / --queries"
        )
    corpus_vectors = vectors[:count]
    query_vectors = vectors[count:needed]
    ground_truth = exact_neighbors(corpus_vectors, query_vectors, args.top_k)
    products = load_catalog(catalog_path)[:count] if catalog_path is not None else []
    corpus = SyntheticCorpus(
        vectors=corpus_vectors,
        queries=query_vectors,
        ground_truth=ground_truth,
        products=products,
        seed=args.seed,
        clusters=0,
        spread=0.0,
    )
    summary = {
        "source": "real-vectors",
        "name": vectors_path.name,
        "vectors": int(corpus_vectors.shape[0]),
        "dimension": int(corpus_vectors.shape[1]),
        "queries": int(query_vectors.shape[0]),
        "raw_vectors_mb": round(corpus_vectors.nbytes / (1024 * 1024), 3),
    }
    return corpus, summary


def benchmark_index(kind: str, corpus, args: argparse.Namespace) -> dict:
    config = make_config(kind, corpus.dimension, corpus.count, args)
    before = rss_mb()
    start = time.perf_counter()
    try:
        ann = AnnIndex.build(corpus.vectors, config)
    except Exception as error:  # noqa: BLE001 - 压测脚本需要把失败原因带进报告而不是中断
        return {"kind": kind, "status": "error", "reason": f"{type(error).__name__}: {error}"}
    build_seconds = time.perf_counter() - start
    after = rss_mb()

    latency = measure_latency(ann, corpus.queries, args.top_k)
    throughput = measure_throughput(ann, corpus.queries, args.top_k, args.batch)
    _, indices = ann.search(corpus.queries, args.top_k)
    recall = corpus.recall(indices, args.top_k)

    # 重排对照直接复用刚建好的索引：另建一份会把压测时间翻倍，而重排本身不需要新索引。
    rerank_rows = [{"candidates": args.top_k, "recall": round(float(recall), 4)}]
    for size in args.rerank_sizes:
        k = min(corpus.count, int(size))
        if k <= args.top_k:
            continue
        _, candidates = ann.search(corpus.queries, k)
        reranked = rerank_exact(corpus.vectors, corpus.queries, candidates, args.top_k)
        rerank_rows.append({"candidates": k, "recall": round(corpus.recall(reranked, args.top_k), 4)})

    result = {
        "kind": kind,
        "status": "ok",
        "describe": config.describe(),
        "effective": ann.effective,
        "build_seconds": round(build_seconds, 3),
        "index_mb": ann.size_mb,
        "bytes_per_vector": round(ann.index_bytes / max(corpus.count, 1), 1),
        "rss_delta_mb": round(after - before, 1) if before is not None and after is not None else None,
        "recall_at_k": round(float(recall), 4),
        "rerank": rerank_rows,
        "qps_batch32": throughput,
        **latency,
    }
    del ann
    gc.collect()
    return result


def requests_for(budget: float, targets: list[str], args: argparse.Namespace) -> list[SearchRequest]:
    return [SearchRequest(query=query_for(target), max_price=budget, top_k=args.top_k) for target in targets]


def evaluate_retriever(retriever, targets: list[str], budgets: dict[str, float], reference: dict, args) -> dict:
    """点名查询 + 两档过滤压力。普通后端与分段后端共用这套判据，才谈得上对比。"""
    top1 = 0
    for target in targets:
        hits = retriever.search(SearchRequest(query=query_for(target), top_k=1))
        if hits and hits[0].product.id == target:
            top1 += 1

    entry: dict[str, object] = {"status": "ok", "point_query_top1": round(top1 / max(len(targets), 1), 4)}
    for name, budget in budgets.items():
        exact_sets = [set(ids) for ids in reference[name]]
        exact_sizes = [len(ids) for ids in reference[name]]
        rounds: list[int] = []
        scanned: list[float] = []
        exact_match = 0
        jaccard = 0.0
        size_gap = 0
        for request, truth, truth_size in zip(requests_for(budget, targets, args), exact_sets, exact_sizes):
            hits = retriever.search(request)
            got = {hit.product.id for hit in hits}
            rounds.append(int(retriever.last_stats.get("rounds", 0)))
            if got == truth:
                exact_match += 1
            union = got | truth
            jaccard += len(got & truth) / len(union) if union else 1.0
            size_gap += max(0, truth_size - len(got))
            fraction = retriever.last_stats.get("scanned_fraction")
            if fraction is not None:
                scanned.append(float(fraction))
        total = max(len(targets), 1)
        entry[f"{name}_set_match"] = round(exact_match / total, 4)
        entry[f"{name}_jaccard"] = round(jaccard / total, 4)
        entry[f"{name}_missing_total"] = size_gap
        entry[f"{name}_rounds_mean"] = round(statistics.fmean(rounds), 2) if rounds else 0
        entry[f"{name}_rounds_max"] = int(max(rounds)) if rounds else 0
        if scanned:
            # 分段索引的关键读数：**这个档位下**下推省掉了多少比例的检索量。
            # 必须分档位报：两个档位混在一起取均值，会得到一个谁也对不上的数
            # （严苛档 1.6% 与宽松档 20% 平均成 11%，看着像 bug）。
            entry[f"{name}_scanned_fraction"] = round(statistics.fmean(scanned), 4)
    entry["last_stats"] = retriever.last_stats
    return entry


def benchmark_retriever(corpus, catalog_path: Path, args: argparse.Namespace) -> dict:
    """端到端：编码 → ANN（→ 选段）→ 业务过滤 → 排序。

    查询是"点名式"的（文本里带 ``syn0001234``），因此标准答案确定，
    可以在没有 GPU、没有真实模型的情况下验证换后端后排序是否仍然正确。

    过滤压力测试的判据刻意不是"是否凑满 top_k"，而是**与精确后端的结果集是否一致**：
    在只放行 1% 商品的目录上，精确检索本身也凑不满 top_k，用"凑满"当标准会得出
    "精确后端也不合格"的荒谬结论。正确的问题是：ANN 有没有因为过滤而漏掉本该返回的商品。

    分段后端（``<kind>+partition``）走的是同一套判据，因此两者的差距可以直接读出来——
    不是"分段更快"，而是"分段在同样的过滤下还能不能与精确结果对齐"。
    """
    encoder = SyntheticEncoder(corpus)
    # 商品、向量、目录只准备一次：五个后端共用，否则十万级光重复编码就会盖过索引开销。
    products = corpus.products
    catalog_path = write_catalog(products, catalog_path)
    vectors = normalize(encoder.encode_texts([product.searchable_text() for product in products]))
    rows = np.linspace(0, corpus.count - 1, num=min(args.point_queries, corpus.count), dtype=np.int64)
    targets = [f"syn{int(row):07d}" for row in rows.tolist()]
    report: dict[str, object] = {
        "catalog": str(catalog_path),
        "probes": len(targets),
        "budget_moderate": None,
        "budget_tight": None,
        "backends": [],
    }

    prices = np.asarray([product.price for product in corpus.products], dtype=np.float64)
    budgets = {
        "moderate": float(np.percentile(prices, 20.0)),  # 放行约 20%，精确检索理应能凑满 top_k
        "tight": float(np.percentile(prices, 1.0)),  # 放行约 1%，逼出超采样
    }
    report["budget_moderate"] = budgets["moderate"]
    report["budget_tight"] = budgets["tight"]

    # 参考基线：精确后端在同过滤条件下的结果。ANN 要与它对齐，而不是与"top_k"对齐。
    reference_retriever = FaissRetriever.from_vectors(
        products, catalog_path, encoder, vectors, make_config("numpy", corpus.dimension, corpus.count, args)
    )
    reference = {
        name: [
            [hit.product.id for hit in reference_retriever.search(request)]
            for request in requests_for(budget, targets, args)
        ]
        for name, budget in budgets.items()
    }
    del reference_retriever
    gc.collect()

    for kind in args.kinds:
        if kind not in ALL_KINDS:
            continue
        config = make_config(kind, corpus.dimension, corpus.count, args)
        try:
            retriever = FaissRetriever.from_vectors(products, catalog_path, encoder, vectors, config)
        except Exception as error:  # noqa: BLE001
            report["backends"].append({"kind": kind, "status": "error", "reason": f"{type(error).__name__}: {error}"})
            continue
        entry = {"kind": kind, **evaluate_retriever(retriever, targets, budgets, reference, args)}
        report["backends"].append(entry)
        del retriever
        gc.collect()

    for kind in [item for item in args.partition_kinds if item in ALL_KINDS]:
        label = f"{kind}+partition"
        try:
            retriever = PartitionedRetriever.from_vectors(
                products,
                catalog_path,
                encoder,
                vectors,
                make_config(kind, corpus.dimension, corpus.count, args),
                by=args.partition_by,
                buckets=args.partition_buckets,
            )
        except Exception as error:  # noqa: BLE001
            # 分段后每段样本变少，IVF-PQ 这类需要训练的后端可能直接建不起来。
            # 这不是脚本出错，而是"分段有下限"这个事实本身，必须出现在报告里。
            report["backends"].append({"kind": label, "status": "error", "reason": f"{type(error).__name__}: {error}"})
            continue
        entry = {"kind": label, **evaluate_retriever(retriever, targets, budgets, reference, args)}
        described = retriever.describe()
        entry["partitions"] = described["partitions"]
        entry["partition_min"] = described["partition_min"]
        entry["partition_max"] = described["partition_max"]
        report["backends"].append(entry)
        del retriever
        gc.collect()
    return report


def render_markdown(payload: dict) -> str:
    config = payload["config"]
    if config.get("corpus") == "real":
        dimension = (
            payload["scales"][0]["corpus"]["dimension"] if payload["scales"] else config["dimension"]
        )
        corpus_line = (
            f"- 语料：**真实向量**（{config.get('corpus_name', 'vectors.npy')}，{dimension} 维；"
            f"{config['queries']} 条查询从向量文件末尾切出，与语料不重叠）"
        )
    else:
        corpus_line = (
            f"- 语料：合成语料，{config['dimension']} 维、{config['clusters']} 个簇、"
            f"spread={config['spread']}、seed={config['seed']}"
        )
    lines = [
        "# ANN 规模化压测结果",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 环境：Python {payload['environment']['python']}｜numpy {payload['environment']['numpy']}"
        f"｜faiss {payload['environment']['faiss']}｜{payload['environment']['platform']}",
        corpus_line,
        f"- 查询：{config['queries']} 条/规模，recall@{config['top_k']}",
        "",
        "## 索引层",
        "",
        "| 规模 | 索引 | 构建(s) | 索引体积(MB) | P50(ms) | P95(ms) | P99(ms) | 单条 QPS | 批量 QPS | recall@k |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for scale in payload["scales"]:
        for row in scale["results"]:
            if row.get("status") != "ok":
                lines.append(
                    f"| {scale['count']:,} | {row['kind']} | — | — | — | — | — | — | — | 失败（{row.get('reason', '')}） |"
                )
                continue
            lines.append(
                f"| {scale['count']:,} | {row['kind']} | {row['build_seconds']} | {row['index_mb']} |"
                f" {row['p50_ms']} | {row['p95_ms']} | {row['p99_ms']} | {row['qps_single']} |"
                f" {row['qps_batch32']} | {row['recall_at_k']} |"
            )
    lines += ["", "### 实际生效参数", ""]
    for scale in payload["scales"]:
        for row in scale["results"]:
            if row.get("status") == "ok":
                lines.append(f"- {scale['count']:,} × {row['kind']}：{row['describe']}｜effective={row['effective']}")

    rerank_rows = [
        (scale["count"], row)
        for scale in payload["scales"]
        for row in scale["results"]
        if row.get("status") == "ok" and row.get("rerank")
    ]
    if rerank_rows:
        header = [item["candidates"] for item in rerank_rows[-1][1]["rerank"]]
        lines += [
            "",
            "### 粗召回 + 精确重排",
            "",
            "压缩索引（尤其 IVF-PQ）给出的**分数是有偏的，但候选集是好的**。"
            "下表是「多取候选、再用原始向量精确算分」之后 recall@k 的变化，列头是取的候选数。",
            "",
            "代价必须一起读：重排需要原始向量，也就是 ``vectors.npy``。"
            "PQ 省下的那 180 MB，是用这份副本换回来的。",
            "",
            "| 规模 | 索引 | " + " | ".join(f"候选 {size}" for size in header) + " |",
            "| --- | --- | " + " | ".join("---" for _ in header) + " |",
        ]
        for count, row in rerank_rows:
            values = {item["candidates"]: item["recall"] for item in row["rerank"]}
            cells = " | ".join(str(values.get(size, "—")) for size in header)
            lines.append(f"| {count:,} | {row['kind']} | {cells} |")

    retriever_sections = [
        (scale["count"], scale["retriever"]) for scale in payload["scales"] if scale.get("retriever")
    ]
    if retriever_sections:
        lines += [
            "",
            "## 检索层（端到端 + 过滤压力）",
            "",
            "探针是点名查询（文本里带商品 token），编码 → ANN → 业务过滤 → 排序全链路走一遍。",
            "两个过滤档位：宽松档放行约 20% 商品，严苛档放行约 1%。",
            "",
            "判据是**与精确后端的结果集是否一致**，而不是「是否凑满 top_k」——"
            "在只放行 1% 商品的目录上，精确检索本身也凑不满 top_k。",
            "正确的问题是：ANN 有没有因为过滤而漏掉本该返回的商品。",
            "",
            "``+partition`` 后缀是**过滤下推**版本：同样后端，但按价位切成若干段，"
            "查询时先按 ``max_price`` 选段，只在选中的段里检索。它和同名前缀的普通后端"
            "是同一套判据，所以两者的差距可以直接读出来。",
            "",
            "「扫描商品占比」= 实际参与检索的商品数 / 全库商品数，**分档位给出**："
            "严苛档的读数才说明下推在最难的情形下省了多少。普通后端没有这个读数（标 —）——"
            "它们扫的是整份索引，而 IVF 这类只扫部分桶，「扫了多少商品」并不是一份整份索引能回答的问题。",
            "",
            "分段的行读的是**一个固定操作点**（价位分段、``buckets`` 见配置文件节）。"
            "这个操作点会明显改变结论——粒度扫描与「兜底掩盖了分段效果」的实测见 "
            "``results/partition_tradeoff.md``。",
            "",
            "本表所有后端都**开着精确兜底**（默认保留向量），所以行与行之间是可比的；"
            "但这也意味着某一行的「好」可能来自一次 O(N) 全量扫描，而不是 ANN 本身——"
            "把兜底这个变量单独剥离，同样见 ``results/partition_tradeoff.md``。",
            "",
            "> 小规模下所有后端当然都是 1.0（候选远远够用）。这张表的价值在**大规模那一行**："
            "只有当过滤把候选打到只剩个位数时，自适应超采样才是真的在被考验。",
        ]
        for count, retriever in retriever_sections:
            lines += [
                "",
                f"### 规模 {count:,}（{retriever['probes']} 条探针）",
                "",
                f"预算档位：宽松 ¥{retriever.get('budget_moderate', '—')}、严苛 ¥{retriever.get('budget_tight', '—')}。",
                "",
                "| 后端 | 点名 top1 | 宽松集合一致 | 宽松 Jaccard | 宽松漏检数 | 严苛集合一致 | 严苛 Jaccard |"
                " 严苛漏检数 | 严苛超采样轮数(均/最大) | 严苛扫描占比 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for row in retriever["backends"]:
                if row.get("status") != "ok":
                    lines.append(
                        f"| {row['kind']} | — | — | — | — | — | — | — | 失败（{row.get('reason', '')}） | — |"
                    )
                    continue
                lines.append(
                    f"| {row['kind']} | {row['point_query_top1']} | {row['moderate_set_match']} |"
                    f" {row['moderate_jaccard']} | {row['moderate_missing_total']} | {row['tight_set_match']} |"
                    f" {row['tight_jaccard']} | {row['tight_missing_total']} |"
                    f" {row['tight_rounds_mean']} / {row['tight_rounds_max']} |"
                    f" {row.get('tight_scanned_fraction', '—')} |"
                )
    lines += [
        "",
        "> recall 相对**精确检索真值**计算；索引体积取 FAISS 序列化后的字节数（numpy 后端取矩阵 nbytes）。",
        "> 单条 QPS 反映在线请求延迟，批量 QPS 反映离线补数或服务端合并请求时的吞吐。",
        "> 分段索引会把每段缩小 ``buckets`` 倍，因此需要训练的后端（IVF-PQ）在段内样本不足时会直接构建失败——"
        "报告里的「失败」是分段有下限这个事实，不是脚本异常。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ANN 索引规模化压测")
    parser.add_argument("--scales", default="100,1000,10000,100000")
    parser.add_argument("--kinds", default=",".join(ALL_KINDS))
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--spread", type=float, default=0.5, help="簇内相对散度；见 synthetic._draw 的说明")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--point-queries", type=int, default=50)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--ivf-nlist", type=int, default=1024)
    parser.add_argument(
        "--ivf-nprobe", type=int, default=32, help="默认 32/1024≈3%% 的桶；调大召回更高、延迟更差"
    )
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument(
        "--rerank-sizes",
        default="10,100,500",
        help="重排对照的候选数，逗号分隔；留空可跳过（重排复用已建好的索引，几乎不增加耗时）",
    )
    parser.add_argument(
        "--partition-kinds",
        default="ivf,ivfpq",
        help="额外跑一遍分段索引的后端，逗号分隔；留空可跳过。"
        "默认选这两个是因为它们正是过滤压力下最先失守的后端",
    )
    parser.add_argument("--partition-by", default="price", choices=("price", "category", "category_price"))
    parser.add_argument(
        "--partition-buckets",
        type=int,
        default=16,
        help="分段数。默认 16 是折中：严苛档放行约 1%% 的商品，与它匹配的粒度约在百段量级，"
        "但段越小越容易撞上需要训练的后端（IVF-PQ 每段至少要 39 × 2^bits 条样本）而直接建不动。"
        "这个操作点会明显影响分段的效果，粒度扫描见 scripts/benchmark_partition_tradeoff.py",
    )
    parser.add_argument("--skip-retriever", action="store_true", help="只压索引层，跳过端到端点名与过滤压力")
    parser.add_argument(
        "--vectors",
        default=None,
        help="外部真实向量 .npy（N×D，如 Chinese-CLIP 导出的商品向量）。给了它就不再生成合成语料，"
        "改用真实 embedding 分布压测——这是验证「合成语料的簇结构假设是否成立」的唯一方式",
    )
    parser.add_argument("--vectors-catalog", default=None, help="与 --vectors 逐行对齐的商品目录 jsonl（可选）")
    parser.add_argument(
        "--vectors-query-count",
        type=int,
        default=0,
        help="从向量文件末尾切多少条做查询；默认与 --queries 相同。查询段与语料段不重叠",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    args.kinds = [kind.strip() for kind in args.kinds.split(",") if kind.strip()]
    args.rerank_sizes = [int(item) for item in args.rerank_sizes.split(",") if item.strip()]
    args.partition_kinds = [kind.strip() for kind in args.partition_kinds.split(",") if kind.strip()]
    scales = [int(item) for item in args.scales.split(",") if item.strip()]
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_vectors = Path(args.vectors).resolve() if args.vectors else None
    real_catalog = Path(args.vectors_catalog).resolve() if args.vectors_catalog else None
    if real_vectors is not None and not real_vectors.exists():
        raise SystemExit(f"--vectors 指向的文件不存在：{real_vectors}")
    if args.vectors_query_count:
        args.queries = args.vectors_query_count
    if real_vectors is not None and not args.skip_retriever:
        # 检索层探针靠 SyntheticEncoder 的商品 token 定位答案，真实向量没有这套 token，
        # 硬跑只会得到一堆"点名 top1 = 0"的伪失败。真实模式只压索引层。
        print("提示：--vectors 模式自动跳过检索层（探针依赖合成 token），只压索引层")
        args.skip_retriever = True

    faiss_ok = faiss_available()
    if not faiss_ok:
        print("提示：未检测到 faiss，本次只会压 numpy 精确后端。安装：pip install -e \".[ann]\"")
    active_kinds = [kind for kind in args.kinds if faiss_ok or kind == "numpy"]

    payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "faiss": "absent",
            "platform": f"{platform.system()} {platform.machine()}",
        },
        "config": {
            "scales": scales,
            "corpus": "real" if real_vectors is not None else "synthetic",
            "corpus_name": real_vectors.name if real_vectors is not None else "",
            "dimension": args.dimension,
            "queries": args.queries,
            "top_k": args.top_k,
            "clusters": args.clusters,
            "spread": args.spread,
            "seed": args.seed,
            "kinds": active_kinds,
            "rerank_sizes": args.rerank_sizes,
            "partition_by": args.partition_by,
            "partition_buckets": args.partition_buckets,
            "partition_kinds": args.partition_kinds,
        },
        "scales": [],
    }
    if faiss_ok:
        from shopping_agent.ann_index import faiss_module

        payload["environment"]["faiss"] = faiss_module().__version__  # type: ignore[index]

    for count in scales:
        print(f"\n=== 规模 {count:,} ===")
        if real_vectors is not None:
            corpus, corpus_summary = load_real_corpus(real_vectors, real_catalog, count, args)
        else:
            corpus = generate_corpus(
                count=count,
                dimension=args.dimension,
                queries=args.queries,
                top_k=args.top_k,
                clusters=min(args.clusters, max(1, count)),
                spread=args.spread,
                seed=args.seed,
            )
            corpus_summary = corpus.summarize()
        print(f"  语料：{corpus_summary}")
        entry: dict[str, object] = {"count": count, "corpus": corpus_summary, "results": []}
        for kind in active_kinds:
            result = benchmark_index(kind, corpus, args)
            entry["results"].append(result)  # type: ignore[union-attr]
            if result.get("status") == "ok":
                print(
                    f"  {kind:<7} 构建 {result['build_seconds']:>8.2f}s"
                    f"｜{result['index_mb']:>9.2f} MB"
                    f"｜P50 {result['p50_ms']:>8.3f}ms"
                    f"｜P99 {result['p99_ms']:>8.3f}ms"
                    f"｜recall@{args.top_k} {result['recall_at_k']}"
                )
            else:
                print(f"  {kind:<7} 跳过：{result.get('reason')}")
        if not args.skip_retriever and corpus.products:
            # 合成目录只是检索层的输入载体，写到临时目录避免污染仓库。
            catalog_path = Path(tempfile.mkdtemp(prefix="ann-bench-")) / f"synthetic_catalog_{count}.jsonl"
            retriever_report = benchmark_retriever(corpus, catalog_path, args)
            entry["retriever"] = retriever_report  # type: ignore[assignment]
            # 每档规模都留一份：过滤压力在小规模上恒等于 1.0，只有大规模那档才说明问题。
            # 同时把最新（即最大规模）那份挂到顶层，方便只想看一张表的人。
            payload["retriever"] = retriever_report
            payload["retriever_scale"] = count
            catalog_path.unlink(missing_ok=True)
        payload["scales"].append(entry)  # type: ignore[union-attr]
        print(f"  峰值 RSS：{rss_mb()} MB")
        del corpus
        gc.collect()

    json_path = output_dir / "ann_scaling_benchmark.json"
    md_path = output_dir / "ann_scaling_benchmark.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"\n已写入 {json_path}")
    print(f"已写入 {md_path}")


if __name__ == "__main__":
    main()
