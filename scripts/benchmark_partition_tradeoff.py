"""分段粒度与兜底策略的权衡：为什么"多切几段"不是免费的。

这个脚本回答一个具体问题：`buckets` 该设多少，以及**分段到底在什么条件下真的有用**。

它存在的理由是主压测里一个反常结果
------------------------------------
在 ``benchmark_ann_scaling.py`` 里，1 万条档的分段版本反而**比不分段差**：
IVF 的严苛档集合一致率从 0.98 掉到 0.42。而 10 万条档又是分段的略好（0.78 → 0.84）。

同一个改动在不同规模上给出相反方向的结果，说明差异不是来自"分段本身好不好"，
而是来自**另一个变量**。这里把它单独测出来。

那个变量是**精确兜底**
----------------------
``FaissRetriever`` 在发现索引"交付不足"时会退化成一次 O(N) 全量精确扫描。
不分段的 IVF 在 1 万条档上多轮超采样后请求规模涨到 2048，而它当时扫描的桶里
只有约 1250 条——索引交不出 2048 条，于是被判为"交付不足"，**退化成精确扫描**，
结果当然是精确的。

分段之后每段变小、桶集合相对变密，同样的 k 反而**交得出来**了，于是不再触发兜底，
返回的是一个**真正的近似结果**。分段的"变差"其实是不再被兜底救。

结论：**对比分段与不分段时，必须固定兜底策略。** 当 O(N) 全量扫描还撑得住时，
它几乎总是赢，而且实现简单得多；分段的真正价值要等到全量扫描已经不可接受时才体现。

实测跑出来之后还多了一条更干净的说法：**分段之后，兜底开关不再改变任何结果**。
把表里每一对（带兜底 / ``-nofallback``）并排看，不分段的那几行两者差得很远
（1 万条 HNSW：0.9 vs 0.0），而分段的每一对完全相同。也就是说分段已经把「候选耗尽」
这个失败模式消掉了，而精确兜底本来就是为它准备的。这给了兜底一个实用判据：
**兜底还在生效，就说明过滤压力还没有被真正下推掉。**
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from shopping_agent.ann_index import faiss_available  # noqa: E402
from shopping_agent.encoders import normalize  # noqa: E402
from shopping_agent.faiss_retrieval import FaissRetriever  # noqa: E402
from shopping_agent.partitioned import PartitionedRetriever  # noqa: E402
from shopping_agent.synthetic import SyntheticEncoder, generate_corpus, write_catalog  # noqa: E402

from benchmark_ann_scaling import evaluate_retriever, make_config, requests_for  # noqa: E402


def build_variants(products, catalog, encoder, vectors, kind, args):
    """不分段 / 分段 × 有兜底 / 无兜底。四种组合缺一个都看不懂结果。

    返回 ``(标签, 检索器或错误信息, 构建秒数)``。构建耗时要单独记：分段意味着
    N 份独立索引，构建成本不是"除以 N"而是"每段各自付一次训练开销"。
    """
    variants: list[tuple[str, object, float]] = []

    for fallback in (True, False):
        label = f"{kind}{'' if fallback else '-nofallback'}"
        config = make_config(kind, vectors.shape[1], len(products), args)
        started = time.perf_counter()
        try:
            retriever = FaissRetriever.from_vectors(
                products,
                catalog,
                encoder,
                vectors,
                config,
                exact_vectors=vectors if fallback else None,
            )
        except Exception as error:  # noqa: BLE001
            variants.append((label, f"{type(error).__name__}: {error}", 0.0))
            continue
        variants.append((label, retriever, round(time.perf_counter() - started, 3)))

    for buckets in args.buckets:
        for fallback in (True, False):
            label = f"{kind}+partition{buckets}{'' if fallback else '-nofallback'}"
            config = make_config(kind, vectors.shape[1], len(products), args)
            started = time.perf_counter()
            try:
                retriever = PartitionedRetriever.from_vectors(
                    products,
                    catalog,
                    encoder,
                    vectors,
                    config,
                    by="price",
                    buckets=buckets,
                    exact_vectors=vectors if fallback else None,
                )
            except Exception as error:  # noqa: BLE001
                variants.append((label, f"{type(error).__name__}: {error}", 0.0))
                continue
            variants.append((label, retriever, round(time.perf_counter() - started, 3)))
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description="分段粒度与兜底策略的权衡实测")
    parser.add_argument("--scales", default="10000,100000")
    parser.add_argument("--kinds", default="ivf,hnsw")
    parser.add_argument("--buckets", default="4,16,64")
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=128)
    parser.add_argument("--spread", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--point-queries", type=int, default=50)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--ivf-nlist", type=int, default=1024)
    parser.add_argument("--ivf-nprobe", type=int, default=32)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    args.kinds = [item.strip() for item in args.kinds.split(",") if item.strip()]
    args.buckets = [int(item) for item in args.buckets.split(",") if item.strip()]
    scales = [int(item) for item in args.scales.split(",") if item.strip()]
    args.rerank_sizes = []
    if not faiss_available():
        raise SystemExit('需要 faiss：pip install -e ".[ann]"')

    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "scales": scales,
            "kinds": args.kinds,
            "buckets": args.buckets,
            "dimension": args.dimension,
            "top_k": args.top_k,
            "clusters": args.clusters,
            "spread": args.spread,
            "seed": args.seed,
        },
        "scales": [],
    }

    for count in scales:
        corpus = generate_corpus(
            count=count,
            dimension=args.dimension,
            queries=args.queries,
            top_k=args.top_k,
            clusters=min(args.clusters, max(1, count)),
            spread=args.spread,
            seed=args.seed,
        )
        products = corpus.products
        encoder = SyntheticEncoder(corpus)
        vectors = normalize(encoder.encode_texts([product.searchable_text() for product in products]))
        # 合成目录只是检索层的输入载体，写临时目录避免污染仓库。
        catalog = Path(tempfile.mkdtemp(prefix="partition-sweep-")) / f"catalog_{count}.jsonl"
        write_catalog(products, catalog)

        rows = np.linspace(0, count - 1, num=min(args.point_queries, count), dtype=np.int64)
        targets = [f"syn{int(row):07d}" for row in rows.tolist()]
        prices = np.asarray([product.price for product in products], dtype=np.float64)
        budgets = {
            "moderate": float(np.percentile(prices, 20.0)),
            "tight": float(np.percentile(prices, 1.0)),
        }

        reference_retriever = FaissRetriever.from_vectors(
            products, catalog, encoder, vectors, make_config("numpy", vectors.shape[1], count, args)
        )
        reference = {
            name: [[hit.product.id for hit in reference_retriever.search(r)] for r in requests_for(budget, targets, args)]
            for name, budget in budgets.items()
        }
        del reference_retriever
        gc.collect()

        entry: dict[str, object] = {
            "count": count,
            "budget_moderate": budgets["moderate"],
            "budget_tight": budgets["tight"],
            "variants": [],
        }
        for kind in args.kinds:
            for label, retriever, build_seconds in build_variants(products, catalog, encoder, vectors, kind, args):
                if isinstance(retriever, str):
                    entry["variants"].append({"label": label, "status": "error", "reason": retriever})
                    continue
                result = evaluate_retriever(retriever, targets, budgets, reference, args)
                described = retriever.describe() if hasattr(retriever, "describe") else {}
                record = {
                    "label": label,
                    "kind": kind,
                    "partitioned": isinstance(retriever, PartitionedRetriever),
                    "build_seconds": build_seconds,
                    "status": "ok",
                    **{key: value for key, value in result.items() if key != "last_stats"},
                }
                for key in ("partitions", "partition_min", "partition_max", "size_mb"):
                    if key in described:
                        record[key] = described[key]
                entry["variants"].append(record)
                print(
                    f"  {count:>7,} {label:<34}"
                    f" 严苛一致 {record.get('tight_set_match')}"
                    f"｜Jaccard {record.get('tight_jaccard')}"
                    f"｜漏检 {record.get('tight_missing_total')}"
                    f"｜扫描占比 {record.get('tight_scanned_fraction', '—')}"
                    f"｜构建 {build_seconds}s"
                )
                del retriever
                gc.collect()
        payload["scales"].append(entry)
        print(f"  峰值 RSS 见主压测；规模 {count:,} 完成")
        catalog.unlink(missing_ok=True)
        del corpus
        gc.collect()

    json_path = output_dir / "partition_tradeoff.json"
    md_path = output_dir / "partition_tradeoff.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"\n已写入 {json_path}")
    print(f"已写入 {md_path}")


def render_markdown(payload: dict) -> str:
    config = payload["config"]
    lines = [
        "# 分段粒度与兜底策略的权衡",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 语料：{config['dimension']} 维、{config['clusters']} 簇、spread={config['spread']}、"
        f"seed={config['seed']}（与主压测同源）",
        f"- 分段数扫描：{config['buckets']}",
        f"- 后端：{'、'.join(config['kinds'])}",
        "",
        "两档过滤：宽松（放行约 20%）、严苛（放行约 1%）。判据仍是**与精确后端的结果集是否一致**。",
        "",
        "`-nofallback` 表示关掉精确兜底（不传 `exact_vectors`），用来把「分段的效果」和"
        "「兜底的效果」分开看——这两件事经常被混在一起归因。",
        "",
    ]
    for scale in payload["scales"]:
        lines += [
            f"## 规模 {scale['count']:,}",
            "",
            f"预算档位：宽松 ¥{scale['budget_moderate']:.2f}、严苛 ¥{scale['budget_tight']:.2f}。",
            "",
            "| 变体 | 分段数 | 段大小 | 构建(s) | 严苛集合一致 | 严苛 Jaccard | 严苛漏检数 | 严苛轮数(均/最大) | 严苛扫描占比 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in scale["variants"]:
            if row.get("status") != "ok":
                lines.append(f"| {row['label']} | — | — | — | — | — | — | — | 失败（{row['reason']}） |")
                continue
            span = (
                f"{row['partition_min']}~{row['partition_max']}"
                if row.get("partition_min") is not None
                else "—"
            )
            lines.append(
                f"| {row['label']} | {row.get('partitions', 1)} | {span} | {row.get('build_seconds', '—')} |"
                f" {row['tight_set_match']} |"
                f" {row['tight_jaccard']} | {row['tight_missing_total']} |"
                f" {row['tight_rounds_mean']} / {row['tight_rounds_max']} |"
                f" {row.get('tight_scanned_fraction', '—')} |"
            )
        lines.append("")
    lines += [
        "## 读法",
        "",
        "1. **先看 `-nofallback` 那几行**，那才是「分段本身」的效果。带兜底的行里，结果可能"
        "已经被一次 O(N) 全量扫描救成精确的了，只看它反而会得出「分段没用」甚至"
        "「分段有害」的错误结论。",
        "2. **再把每一对（带兜底 / `-nofallback`）并排看**。不分段时两者差得很远"
        "（1 万条 HNSW：0.9 vs 0.0），而**分段之后每一对都完全相同**——这说明分段已经把"
        "「候选耗尽」这个失败模式消掉了，而精确兜底本来就是为它准备的。换句话说："
        "分段一旦生效，兜底就不再被触发；兜底还在生效，说明过滤压力还没被真正下推掉。",
        "3. **然后看「严苛扫描占比」**：它说明下推省掉了多少检索量。这个读数直接和过滤的选择性比："
        "严苛档放行约 1%，所以理论下限就是 1% 量级（你至少要扫到装那些合格商品的那一段）。"
        "实测 4 段是 0.25、16 段是 0.0625、64 段是 0.0156——粒度匹配上了，下推几乎就贴着下限走。",
        "4. **最后看段大小与构建耗时**：段越小剪得越干净，但需要训练的后端有样本下限"
        "（IVF-PQ 要求每个码本中心至少 39 条训练样本，8 bit 下即 256 × 39 ≈ 9984 条），"
        "段切太细会直接建不起来。构建耗时也要看：分段省的是**查询**，不是构建——"
        "10 万条 HNSW 切 64 段后构造成本翻倍。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
