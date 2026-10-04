from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import ShoppingAgent
from .abo import build_cross_view_benchmark, convert_abo, fetch_abo_subset_images
from .ann_index import AnnIndexConfig
from .benchmark import evaluate_cases, generate_attribute_cases, read_benchmark, write_benchmark
from .encoders import ChineseClipEncoder
from .dataset import prepare_dataset
from .evaluation import evaluate
from .faiss_retrieval import FaissRetriever
from .fusion_retrieval import ScoreFusionRetriever
from .incremental import (
    IndexVersionManager,
    build_incremental_version,
    load_changes,
)
from .llm_planner import LLMPlanner
from .indexing import build_ann_index, build_index, encode_products, load_catalog
from .models import AgentRequest
from .partitioned import PartitionedRetriever
from .planner import RulePlanner
from .planner_evaluation import evaluate_planner
from .react import ReActAgent
from .retrieval import HybridRetriever
from .retrieval_evaluation import evaluate_retrieval
from .semantic_retrieval import SemanticRetriever
from .session import ContextBudget, SessionMemory
from .tools import ShoppingTools


def run_chat(agent: ReActAgent) -> None:
    """多轮会话模式：一条 session 贯穿全程，每轮重算上下文预算。"""
    print("多轮会话模式（会话记忆 + 上下文预算 + ReAct）。输入 exit 退出。")
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in {"exit", "quit", ":q"}:
            break
        if not line:
            continue
        result = agent.run(line)
        print(f"助手> {result.answer}")
        stats = result.context
        print(
            f"  [意图 {result.intent}｜工具调用 {result.tool_calls} 次｜终止 {result.stop_reason}"
            f"｜上下文 {stats['tokens']}/{stats['max_tokens']} tokens"
            f"｜丢弃 {stats['dropped_turns']} 条]"
        )
    print("会话结束。")


def ann_config_from(args) -> AnnIndexConfig:
    """把 CLI 参数收成一个索引配置。

    维度只是占位：``build_ann_index`` 会以**真实向量维度**为准覆盖它，
    因为维度由编码器决定，而不是用户可以选的东西。
    """
    return AnnIndexConfig(
        kind=args.ann_kind,
        dimension=args.ann_dimension,
        m=args.ann_m,
        ef_construction=args.ann_ef_construction,
        ef_search=args.ann_ef_search,
        nlist=args.ann_nlist,
        nprobe=args.ann_nprobe,
        pq_segments=args.ann_pq_segments,
        pq_bits=args.ann_pq_bits,
    )


def rerank_kwargs(args) -> dict[str, int]:
    """重排是可选开关：只在显式给了候选数时才传下去。

    不传（而不是统一传 0）保留了默认值的单一来源——默认值只该在 ``FaissRetriever``
    里定义一次，否则改默认值就会漏掉所有手工传参的调用点。
    """
    return {"rerank_candidates": args.rerank_candidates} if args.rerank_candidates else {}


def handle_index_versioning(args) -> bool:
    """索引版本管理：建版本 → 校验 → 发布 → 回滚 → 清理。返回 True 表示已处理完。"""
    manager = IndexVersionManager(args.index_root)
    if args.index_status:
        print(json.dumps(manager.status(), ensure_ascii=False, indent=2))
        return True
    if args.rollback_index:
        target = manager.rollback()
        print(json.dumps({"rolled_back_to": target, **manager.status()}, ensure_ascii=False, indent=2))
        return True
    if args.promote_version:
        directory = manager.promote(args.promote_version)
        print(json.dumps({"promoted": args.promote_version, "directory": str(directory)}, ensure_ascii=False, indent=2))
        return True
    if args.prune_versions is not None:
        removed = manager.prune(args.prune_versions)
        print(json.dumps({"removed": removed, **manager.status()}, ensure_ascii=False, indent=2))
        return True
    return False


def handle_incremental(args) -> bool:
    """增量更新：基于当前版本 + 变更日志产出新版本，可选立刻发布。"""
    if not args.apply_changes:
        return False
    manager = IndexVersionManager(args.index_root)
    changes = load_changes(args.apply_changes)
    if args.from_current:
        base = manager.current_dir()
    elif args.base_version_dir:
        base = Path(args.base_version_dir).resolve()
    else:
        raise SystemExit("增量更新需要 --from-current 或 --base-version-dir 指定基线版本")
    version = manager.next_version()
    target = manager.version_dir(version)
    encoder = ChineseClipEncoder(args.model, args.device, args.batch_size)
    manifest, stats = build_incremental_version(
        base,
        args.catalog,
        target,
        changes,
        encoder,
        config=ann_config_from(args),
        model_name=args.model_id,
        image_weight=args.image_weight,
    )
    summary = {
        "version": version,
        "directory": str(target),
        "products": manifest.product_count,
        "index_type": manifest.index_type,
        **stats,
    }
    if args.publish:
        manager.promote(version)
        summary["published"] = True
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or evaluate the shopping agent")
    parser.add_argument("--catalog", default="data/products.jsonl")
    parser.add_argument("--evaluate", metavar="JSONL")
    parser.add_argument("--query")
    parser.add_argument("--image")
    parser.add_argument("--max-price", type=float)
    parser.add_argument("--build-index", metavar="DIRECTORY")
    parser.add_argument("--model", default="OFA-Sys/chinese-clip-vit-base-patch16")
    parser.add_argument("--model-id", default="OFA-Sys/chinese-clip-vit-base-patch16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-weight", type=float, default=0.5)
    parser.add_argument(
        "--retriever-backend",
        choices=("baseline", "clip", "fusion", "ann", "partitioned"),
        default="baseline",
    )
    parser.add_argument("--lexical-weight", type=float, default=0.65)
    parser.add_argument("--index-dir")
    parser.add_argument("--build-ann-index", metavar="DIRECTORY", help="按 --ann-* 参数构建指定类型的 ANN 索引")
    parser.add_argument(
        "--ann-kind", choices=("numpy", "flat", "hnsw", "ivf", "ivfpq"), default="hnsw", help="ANN 后端类型"
    )
    parser.add_argument("--ann-dimension", type=int, default=512, help="占位维度，构建时会以真实向量维度覆盖")
    parser.add_argument("--ann-m", type=int, default=32, help="HNSW 连接度")
    parser.add_argument("--ann-ef-construction", type=int, default=200, help="HNSW 建图宽度")
    parser.add_argument("--ann-ef-search", type=int, default=64, help="HNSW 查询宽度（可直接调，无需重建）")
    parser.add_argument("--ann-nlist", type=int, default=1024, help="IVF 桶数（数据量小时会自动收缩）")
    parser.add_argument("--ann-nprobe", type=int, default=32, help="IVF 查询扫描桶数")
    parser.add_argument("--ann-pq-segments", type=int, default=None, help="PQ 子空间数，默认按维度自动选择")
    parser.add_argument("--ann-pq-bits", type=int, default=8, help="PQ 每段编码位数")
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=0,
        help="粗召回 + 精确重排的候选数（0 表示关闭）。开启后分数改为用原始向量重算，"
        "需要索引保留 vectors.npy（即不要用 --no-vectors）",
    )
    parser.add_argument(
        "--build-partitioned-index",
        metavar="DIRECTORY",
        help="按 --partition-* 分段构建索引（过滤下推），与 --build-ann-index 互不依赖",
    )
    parser.add_argument(
        "--partition-by",
        choices=("price", "category", "category_price"),
        default="price",
        help="分段维度；应当选**过滤条件用到的**那一维，否则路由剪不掉任何段",
    )
    parser.add_argument(
        "--partition-buckets",
        type=int,
        default=8,
        help="分段数（按分位数切，每段商品数相近）。段切得越细，IVF-PQ 这类需要训练的后端越容易建不动",
    )
    parser.add_argument("--no-vectors", action="store_true", help="不落盘 vectors.npy（省磁盘，但失去精确兜底）")
    parser.add_argument("--catalog-snapshot", action="store_true", help="把商品目录快照写进索引目录（版本自证）")
    parser.add_argument("--index-root", default="data/index", help="索引版本根目录")
    parser.add_argument("--index-status", action="store_true", help="查看当前索引版本状态")
    parser.add_argument("--promote-version", type=int, help="原子发布指定版本")
    parser.add_argument("--rollback-index", action="store_true", help="回滚到上一个索引版本")
    parser.add_argument("--prune-versions", type=int, metavar="KEEP", help="清理旧版本，保留 KEEP 个")
    parser.add_argument("--apply-changes", metavar="CHANGES_JSONL", help="增量应用的变更日志")
    parser.add_argument("--from-current", action="store_true", help="以当前发布版本为增量基线")
    parser.add_argument("--base-version-dir", help="增量基线版本目录")
    parser.add_argument("--publish", action="store_true", help="增量建完后立即发布")
    parser.add_argument("--planner-backend", choices=("rule", "llm"), default="rule")
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--evaluate-planner", metavar="JSONL")
    parser.add_argument("--prepare-data", metavar="CSV_OR_JSONL")
    parser.add_argument("--output-catalog", default="data/processed/products.jsonl")
    parser.add_argument("--image-dir", default="data/processed/images")
    parser.add_argument("--convert-abo", metavar="EXTRACTED_ABO_ROOT")
    parser.add_argument("--abo-limit", type=int, default=1000)
    parser.add_argument("--abo-output", default="data/raw/abo.jsonl")
    parser.add_argument("--fetch-abo-images", metavar="ABO_ROOT")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--include-other-images", action="store_true")
    parser.add_argument("--build-cross-view-benchmark", metavar="ABO_ROOT")
    parser.add_argument("--cross-view-output", default="outputs/cross_view_benchmark.jsonl")
    parser.add_argument("--evaluate-retrieval", choices=("text", "image"))
    parser.add_argument("--build-benchmark", metavar="OUTPUT_JSONL")
    parser.add_argument("--benchmark", metavar="BENCHMARK_JSONL")
    parser.add_argument("--chat", action="store_true", help="进入多轮会话模式（会话记忆 + 上下文预算 + ReAct）")
    parser.add_argument("--max-steps", type=int, default=6, help="ReAct 单轮最大步数")
    parser.add_argument("--context-budget", type=int, default=2048, help="会话上下文 token 预算上限")
    args = parser.parse_args()
    if handle_index_versioning(args) or handle_incremental(args):
        return
    if args.build_cross_view_benchmark:
        summary = build_cross_view_benchmark(args.build_cross_view_benchmark, args.catalog, args.cross_view_output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.fetch_abo_images:
        summary = fetch_abo_subset_images(
            args.fetch_abo_images, args.abo_limit, args.download_workers,
            include_other_images=args.include_other_images,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.convert_abo:
        summary = convert_abo(args.convert_abo, args.abo_output, args.abo_limit)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.prepare_data:
        summary = prepare_dataset(args.prepare_data, args.output_catalog, args.image_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.build_index:
        encoder = ChineseClipEncoder(args.model, args.device, args.batch_size)
        manifest = build_index(args.catalog, args.build_index, encoder, args.model_id, args.image_weight)
        print(manifest.model_dump_json(indent=2))
        return
    if args.build_ann_index:
        encoder = ChineseClipEncoder(args.model, args.device, args.batch_size)
        manifest, ann = build_ann_index(
            args.catalog,
            args.build_ann_index,
            encoder,
            args.model_id,
            ann_config_from(args),
            image_weight=args.image_weight,
            keep_vectors=not args.no_vectors,
            catalog_snapshot=args.catalog_snapshot,
        )
        print(manifest.model_dump_json(indent=2))
        print(json.dumps(ann.summary(), ensure_ascii=False, indent=2))
        return
    if args.build_partitioned_index:
        encoder = ChineseClipEncoder(args.model, args.device, args.batch_size)
        catalog = Path(args.catalog).resolve()
        products = load_catalog(catalog)
        vectors = encode_products(products, catalog, encoder, args.image_weight)
        retriever = PartitionedRetriever.from_vectors(
            products,
            catalog,
            encoder,
            vectors,
            ann_config_from(args),
            by=args.partition_by,
            buckets=args.partition_buckets,
            image_weight=args.image_weight,
            model_name=args.model_id,
            **rerank_kwargs(args),
        )
        files = retriever.save(args.build_partitioned_index, catalog_snapshot=args.catalog_snapshot)
        print(json.dumps({"files": files, **retriever.describe()}, ensure_ascii=False, indent=2, default=str))
        return
    if args.retriever_backend == "partitioned":
        if not args.index_dir:
            parser.error("--retriever-backend partitioned 需要 --index-dir")
        encoder = ChineseClipEncoder(args.model, args.device, args.batch_size)
        retriever = PartitionedRetriever.from_index(
            args.catalog, args.index_dir, encoder, ann_config_from(args), **rerank_kwargs(args)
        )
    elif args.retriever_backend == "ann":
        if not args.index_dir:
            parser.error("--retriever-backend ann 需要 --index-dir")
        encoder = ChineseClipEncoder(args.model, args.device, args.batch_size)
        retriever = FaissRetriever.from_index(
            args.catalog, args.index_dir, encoder, args.model_id, ann_config_from(args), **rerank_kwargs(args)
        )
    elif args.retriever_backend in {"clip", "fusion"}:
        encoder = ChineseClipEncoder(args.model, args.device, args.batch_size)
        semantic = (
            SemanticRetriever.from_index(args.catalog, args.index_dir, encoder, args.model_id)
            if args.index_dir
            else SemanticRetriever.from_jsonl(args.catalog, encoder)
        )
        retriever = ScoreFusionRetriever(HybridRetriever.from_jsonl(args.catalog), semantic, args.lexical_weight) if args.retriever_backend == "fusion" else semantic
    else:
        retriever = HybridRetriever.from_jsonl(args.catalog)
    if args.build_benchmark:
        test_products = [product for product in retriever.products if product.split == "test"]
        if not test_products:
            test_products = retriever.products
        cases = generate_attribute_cases(test_products)
        write_benchmark(cases, args.build_benchmark)
        print(json.dumps({"cases": len(cases), "output": args.build_benchmark}, ensure_ascii=False, indent=2))
        return
    if args.benchmark:
        print(json.dumps(evaluate_cases(retriever, read_benchmark(args.benchmark)), ensure_ascii=False, indent=2))
        return
    if args.evaluate_retrieval:
        evaluation_products = [product for product in retriever.products if product.split in (None, "test")]
        print(json.dumps(evaluate_retrieval(retriever, evaluation_products, args.evaluate_retrieval), ensure_ascii=False, indent=2))
        return
    tools = ShoppingTools(retriever)
    categories = [product.category for product in retriever.products]
    if args.planner_backend == "llm":
        if not args.llm_base_url or not args.llm_model:
            parser.error("--planner-backend llm 需要 --llm-base-url 和 --llm-model")
        planner = LLMPlanner(categories, args.llm_base_url, args.llm_model, os.getenv("LLM_API_KEY", ""))
    else:
        planner = RulePlanner(categories)
    if args.evaluate_planner:
        print(json.dumps(evaluate_planner(planner, args.evaluate_planner), ensure_ascii=False, indent=2))
        return
    if args.chat:
        budget = ContextBudget(max_tokens=args.context_budget)
        run_chat(ReActAgent(tools, max_steps=args.max_steps, budget=budget, session=SessionMemory(budget=budget)))
        return
    agent = ShoppingAgent(tools, planner)
    if args.evaluate:
        print(json.dumps(evaluate(agent, args.evaluate).as_dict(), ensure_ascii=False, indent=2))
        return
    response = agent.run(AgentRequest(query=args.query or "", image_path=args.image, max_price=args.max_price))
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
