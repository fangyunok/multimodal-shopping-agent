from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import ShoppingAgent
from .abo import convert_abo, fetch_abo_subset_images
from .benchmark import evaluate_cases, generate_attribute_cases, read_benchmark, write_benchmark
from .encoders import ChineseClipEncoder
from .dataset import prepare_dataset
from .evaluation import evaluate
from .fusion_retrieval import ScoreFusionRetriever
from .indexing import build_index
from .models import AgentRequest
from .retrieval import HybridRetriever
from .retrieval_evaluation import evaluate_retrieval
from .semantic_retrieval import SemanticRetriever
from .tools import ShoppingTools


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
    parser.add_argument("--retriever-backend", choices=("baseline", "clip", "fusion"), default="baseline")
    parser.add_argument("--lexical-weight", type=float, default=0.65)
    parser.add_argument("--index-dir")
    parser.add_argument("--prepare-data", metavar="CSV_OR_JSONL")
    parser.add_argument("--output-catalog", default="data/processed/products.jsonl")
    parser.add_argument("--image-dir", default="data/processed/images")
    parser.add_argument("--convert-abo", metavar="EXTRACTED_ABO_ROOT")
    parser.add_argument("--abo-limit", type=int, default=1000)
    parser.add_argument("--abo-output", default="data/raw/abo.jsonl")
    parser.add_argument("--fetch-abo-images", metavar="ABO_ROOT")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--evaluate-retrieval", choices=("text", "image"))
    parser.add_argument("--build-benchmark", metavar="OUTPUT_JSONL")
    parser.add_argument("--benchmark", metavar="BENCHMARK_JSONL")
    args = parser.parse_args()
    if args.fetch_abo_images:
        summary = fetch_abo_subset_images(args.fetch_abo_images, args.abo_limit, args.download_workers)
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
    if args.retriever_backend in {"clip", "fusion"}:
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
    agent = ShoppingAgent(ShoppingTools(retriever))
    if args.evaluate:
        print(json.dumps(evaluate(agent, args.evaluate).as_dict(), ensure_ascii=False, indent=2))
        return
    response = agent.run(AgentRequest(query=args.query or "", image_path=args.image, max_price=args.max_price))
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
