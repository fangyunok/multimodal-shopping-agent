from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import ShoppingAgent
from .encoders import ChineseClipEncoder
from .evaluation import evaluate
from .indexing import build_index
from .models import AgentRequest
from .retrieval import HybridRetriever
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.build_index:
        manifest = build_index(args.catalog, args.build_index, ChineseClipEncoder(args.model, args.device), args.model)
        print(manifest.model_dump_json(indent=2))
        return
    agent = ShoppingAgent(ShoppingTools(HybridRetriever.from_jsonl(args.catalog)))
    if args.evaluate:
        print(json.dumps(evaluate(agent, args.evaluate).as_dict(), ensure_ascii=False, indent=2))
        return
    response = agent.run(AgentRequest(query=args.query or "", image_path=args.image, max_price=args.max_price))
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
