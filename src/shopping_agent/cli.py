from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import ShoppingAgent
from .evaluation import evaluate
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
    args = parser.parse_args()
    agent = ShoppingAgent(ShoppingTools(HybridRetriever.from_jsonl(args.catalog)))
    if args.evaluate:
        print(json.dumps(evaluate(agent, args.evaluate).as_dict(), ensure_ascii=False, indent=2))
        return
    response = agent.run(AgentRequest(query=args.query or "", image_path=args.image, max_price=args.max_price))
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()

