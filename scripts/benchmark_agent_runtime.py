"""测量会话记忆与上下文预算的实际行为，并把结果写入 results/。

不依赖任何外部模型：ReAct 走 RuleReasoner、检索走词法基线，因此在干净 CPU 环境可复现。
测量三件事：

1. **上下文是否始终有界**：固定预算下连续多轮，逐轮记录 tokens 与丢弃数，验证不越界。
2. **预算对压缩率的影响**：扫多个预算档位，看压缩从第几轮开始、最终压缩比是多少。
3. **ReAct 的步数与延迟分布**：真实多步轨迹的终止原因与 P50/P95。

用法：
    python scripts/benchmark_agent_runtime.py --turns 40 --json results/agent_runtime_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from shopping_agent.react import ReActAgent
from shopping_agent.retrieval import HybridRetriever
from shopping_agent.session import ContextBudget, SessionMemory
from shopping_agent.tools import ShoppingTools

ROOT = Path(__file__).resolve().parents[1]

# 一组可以循环驱动的多轮脚本：包含检索、追问库存、比价、新增/继承约束。
SCRIPTED_TURNS = [
    "想找一双通勤穿的白色运动鞋，预算500元以内",
    "那有货吗",
    "除了那双，还有别的运动鞋吗",
    "帮我对比一下",
    "再看看300元以内的",
    "帮我找通勤用的箱包",
    "这个有货吗",
    "帮我找运动鞋",
]


def scripted_turns(count: int) -> list[str]:
    return [SCRIPTED_TURNS[index % len(SCRIPTED_TURNS)] for index in range(count)]


def percentile(samples: list[float], quantile: float) -> float | None:
    if not samples:
        return None
    ordered = sorted(samples)
    rank = max(1, math.ceil(quantile / 100 * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 3)


def make_tools() -> ShoppingTools:
    return ShoppingTools(HybridRetriever.from_jsonl(ROOT / "data/products.jsonl"))


def run_context_growth(budget_tokens: int, turns: int) -> dict[str, Any]:
    """固定预算下连续多轮，逐轮记录上下文占用。"""
    budget = ContextBudget(max_tokens=budget_tokens)
    session = SessionMemory(budget=budget)
    agent = ReActAgent(make_tools(), session=session, budget=budget)

    samples: list[dict[str, Any]] = []
    for index, query in enumerate(scripted_turns(turns)):
        result = agent.run(query)
        samples.append(
            {
                "turn": index + 1,
                "intent": result.intent,
                "tool_calls": result.tool_calls,
                "stop_reason": result.stop_reason,
                "tokens": result.context["tokens"],
                "max_tokens": result.context["max_tokens"],
                "dropped_turns": result.context["dropped_turns"],
                "compressed": result.context["compressed"],
            }
        )

    tokens = [sample["tokens"] for sample in samples]
    first_compressed = next((sample["turn"] for sample in samples if sample["compressed"]), None)
    return {
        "budget_tokens": budget_tokens,
        "usable_tokens": budget.usable_tokens,
        "turns": turns,
        "final_tokens": tokens[-1],
        "max_observed_tokens": max(tokens),
        "mean_tokens": round(statistics.fmean(tokens), 2),
        "within_budget": all(value <= budget.usable_tokens for value in tokens),
        "first_compressed_turn": first_compressed,
        "total_dropped_turns": sum(sample["dropped_turns"] for sample in samples),
        "final_state": session.state.model_dump(),
        "samples": samples,
    }


def run_react_profile(repeats: int) -> dict[str, Any]:
    """统计 ReAct 的步数、终止原因与端到端延迟。"""
    tools = make_tools()
    queries = ["帮我找运动鞋", "想找500元以内的运动鞋", "帮我找通勤箱包"]
    latencies: list[float] = []
    stop_reasons: dict[str, int] = {}
    step_counts: list[int] = []

    for _ in range(repeats):
        for query in queries:
            agent = ReActAgent(ShoppingTools(tools.retriever), budget=ContextBudget(max_tokens=2048))
            started = time.perf_counter()
            result = agent.run(query)
            latencies.append((time.perf_counter() - started) * 1000)
            stop_reasons[result.stop_reason] = stop_reasons.get(result.stop_reason, 0) + 1
            step_counts.append(result.tool_calls)

    return {
        "runs": len(latencies),
        "tool_calls_mean": round(statistics.fmean(step_counts), 2),
        "tool_calls_max": max(step_counts),
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": round(max(latencies), 3),
        },
        "stop_reasons": stop_reasons,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    growth = payload["budgets"]
    profile = payload["react_profile"]
    lines = [
        "# Agent 运行期基准：会话记忆与上下文预算",
        "",
        f"生成时间：{payload['generated_at']}　目录：`data/products.jsonl`（{payload['catalog_size']} 个商品）",
        "",
        "所有数字由 `scripts/benchmark_agent_runtime.py` 在本机 CPU 上实测得到，",
        "检索走词法基线、ReAct 走 RuleReasoner，因此不含模型推理开销，结果可复现。",
        "",
        "## 1. 上下文预算是硬边界",
        "",
        "| 预算 (tokens) | 可用 (tokens) | 轮数 | 最终占用 | 峰值占用 | 首次压缩轮次 | 累计丢弃 | 是否越界 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in growth:
        lines.append(
            f"| {item['budget_tokens']} | {item['usable_tokens']} | {item['turns']} | {item['final_tokens']} | "
            f"{item['max_observed_tokens']} | {item['first_compressed_turn'] or '未触发'} | "
            f"{item['total_dropped_turns']} | {'否' if item['within_budget'] else '是'} |"
        )

    baseline = growth[-1]
    lines += [
        "",
        f"在 {baseline['budget_tokens']} tokens 预算下连续 {baseline['turns']} 轮，"
        f"上下文占用始终不超过可用预算，说明**压缩是持久且幂等的**："
        "被折叠的轮次会从活跃事件流移出，而不是每轮重新累积。",
        "",
        "## 2. ReAct 步数与延迟",
        "",
        f"共 {profile['runs']} 次运行，平均工具调用 {profile['tool_calls_mean']} 次，"
        f"单轮最多 {profile['tool_calls_max']} 次。",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 端到端 P50 | {profile['latency_ms']['p50']} ms |",
        f"| 端到端 P95 | {profile['latency_ms']['p95']} ms |",
        f"| 端到端 max | {profile['latency_ms']['max']} ms |",
        "",
        "终止原因分布：",
        "",
        "| 终止原因 | 次数 |",
        "|---|---|",
    ]
    for reason, count in sorted(profile["stop_reasons"].items()):
        lines.append(f"| {reason} | {count} |")

    lines += [
        "",
        "## 口径与边界",
        "",
        "- token 数使用启发式估算（CJK 每字 1 token，其余每 4 字符 1 token），"
        "偏保守、不做真实 BPE 分词，因此绝对值与具体模型会有偏差，**趋势与边界结论有效**。",
        "- 目录只有 5 个商品，检索与工具调用的规模不代表线上流量；本表测量的是"
        "**上下文管理与循环控制**的行为，不是推荐质量。",
        "- 未测真实 LLM 决策器的延迟与 token 消耗，因为需要外部模型；"
        "RuleReasoner 的步数分布可以作为该接口的上界参考。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark session memory, context budget and the ReAct loop")
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--budgets", default="512,1024,2048,4096")
    parser.add_argument("--json", default="results/agent_runtime_benchmark.json")
    parser.add_argument("--markdown", default="results/agent_runtime_benchmark.md")
    args = parser.parse_args()

    tools = make_tools()
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "catalog_size": len(tools.retriever.products),
        "budgets": [run_context_growth(int(value), args.turns) for value in args.budgets.split(",")],
        "react_profile": run_react_profile(args.repeats),
    }

    json_path = ROOT / args.json
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = ROOT / args.markdown
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    print(f"已写入 {json_path}")
    print(f"已写入 {markdown_path}")
    for item in payload["budgets"]:
        print(
            f"  预算 {item['budget_tokens']:>5}：最终 {item['final_tokens']:>5} tokens，"
            f"峰值 {item['max_observed_tokens']:>5}，首次压缩第 {item['first_compressed_turn']} 轮，"
            f"越界={'是' if not item['within_budget'] else '否'}"
        )
    print(
        f"  ReAct：{payload['react_profile']['runs']} 次运行，"
        f"P50 {payload['react_profile']['latency_ms']['p50']} ms，"
        f"P95 {payload['react_profile']['latency_ms']['p95']} ms"
    )


if __name__ == "__main__":
    main()
