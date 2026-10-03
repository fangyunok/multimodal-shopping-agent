"""MCP Server 压测：产出可直接引用的 P50/P95/P99 与吞吐数据。

两种传输模式，口径不同，报告里必须分开引用：

- ``--transport inprocess``：内存内回环（``Client(server)``）。测的是
  「JSON-RPC 编解码 + 参数校验 + 并发闸门 + 业务处理」在单进程内的成本上限，
  **不代表网络服务容量**。
- ``--transport stdio``：启动真实服务端子进程，通过 stdio 传输调用。这是
  Claude Desktop / Cursor 等客户端实际使用的链路，包含进程间管道与流式编解码。

用法：

    .venv/Scripts/python scripts/benchmark_mcp.py --transport stdio --requests 300
    .venv/Scripts/python scripts/benchmark_mcp.py --concurrency 1,2,4,8 --json results/mcp_benchmark_inprocess.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp import Client, StdioServerParameters  # noqa: E402

from shopping_agent.mcp_server import (  # noqa: E402
    ServerConfig,
    build_server,
    load_tools,
    percentile,
)
from shopping_agent.planner import RulePlanner  # noqa: E402

QUERY_POOL = (
    "轻量透气通勤运动鞋",
    "防滑耐磨越野跑鞋",
    "大容量通勤双肩包",
    "夏季速干短袖",
    "无线降噪耳机",
)


def parse_concurrency(value: str) -> list[int]:
    levels = [int(item) for item in value.replace(" ", "").split(",") if item]
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("--concurrency 需要逗号分隔的正整数，例如 1,4,8")
    return levels


async def run_level(client, tool: str, total: int, concurrency: int, warmup: int, distinct: bool) -> dict:
    """在给定并发度下跑完 total 次调用，返回延迟与吞吐统计。"""
    arguments = [
        {"query": f"{QUERY_POOL[index % len(QUERY_POOL)]} #{index if distinct else 0}", "top_k": 5}
        for index in range(total)
    ]
    latencies: list[float] = []
    errors = 0

    for index in range(warmup):
        await client.call_tool(tool, arguments[index % total])

    gate = asyncio.Semaphore(concurrency)

    async def one(payload: dict) -> None:
        nonlocal errors
        async with gate:
            started = time.perf_counter()
            try:
                result = await client.call_tool(tool, payload)
                if result.is_error:
                    errors += 1
            except Exception:
                errors += 1
            latencies.append((time.perf_counter() - started) * 1000)

    wall_started = time.perf_counter()
    await asyncio.gather(*(one(payload) for payload in arguments))
    wall = time.perf_counter() - wall_started

    return {
        "concurrency": concurrency,
        "requests": total,
        "errors": errors,
        "wall_seconds": round(wall, 4),
        "throughput_rps": round(total / wall, 1) if wall else None,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "p99": percentile(latencies, 99),
            "mean": round(statistics.fmean(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
    }


def best_of(runs: list[dict]) -> dict:
    """取吞吐最高的一轮。微基准单轮易受调度噪声影响，best-of-N 比取均值更稳定。"""
    return max(runs, key=lambda row: row["throughput_rps"] or 0.0)


def stdio_parameters(args) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "shopping_agent.mcp_server",
            "--transport",
            "stdio",
            "--catalog",
            str(Path(args.catalog).resolve()),
            "--retriever-backend",
            args.retriever_backend,
            "--max-concurrency",
            str(args.max_concurrency),
            "--tool-timeout",
            str(args.tool_timeout),
            "--log-level",
            "WARNING",
        ],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )


async def main_async(args) -> dict:
    tools = load_tools(args.catalog, args.retriever_backend)
    planner = RulePlanner([product.category for product in tools.retriever.products])
    config = ServerConfig(
        timeout_seconds=args.tool_timeout,
        max_retries=args.max_retries,
        max_concurrency=args.max_concurrency,
    )
    server = build_server(tools, planner, config)

    if args.transport == "stdio":
        client_context = Client(stdio_parameters(args), read_timeout_seconds=args.tool_timeout * 4)
    else:
        client_context = Client(server)

    async def sweep(client, distinct: bool) -> list[dict]:
        rows = []
        for level in args.concurrency:
            runs = [
                await run_level(client, args.tool, args.requests, level, args.warmup, distinct)
                for _ in range(args.repeat)
            ]
            chosen = best_of(runs)
            chosen["runs"] = len(runs)
            rows.append(chosen)
        return rows

    async with client_context as client:
        report: dict = {
            "tool": args.tool,
            "transport": args.transport,
            "catalog": args.catalog,
            "catalog_size": len(tools.retriever.products),
            "python": sys.version.split()[0],
            "repeat": args.repeat,
            "config": config.model_dump(),
            "distinct_arguments": await sweep(client, distinct=True),
            "repeated_arguments": None if args.skip_replay else await sweep(client, distinct=False),
            "server_metrics": json.loads((await client.call_tool("get_runtime_metrics", {})).content[0].text),
        }
    return report


def render_markdown(report: dict) -> str:
    transport_label = {
        "inprocess": "进程内回环（内存内传输）",
        "stdio": "stdio 子进程（MCP 客户端实际链路）",
    }.get(report["transport"], report["transport"])
    lines = [
        f"### MCP 工具 `{report['tool']}` 压测 · {transport_label}",
        "",
        f"- 商品目录：`{report['catalog']}`（{report['catalog_size']} 条）｜Python {report['python']}",
        f"- 并发闸门 `max_concurrency = {report['config']['max_concurrency']}`，工具超时 `{report['config']['timeout_seconds']}s`",
        f"- 每档重复 {report.get('repeat', 1)} 轮，取吞吐最高的一轮（best-of-N，用于抑制调度噪声）",
        "- 计时口径：客户端发起 → 编解码 → 参数校验 → 业务处理 → 响应返回",
        "",
        "**场景 A：入参互不相同**（真实检索负载，幂等缓存不生效）",
        "",
        "| 并发度 | 请求数 | 吞吐 (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | 错误 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["distinct_arguments"]:
        latency = row["latency_ms"]
        lines.append(
            f"| {row['concurrency']} | {row['requests']} | {row['throughput_rps']} | "
            f"{latency['p50']} | {latency['p95']} | {latency['p99']} | {row['errors']} |"
        )
    replay = report.get("repeated_arguments")
    if replay:
        lines += [
            "",
            "**场景 B：入参完全相同**（客户端重试 / 重复调用的幂等重放）",
            "",
            "| 并发度 | 请求数 | 吞吐 (req/s) | P50 (ms) | P95 (ms) | P99 (ms) | 错误 |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in replay:
            latency = row["latency_ms"]
            lines.append(
                f"| {row['concurrency']} | {row['requests']} | {row['throughput_rps']} | "
                f"{latency['p50']} | {latency['p95']} | {latency['p99']} | {row['errors']} |"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the MCP server")
    parser.add_argument("--transport", choices=("inprocess", "stdio"), default="inprocess")
    parser.add_argument("--catalog", default="data/products.jsonl")
    parser.add_argument("--tool", default="search_products")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3, help="每个并发度重复轮数，取吞吐最高的一轮")
    parser.add_argument("--concurrency", type=parse_concurrency, default=[1, 2, 4, 8])
    parser.add_argument("--retriever-backend", choices=("baseline", "clip", "fusion"), default="baseline")
    parser.add_argument("--tool-timeout", type=float, default=2.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--json", metavar="PATH", help="把完整报告写入文件")
    parser.add_argument("--markdown", metavar="PATH", help="把 Markdown 表格写入文件")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(main_async(args))
    markdown = render_markdown(report)
    print(markdown)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 JSON：{args.json}")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(markdown + "\n", encoding="utf-8")
        print(f"已写入 Markdown：{args.markdown}")


if __name__ == "__main__":
    main()
