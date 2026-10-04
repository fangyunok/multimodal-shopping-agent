"""用本地 LLM 从商品目录生成「真实分布」的自然语言查询评测集。

为什么要这个脚本
----------------
``benchmark_ann_scaling.py --vectors`` 的查询是从商品向量末尾切出来的——查询与语料
**同分布**。这回答不了"换成真人打字的查询，索引还是这个表现吗"。本脚本从商品侧
生成中文购物查询（模拟用户在搜索框输入的内容），再用与线上**查询端完全一致**的
编码口径（``encode_texts`` 纯文本向量——用户输入的是文字，不带图）产出查询向量，
供压测脚本以 ``--queries-file`` 消费。

评测语义（与压测脚本的对接）
--------------------------
每条查询记录了它由哪个商品生成（``product_id``）。压测时该商品**保留在语料里**：
用户描述想买的东西、检索把它排进 top-k，本来就是正确行为，要测的正是这个跨表达
匹配能力。压测脚本会把真值装配成"单元素集合（原商品行号）"，此时 recall@k 的
数学语义自动退化为 hit@k（原商品出现在 top-k 的比例）——无需新指标。

生成质量的两道防线
----------------
1. 只依据商品自身属性生成，不引入商品没有的特征（prompt 约束 + 生成后人工抽查）
2. persona 多样化：不同查询风格轮换，避免评测集句式单一

用法
----
    python scripts/build_query_set.py \
        --catalog data/processed/products.jsonl \
        --pool 90000 --count 1000 \
        --ollama-model qwen2.5:7b --device cpu \
        --output-dir outputs/real_clip

    # 再压测（原商品必须在语料前缀内，所以 --pool ≥ 最大 scale）
    python scripts/benchmark_ann_scaling.py \
        --vectors outputs/real_clip/vectors.npy \
        --vectors-catalog data/processed/products.jsonl \
        --queries-file outputs/real_clip/queries.npy \
        --queries-catalog outputs/real_clip/queries.jsonl \
        --scales 10000,90000 --top-k 10
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.encoders import ChineseClipEncoder  # noqa: E402
from shopping_agent.indexing import load_catalog, normalize  # noqa: E402

OLLAMA_URL = "http://localhost:11434/api/chat"

# persona：轮换不同的用户查询风格，避免评测集句式单一
PERSONAS = [
    {"id": "keywords", "style": "关键词堆叠：像电商搜索框直接输入，名词+属性词，没有完整句子"},
    {"id": "colloquial", "style": "口语描述：像跟朋友描述想要的东西，自然口语"},
    {"id": "need", "style": "需求导向：先说用途/场景，再落到品类"},
    {"id": "budget", "style": "预算导向：价格是硬约束，预算信息必须出现在查询里（仅当商品有价格时）"},
]

PROMPT_TEMPLATE = """你是购物查询生成器。下面是 {count} 个商品的原始信息（标题多为英文/荷兰语等多语言）。
为每个商品生成一条**中文**购物搜索查询，模拟真实用户在电商搜索框里输入的内容。

本次用户风格：{persona_style}

规则：
1. 只依据商品自身属性（标题/类目/价格），不得编造商品没有的特征
2. 5~25 个汉字，符合上述用户风格
3. 价格为 0 或缺失时，查询里不要出现价格
4. 用中文表达，商品标题里的外语品牌名可保留原文
5. 查询里尽量带上该商品的型号数字或品牌词（如果有），方便对齐商品

商品：
{items}

示例（风格：关键词堆叠）：
商品：Amazon-merk - vinden. Dames Leder Gesloten Teen Hakken,Veelkleurig Vrouw Blauw（类目 SHOES）
查询：蓝色 女士 乐福鞋 皮面

只输出 JSON：{{"queries": [{{"index": 0, "query": "..."}}, {{"index": 1, "query": "..."}}]}}，必须覆盖全部 {count} 个商品。"""


def build_prompt(products: list, persona: dict) -> str:
    lines = []
    for i, product in enumerate(products):
        price = f"｜价格：{product.price:.0f}" if getattr(product, "price", 0) else ""
        lines.append(f"[{i}] 标题：{product.title}｜类目：{getattr(product, 'category', '')}{price}")
    return PROMPT_TEMPLATE.format(
        count=len(products), persona_style=persona["style"], items="\n".join(lines)
    )


# 错位检测：数字与英文品牌名是跨语言稳定的锚点。批量生成时 LLM 容易在
# 同类商品扎堆时看错行（静默错位：index 齐全但内容串位），单靠 index 覆盖
# 校验发现不了——用锚点交叉检查把错位的条目标出来单独补生成。
_ANCHOR_WORD = re.compile(r"[A-Za-z]{3,}")
_ANCHOR_DIGIT = re.compile(r"\d+(?:\.\d+)?")
# 泛词：出现多个商品里是常态，不算错位证据
_GENERIC_WORDS = {
    "amazon", "brand", "the", "for", "with", "and", "from", "by", "in", "of", "on",
    "piece", "pieces", "pcs", "pack", "set", "mix", "men", "women", "girls", "boys",
}


def _anchor_tokens(text: str) -> tuple[set[str], set[str]]:
    words = {word.lower() for word in _ANCHOR_WORD.findall(text)}
    digits = set(_ANCHOR_DIGIT.findall(text))
    return words, digits


def looks_mismatched(product, query: str) -> bool:
    """查询里出现不属于该商品的数字或英文锚点 → 疑似错位。"""
    query_words, query_digits = _anchor_tokens(query)
    product_words, product_digits = _anchor_tokens(product.title + " " + getattr(product, "description", ""))
    foreign_digits = query_digits - product_digits
    foreign_words = (query_words - product_words) - _GENERIC_WORDS
    return bool(foreign_digits or foreign_words)


def call_ollama(model: str, prompt: str, timeout: int = 180) -> dict:
    """调用本地 Ollama chat 接口，强制 JSON 输出。返回解析后的 dict。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.7},
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return json.loads(body["message"]["content"])


def parse_queries(data: dict, expected: int) -> dict[int, str]:
    """校验返回结构并抽取 {index: query}；index 不全时抛异常触发重试。"""
    queries = data.get("queries")
    if not isinstance(queries, list):
        raise ValueError(f"返回缺少 queries 数组：{str(data)[:200]}")
    result: dict[int, str] = {}
    for item in queries:
        index, query = item.get("index"), item.get("query", "")
        if isinstance(index, int) and 0 <= index < expected and isinstance(query, str) and query.strip():
            result[index] = query.strip()
    if len(result) < expected:
        raise ValueError(f"index 覆盖不全：{len(result)}/{expected}")
    return result


def generate_batch(model: str, products: list, persona: dict, retries: int = 3) -> list[str]:
    """一批商品的查询生成。LLM 偶发漏 index 或静默错位——都进入 pending 单独补一轮。"""
    queries: dict[int, str] = {}
    pending = list(range(len(products)))
    last_error: Exception | None = None
    for _ in range(retries + 1):
        if not pending:
            break
        subset = [products[i] for i in pending]
        prompt = build_prompt(subset, persona)
        try:
            got = parse_queries(call_ollama(model, prompt), len(subset))
        except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(2)
            continue
        for local_index, query in got.items():
            full = pending[local_index]
            if looks_mismatched(subset[local_index], query):  # 静默错位：不采用，留待补生成
                continue
            queries[full] = query
        pending = [i for i in pending if i not in queries]
        if pending:
            time.sleep(1)
    if pending:
        raise RuntimeError(f"以下商品重试 {retries} 轮后仍未生成：{last_error}")
    return [queries[i] for i in range(len(products))]


def main() -> None:
    global OLLAMA_URL
    parser = argparse.ArgumentParser(description="生成真实分布的自然语言查询评测集")
    parser.add_argument("--catalog", required=True, help="商品目录 jsonl（与向量文件逐行对齐）")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pool", type=int, default=90000, help="从前 N 条商品里抽样（须 ≥ 压测最大 scale，保证原商品在语料内）")
    parser.add_argument("--count", type=int, default=1000, help="生成多少条查询")
    parser.add_argument("--batch", type=int, default=10, help="每次喂给 LLM 的商品数")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--device", default="cpu", help="查询编码设备（cpu 即够，量小）")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--limit", type=int, default=0, help="只生成前 N 条（冒烟用）；0 表示全部")
    args = parser.parse_args()

    OLLAMA_URL = args.ollama_url

    catalog = Path(args.catalog).resolve()
    products = load_catalog(catalog)
    pool = products[: args.pool]
    pool = [p for p in pool if p.title and p.title.strip()]
    if not pool:
        raise SystemExit("抽样池为空")
    rng = random.Random(args.seed)
    sampled = rng.sample(pool, min(args.count, len(pool)))
    if args.limit:
        sampled = sampled[: args.limit]
    # 按类目轮转重排：同类商品相邻会让 LLM 看错行（静默错位），交错后错位率显著下降
    by_category: dict[str, list] = {}
    for product in sampled:
        by_category.setdefault(getattr(product, "category", "") or "X", []).append(product)
    width = min(len(group) for group in by_category.values())
    interleaved = [p for group in zip(*by_category.values()) for p in group]
    interleaved += [p for group in by_category.values() for p in group[width:]]
    sampled = interleaved
    print(f"抽样 {len(sampled)} 条（池：前 {args.pool} 条中 {len(pool)} 条有标题，按类目轮转防扎堆）")

    rows: list[dict] = []
    start = time.perf_counter()
    for offset in range(0, len(sampled), args.batch):
        batch = sampled[offset : offset + args.batch]
        persona = PERSONAS[(offset // args.batch) % len(PERSONAS)]
        queries = generate_batch(args.ollama_model, batch, persona)
        for product, query in zip(batch, queries):
            rows.append(
                {
                    "product_id": product.id,
                    "query": query,
                    "persona": persona["id"],
                    "source_title": product.title,
                }
            )
        done = min(offset + args.batch, len(sampled))
        print(f"  {done}/{len(sampled)} 条已生成（{time.perf_counter() - start:.0f}s）")

    print(f"查询生成完成：{len(rows)} 条，开始编码（device={args.device}，纯文本向量）")
    encoder = ChineseClipEncoder(device=args.device, batch_size=32)
    encode_start = time.perf_counter()
    vectors = normalize(encoder.encode_texts([row["query"] for row in rows]))
    encode_seconds = time.perf_counter() - encode_start

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "queries.npy", np.asarray(vectors, dtype=np.float32))
    with (output_dir / "queries.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queries": len(rows),
        "dimension": int(vectors.shape[1]),
        "pool": args.pool,
        "catalog": str(catalog),
        "llm_model": args.ollama_model,
        "clip_model": encoder.model.name_or_path,
        "encode_mode": "text-only（与线上查询端口径一致）",
        "encode_seconds": round(encode_seconds, 2),
        "seed": args.seed,
        "personas": [p["id"] for p in PERSONAS],
        "environment": {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
        },
    }
    (output_dir / "queries.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {output_dir / 'queries.npy'}（{len(rows)} × {vectors.shape[1]}，编码 {encode_seconds:.1f}s）")
    print(f"元数据 {output_dir / 'queries.meta.json'}")
    print("提示：压测时 --pool 需 ≥ --scales 最大档，否则部分查询的原商品不在语料内、会被跳过")


if __name__ == "__main__":
    main()