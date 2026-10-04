"""A1「真实查询评测集」的测试：生成脚本纯逻辑 + 压测脚本的 hit@k 装配语义。

不依赖 Ollama / Chinese-CLIP / faiss——生成脚本的 LLM 调用与编码都有独立入口，
这里只测纯逻辑：prompt 构造、JSON 解析校验、product_id → 语料行号映射、
以及「单元素真值 ⇒ recall@k = hit@k」这条数学等价。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


query_set = load_script("build_query_set")
benchmark = load_script("benchmark_ann_scaling")

from shopping_agent.indexing import normalize  # noqa: E402
from shopping_agent.models import Product  # noqa: E402
from shopping_agent.synthetic import SyntheticCorpus  # noqa: E402


# ---------- build_query_set：prompt 与解析 ----------

def test_parse_queries_accepts_full_coverage() -> None:
    data = {"queries": [{"index": 0, "query": "红色连衣裙"}, {"index": 1, "query": " 蓝色乐福鞋 "}]}
    assert query_set.parse_queries(data, 2) == {0: "红色连衣裙", 1: "蓝色乐福鞋"}


def test_parse_queries_rejects_partial_coverage() -> None:
    with pytest.raises(ValueError, match="不全"):
        query_set.parse_queries({"queries": [{"index": 0, "query": "x"}]}, 2)


def test_parse_queries_rejects_missing_array() -> None:
    with pytest.raises(ValueError, match="queries"):
        query_set.parse_queries({"result": []}, 1)


def test_build_prompt_contains_product_facts() -> None:
    product = Product(id="p1", title="Dames Leder Hakken Blauw", category="SHOES", description="", price=0)
    prompt = query_set.build_prompt([product], query_set.PERSONAS[0])
    assert "Dames Leder Hakken Blauw" in prompt  # 商品事实进 prompt
    assert "SHOES" in prompt
    assert "只输出 JSON" in prompt  # 输出约束在
    assert "5~25 个汉字" in prompt  # 长度约束在
    # 价格为 0 时不带价格段（避免 LLM 把缺价商品编出预算）
    assert "价格：" not in prompt


# ---------- 错位检测：数字/英文锚点交叉检查 ----------

def test_looks_mismatched_flags_foreign_anchors() -> None:
    """查询里出现不属于该商品的数字或英文锚点 → 判错位。"""
    glass = Product(id="g", title="16-Piece Old Fashioned Glass Set", category="DRINK", description="", price=1)
    assert query_set.looks_mismatched(glass, "手机壳 小米红米 y2")  # 完全说别的商品
    assert query_set.looks_mismatched(glass, "16件 玻璃 手机壳 redmi")  # 数字对但英文锚点不对
    assert not query_set.looks_mismatched(glass, "复古玻璃 饮品套装 16件")  # 锚点都在
    assert not query_set.looks_mismatched(glass, "玻璃酒杯")  # 无锚点 → 无法判错，放行


# ---------- parse_queries：鲁棒性 ----------


def test_parse_queries_skips_non_dict_items() -> None:
    """小模型偶发把 queries 写成 [0, 1, 2] 纯数字数组——跳过非 dict 项（不报 AttributeError），
    覆盖不全时按设计抛 ValueError，由 generate_batch 的补齐逻辑重试。"""
    with pytest.raises(ValueError):
        query_set.parse_queries({"queries": [0, 1, {"index": 2, "query": "红色连衣裙"}]}, 3)
    # 覆盖完整时正常抽取
    assert query_set.parse_queries(
        {"queries": [0, {"index": 1, "query": "a"}, {"index": 0, "query": "b"}]}, 2
    ) == {0: "b", 1: "a"}


def test_parse_queries_rejects_incomplete_coverage() -> None:
    with pytest.raises(ValueError):
        query_set.parse_queries({"queries": [{"index": 0, "query": "a"}]}, 3)


# ---------- benchmark：apply_external_queries 装配 ----------

def make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "queries_file": None,
        "queries_catalog": None,
        "vectors_catalog": None,
        "queries": 7,
        "top_k": 10,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def write_catalog(path: Path, products: list[Product]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for product in products:
            stream.write(json.dumps(product.model_dump(), ensure_ascii=False) + "\n")


def write_queries(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_apply_external_queries_maps_and_skips(tmp_path: Path) -> None:
    dimension = 4
    products = [
        Product(id=f"p{i}", title=f"item {i}", category="X", description="", price=1) for i in range(5)
    ]
    catalog_path = tmp_path / "products.jsonl"
    write_catalog(catalog_path, products)
    # 语料 = 前 4 条（count=4），p4 在语料外 → 跳过
    corpus = SyntheticCorpus(
        vectors=normalize(np.random.default_rng(0).normal(size=(4, dimension)).astype(np.float32)),
        queries=np.zeros((0, dimension), dtype=np.float32),
        ground_truth=np.zeros((0, 10), dtype=np.int64),
    )
    rows = [
        {"product_id": "p0", "query": "q0"},
        {"product_id": "p3", "query": "q3"},
        {"product_id": "p4", "query": "out of corpus"},
        {"product_id": "p1", "query": "q1"},
    ]
    queries_path = tmp_path / "queries.npy"
    np.save(queries_path, normalize(np.random.default_rng(1).normal(size=(len(rows), dimension)).astype(np.float32)))
    queries_jsonl = tmp_path / "queries.jsonl"
    write_queries(queries_jsonl, rows)

    args = make_args(queries_file=str(queries_path), queries_catalog=str(queries_jsonl))
    summary = benchmark.apply_external_queries(corpus, catalog_path, args, count=4)

    assert summary["queries"] == 3
    assert summary["queries_skipped_unmapped"] == 1
    assert args.queries == 3
    assert corpus.queries.shape == (3, dimension)
    # 真值是单元素集合：每行只有一个语料行号
    assert corpus.ground_truth.tolist() == [[0], [3], [1]]


def test_apply_external_queries_rejects_misaligned(tmp_path: Path) -> None:
    dimension = 4
    products = [Product(id="p0", title="t", category="X", description="", price=1)]
    catalog_path = tmp_path / "products.jsonl"
    write_catalog(catalog_path, products)
    corpus = SyntheticCorpus(
        vectors=normalize(np.random.default_rng(0).normal(size=(1, dimension)).astype(np.float32)),
        queries=np.zeros((0, dimension), dtype=np.float32),
        ground_truth=np.zeros((0, 10), dtype=np.int64),
    )
    queries_path = tmp_path / "queries.npy"
    np.save(queries_path, np.ones((2, dimension), dtype=np.float32))  # 2 条向量
    queries_jsonl = tmp_path / "queries.jsonl"
    write_queries(queries_jsonl, [{"product_id": "p0", "query": "q"}])  # 1 行 jsonl

    args = make_args(queries_file=str(queries_path), queries_catalog=str(queries_jsonl))
    with pytest.raises(ValueError, match="不对齐"):
        benchmark.apply_external_queries(corpus, catalog_path, args, count=1)


def test_single_element_truth_makes_recall_equal_hit_rate() -> None:
    """hit@k 的数学等价：真值为单元素集合时，recall@k 就是命中率。"""
    corpus = SyntheticCorpus(
        vectors=normalize(np.random.default_rng(0).normal(size=(10, 4)).astype(np.float32)),
        queries=np.zeros((3, 4), dtype=np.float32),
        ground_truth=np.asarray([[2], [5], [9]], dtype=np.int64),
    )
    # 查询 0：真值 2 在结果首位（命中）；查询 1：真值 5 不在结果里（未命中）；查询 2：真值 9 在首位（命中）
    indices = np.asarray(
        [
            [2, 0, 1, 3, 4, 6, 7, 8, 9, 5],
            [0, 1, 2, 3, 4, 6, 7, 8, 9, 3],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        ],
        dtype=np.int64,
    )
    recall = corpus.recall(indices, top_k=10)
    assert recall == pytest.approx(2 / 3)  # 3 条查询命中 2 条
