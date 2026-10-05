from __future__ import annotations

import numpy as np
import pytest

from shopping_agent.bm25 import BM25Index, BM25Retriever, analyze, bm25_idf
from shopping_agent.models import Product, SearchRequest


def make_products(rows: list[tuple[str, str, str, float]]) -> list[Product]:
    return [
        Product(id=pid, title=title, category=category, description="", price=price)
        for pid, title, category, price in rows
    ]


def test_analyze_keeps_latin_tokens_whole_and_emits_cjk_bigrams() -> None:
    terms = analyze("LED灯泡 5W-30 A19")
    # 型号与规格必须整体保留：拆成单字符会让这些最有区分度的词项消失。
    assert "5w-30" in terms
    assert "a19" in terms
    assert "led" in terms
    # 中文单字保召回、二元组保精度。
    assert "灯" in terms
    assert "灯泡" in terms


def test_analyze_emits_single_chars_for_lone_chinese_character() -> None:
    # 单字成词的情况（二元组要能退化成单字，不能产出越界词项）。
    assert analyze("灯") == ["灯"]


def test_idf_is_always_positive() -> None:
    # 出现在全部文档里的词项也必须拿到正 IDF，否则"普遍出现"会变成"被惩罚"。
    assert bm25_idf(100, 100) > 0
    assert bm25_idf(100, 1) > bm25_idf(100, 50)


def test_exact_model_term_outranks_partial_overlap() -> None:
    documents = [
        "5W-30 汽车机油 5升",
        "汽车机油 1夸脱",
        "厨房收纳盒",
    ]
    index = BM25Index.build(documents)
    scores, rows = index.search("5W-30 机油", 3)
    assert rows[0] == 0
    assert scores[0] > scores[1]


def test_empty_query_terms_yield_zero_scores() -> None:
    index = BM25Index.build(["灯泡", "机油"])
    scores, _ = index.search("", 2)
    assert scores.tolist() == [0.0, 0.0]


def test_empty_document_collection_is_rejected() -> None:
    with pytest.raises(ValueError, match="空文档"):
        BM25Index.build([])


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="k1"):
        BM25Index.build(["a"], k1=-1)
    with pytest.raises(ValueError, match="b"):
        BM25Index.build(["a"], b=1.5)


def test_save_and_load_roundtrip_preserves_scores(tmp_path) -> None:
    index = BM25Index.build(["5W-30 汽车机油 5升", "LED 灯泡 A19", "厨房收纳盒"], k1=1.2, b=0.6)
    index.save(tmp_path)
    loaded = BM25Index.load(tmp_path)
    assert loaded.k1 == 1.2
    assert loaded.b == 0.6
    assert loaded.count == index.count
    for query in ("5W-30 机油", "led 灯泡", "收纳"):
        before, rows_before = index.search(query, 3)
        after, rows_after = loaded.search(query, 3)
        assert rows_before.tolist() == rows_after.tolist()
        assert np.allclose(before, after)


def test_bm25_retriever_search_respects_top_k(tmp_path) -> None:
    products = [
        Product(id="a", title="5W-30 汽车机油 5升", category="AUTO", description="", price=10),
        Product(id="b", title="LED 灯泡 A19", category="LIGHT", description="", price=5),
        Product(id="c", title="厨房收纳盒", category="HOME", description="", price=20),
    ]
    retriever = BM25Retriever.from_products(products, tmp_path / "products.jsonl")
    hits = retriever.search(SearchRequest(query="机油", top_k=2))
    assert len(hits) == 2
    assert hits[0].product.id == "a"
    assert "BM25 词项匹配" in hits[0].reasons


def test_bm25_retriever_applies_filters(tmp_path) -> None:
    products = [
        Product(id="a", title="机油 5升", category="AUTO", description="", price=100),
        Product(id="b", title="机油 1夸脱", category="AUTO", description="", price=10),
    ]
    retriever = BM25Retriever.from_products(products, tmp_path / "products.jsonl")
    hits = retriever.search(SearchRequest(query="机油", max_price=20, top_k=5))
    assert [hit.product.id for hit in hits] == ["b"]


def test_bm25_retriever_without_query_falls_back_to_rating(tmp_path) -> None:
    products = [
        Product(id="a", title="a", category="x", description="", price=1, rating=3),
        Product(id="b", title="b", category="x", description="", price=1, rating=5),
    ]
    retriever = BM25Retriever.from_products(products, tmp_path / "products.jsonl")
    hits = retriever.search(SearchRequest(query="", top_k=2))
    assert [hit.product.id for hit in hits] == ["b", "a"]
    assert retriever.last_stats["mode"] == "rating_fallback"


def test_search_candidates_returns_more_than_top_k(tmp_path) -> None:
    """两阶段检索的前提：粗排必须能交出比 top_k 更宽的候选集。"""
    products = [Product(id=f"p{i}", title=f"机油 型号{i}", category="AUTO", description="", price=1) for i in range(30)]
    retriever = BM25Retriever.from_products(products, tmp_path / "products.jsonl")
    request = SearchRequest(query="机油", top_k=5)
    assert len(retriever.search(request)) == 5
    assert len(retriever.search_candidates(request, 25)) == 25


def test_product_count_mismatch_is_rejected(tmp_path) -> None:
    index = BM25Index.build(["a", "b"])
    products = [Product(id="p", title="p", category="x", description="", price=1)]
    with pytest.raises(ValueError, match="不一致"):
        BM25Retriever(products, tmp_path / "products.jsonl", index)


def test_summary_exposes_index_size(tmp_path) -> None:
    index = BM25Index.build(["机油 五升", "灯泡"])
    summary = index.summary()
    assert summary["count"] == 2
    # size_mb 保留 3 位小数，小索引会舍入到 0.0；字节数才是可断言的下界。
    assert index.index_bytes > 0
    assert summary["size_mb"] == round(index.index_bytes / (1024 * 1024), 3)
    assert summary["vocabulary_size"] == index.vocabulary_size
