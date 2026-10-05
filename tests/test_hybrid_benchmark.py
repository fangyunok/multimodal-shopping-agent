"""``TableRerankRetriever`` 的行为约束。

这个类只活在 ``scripts/benchmark_hybrid_retrieval.py`` 里，但它是 A3 结论的唯一来源，
所以它的两条关键性质必须有测试守着：

1. 候选级入口返回的是**重排后**的顺序，不是粗排顺序；
2. 粗排阶段不按 top_k 截断。

第1 条踩过一次：当时把 ``search_candidates`` 写成透传粗排候选，而评测脚本按
"有候选级入口就自己切片 top_k"分支，于是重排被静默绕过——指标看着正常（0.770），
实际跑的是粗排（真值 0.918）。这类 bug 不会报错，只会给出一个更差但合理的数字。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from shopping_agent.models import Product, SearchHit, SearchRequest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_hybrid_retrieval.py"


def _load_script():
    """按文件路径加载压测脚本。

    脚本不是包的一部分（``scripts/`` 不在 ``src/shopping_agent`` 里），
    但它的 retriever 类需要被测，所以用 importlib 按路径加载而不是复制一份实现——
    复制出来的实现只会测到自己那份副本，脚本本身坏了也测不出来。
    """
    spec = importlib.util.spec_from_file_location("benchmark_hybrid_retrieval", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_script()


class StubReranker:
    """按商品 id 的字典序反向打分——确定性且与粗排顺序无关。"""

    name = "stub"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.encoder_position = 0
        self.last_stats: dict[str, object] = {}

    def score(self, query: str, candidates: list[Product]) -> np.ndarray:
        self.last_stats = {"model_inference_ms": 1.0}
        return np.array([self.scores.get(product.id, 0.0) for product in candidates], dtype=np.float32)


class StubBase:
    """粗排桩：始终按传入 id 顺序返回，可记录实际被索取的候选数。"""

    def __init__(self, ids: list[str], products: list[Product]) -> None:
        self.ids = ids
        self.products = products
        self.catalog_path = None
        self.requested_counts: list[int] = []
        self.last_stats: dict[str, object] = {}

    def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
        self.requested_counts.append(int(count))
        return [
            SearchHit(product=product, score=1.0, text_score=1.0, image_score=0, reasons=["粗排"])
            for product in self.products[:count]
        ]

    def search(self, request: SearchRequest) -> list[SearchHit]:
        return self.search_candidates(request, request.top_k)


def make_products(count: int) -> list[Product]:
    return [
        Product(id=f"p{index:02d}", title=f"商品{index}", category="CAT", description="", price=10.0 + index)
        for index in range(count)
    ]


def test_search_candidates_returns_reranked_order_not_coarse_order() -> None:
    """候选级入口必须给出重排后的顺序。

    反向打分让 p09 排第一，而粗排原序是 p00 起——两条路径若不一致就说明透传了粗排。
    """
    products = make_products(10)
    base = StubBase([product.id for product in products], products)
    # 递增打分：粗排原序是 p00 起、p09 末，而分数让 p09 最高——顺序恰好完全倒过来。
    # 这样"透传粗排"和"重排生效"会给出两个不同的首条，断言才有分辨力。
    reranker = StubReranker({product.id: float(index) for index, product in enumerate(products)})
    retriever = bench.TableRerankRetriever(base, reranker, candidates=10)

    request = SearchRequest(query="q", top_k=10)
    from_candidates = [hit.product.id for hit in retriever.search_candidates(request, 10)]
    from_search = [hit.product.id for hit in retriever.search(request)]

    assert from_candidates == from_search
    assert from_candidates[0] == "p09", "候选级入口返回了粗排顺序，重排被绕过"


def test_coarse_stage_is_not_truncated_to_top_k() -> None:
    """粗排必须按 ``candidates`` 放宽，不能按 ``request.top_k`` 截断。

    这是两阶段检索最容易做错的地方：截到 top_k 后重排只能在 k 条里换顺序。
    """
    products = make_products(50)
    base = StubBase([product.id for product in products], products)
    reranker = StubReranker({product.id: float(index) for index, product in enumerate(products)})
    retriever = bench.TableRerankRetriever(base, reranker, candidates=50)

    hits = retriever.search(SearchRequest(query="q", top_k=10))

    assert base.requested_counts == [50], f"粗排只索取了 {base.requested_counts} 条候选"
    assert len(hits) == 10, "返回条数应等于 top_k"
    # 递增打分 → p49 最高，验证重排真的重排了
    assert hits[0].product.id == "p49"


def test_empty_candidate_set_short_circuits() -> None:
    """粗排为空时不打分，直接返回——避免重排器拿到空列表后产生无意义的分派。"""
    base = StubBase([], [])
    reranker = StubReranker({})
    retriever = bench.TableRerankRetriever(base, reranker, candidates=100)

    assert retriever.search(SearchRequest(query="q", top_k=10)) == []
    assert retriever.last_stats["rerank"] == "off"


def test_score_size_mismatch_falls_back_to_coarse_order() -> None:
    """打分条数与候选数不符时退回粗排，并把原因写进 ``last_stats``。

    静默错位是比报错更糟的失败：分数与候选错位会让指标低得莫名其妙却看不出原因。
    """

    class WrongSizeReranker(StubReranker):
        def score(self, query: str, candidates: list[Product]) -> np.ndarray:
            return np.zeros(len(candidates) + 1, dtype=np.float32)

    products = make_products(5)
    base = StubBase([product.id for product in products], products)
    retriever = bench.TableRerankRetriever(base, WrongSizeReranker({}), candidates=5)

    hits = retriever.search(SearchRequest(query="q", top_k=3))

    assert retriever.last_stats["rerank"] == "skipped_size_mismatch"
    assert [hit.product.id for hit in hits] == ["p00", "p01", "p02"]


def test_sweep_candidate_window_reports_monotonic_ceiling() -> None:
    """窗口扫描：候选召回随窗口放宽单调不降，hit@k 不会超过它。

    这两个性质是"天花板"这个概念成立的前提——若候选召回会随窗口下降，
    那它就不是上限，只是某个窗口下的巧合数字。
    """
    products = make_products(40)
    rows = [{"truth_id": f"p{index:02d}"} for index in range(40)]

    class QueryAwareBase(StubBase):
        def search_candidates(self, request: SearchRequest, count: int) -> list[SearchHit]:
            self.requested_counts.append(int(count))
            return [
                SearchHit(product=product, score=1.0, text_score=1.0, image_score=0, reasons=["粗排"])
                for product in self.products[:count]
            ]

    swept = bench.sweep_candidate_window(
        QueryAwareBase([p.id for p in products], products),
        bench.FixedQueryEncoder(np.zeros((40, 4), dtype=np.float32), products),
        [{**row, "query": f"q{index}"} for index, row in enumerate(rows)],
        [5, 10, 20, 40],
        top_k=10,
    )

    assert [row["window"] for row in swept] == [5, 10, 20, 40]
    # 键名里嵌 f-string 需要 Python 3.12+（PEP 701），CI 跑 3.11，所以先算出来再用。
    ceilings = [row[f"candidate_recall@{row['window']}"] for row in swept]
    assert ceilings == sorted(ceilings), f"候选召回非单调：{ceilings}"
    for row in swept:
        assert row["hit@10"] <= row[f"candidate_recall@{row['window']}"]


def test_sweep_returns_empty_without_candidate_entrypoint() -> None:
    """没有候选级入口的后端不参与窗口扫描——返回空而不是抛错。"""

    class NoCandidateBase:
        products: list[Product] = []

        def search(self, request: SearchRequest) -> list[SearchHit]:  # pragma: no cover - 不该被调用
            raise AssertionError("无候选级入口时不应走 search")

    assert bench.sweep_candidate_window(NoCandidateBase(), None, [], [10], top_k=10) == []
