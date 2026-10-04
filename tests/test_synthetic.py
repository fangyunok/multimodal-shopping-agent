"""合成语料的回归测试。

这一层看着像"测试工具"，但它其实是整个规模化论证的地基：如果语料本身没有簇结构、
或真值算错了，后面所有 ANN 召回率数字都是噪音。所以这里逐条钉死它的语义。
"""

from __future__ import annotations

import numpy as np
import pytest

from shopping_agent.ann_index import AnnIndex, AnnIndexConfig
from shopping_agent.synthetic import (
    PRODUCT_TOKEN_PATTERN,
    SyntheticEncoder,
    generate_corpus,
    query_for,
    write_catalog,
)


def test_same_seed_is_reproducible() -> None:
    first = generate_corpus(count=200, dimension=32, queries=10, clusters=8, seed=7)
    second = generate_corpus(count=200, dimension=32, queries=10, clusters=8, seed=7)
    assert np.array_equal(first.vectors, second.vectors)
    assert np.array_equal(first.ground_truth, second.ground_truth)


def test_different_seed_changes_corpus() -> None:
    first = generate_corpus(count=100, dimension=32, queries=8, clusters=4, seed=1)
    second = generate_corpus(count=100, dimension=32, queries=8, clusters=4, seed=2)
    assert not np.allclose(first.vectors, second.vectors)


def test_vectors_are_l2_normalized() -> None:
    corpus = generate_corpus(count=256, dimension=64, queries=16, clusters=16)
    norms = np.linalg.norm(corpus.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    query_norms = np.linalg.norm(corpus.queries, axis=1)
    assert np.allclose(query_norms, 1.0, atol=1e-5)


def test_zero_spread_collapses_the_cluster() -> None:
    """spread=0 时同簇向量完全重合——这是"簇内相对散度"语义的边界锚点。"""
    corpus = generate_corpus(count=64, dimension=32, queries=8, clusters=4, spread=0.0, seed=3)
    # 归一化后，同簇向量应当彼此相同：去重后恰好剩 clusters 种。
    unique = np.unique(np.round(corpus.vectors, 5), axis=0)
    assert unique.shape[0] == 4


def test_spread_controls_intra_cluster_spread() -> None:
    """spread 必须真正影响"检索难度"：调大后同簇相似度下降。"""
    tight = generate_corpus(count=128, dimension=64, queries=8, clusters=8, spread=0.1, seed=5)
    loose = generate_corpus(count=128, dimension=64, queries=8, clusters=8, spread=1.5, seed=5)
    tight_sim = float((tight.vectors @ tight.vectors.T)[~np.eye(tight.count, dtype=bool)].mean())
    loose_sim = float((loose.vectors @ loose.vectors.T)[~np.eye(loose.count, dtype=bool)].mean())
    assert tight_sim > loose_sim


def test_ground_truth_matches_exact_search() -> None:
    """真值必须能被精确检索复现——真值错了，recall 就是自欺。"""
    corpus = generate_corpus(count=300, dimension=48, queries=20, top_k=5, clusters=12, seed=11)
    exact = AnnIndex.build(corpus.vectors, AnnIndexConfig(kind="numpy", dimension=48))
    _, indices = exact.search(corpus.queries, 5)
    assert np.array_equal(corpus.ground_truth, indices)


def test_ground_truth_top1_is_the_nearest_neighbour() -> None:
    corpus = generate_corpus(count=200, dimension=32, queries=12, top_k=3, clusters=8, seed=13)
    nearest = corpus.vectors @ corpus.queries.T
    for column in range(corpus.queries.shape[0]):
        assert int(np.argmax(nearest[:, column])) == int(corpus.ground_truth[column, 0])


def test_recall_is_one_for_perfect_result() -> None:
    corpus = generate_corpus(count=120, dimension=32, queries=10, top_k=4, clusters=6, seed=17)
    assert corpus.recall(corpus.ground_truth, top_k=4) == pytest.approx(1.0)


def test_recall_drops_for_wrong_result() -> None:
    corpus = generate_corpus(count=120, dimension=32, queries=10, top_k=4, clusters=6, seed=19)
    wrong = np.full_like(corpus.ground_truth, -1)
    assert corpus.recall(wrong, top_k=4) == pytest.approx(0.0)


def test_recall_ignores_padding_negatives() -> None:
    """目录比 top_k 小时，真值里会用 -1 占位，这些不能被算成漏检。"""
    corpus = generate_corpus(count=3, dimension=16, queries=4, top_k=10, clusters=2, seed=23)
    assert corpus.ground_truth.shape[1] == 3
    assert corpus.recall(corpus.ground_truth, top_k=10) == pytest.approx(1.0)


def test_summarize_reports_shape() -> None:
    corpus = generate_corpus(count=140, dimension=32, queries=6, clusters=7, spread=0.4, seed=29)
    summary = corpus.summarize()
    assert summary["vectors"] == 140
    assert summary["dimension"] == 32
    assert summary["queries"] == 6
    assert summary["clusters"] == 7
    assert summary["spread"] == 0.4
    assert summary["seed"] == 29


def test_encoder_maps_token_back_to_its_row() -> None:
    """点名查询的核心契约：标题里的 token 必须精确还原成那一行向量。"""
    corpus = generate_corpus(count=500, dimension=32, queries=4, clusters=8, seed=31)
    encoder = SyntheticEncoder(corpus)
    for row in (0, 42, 499):
        encoded = encoder.encode_texts([f"找一款 syn{row:07d} 同款"])[0]
        assert np.allclose(encoded, corpus.vectors[row], atol=1e-5)


def test_encoder_hash_fallback_is_stable() -> None:
    """无法解析的文本走哈希兜底——必须跨调用稳定，否则同一 query 两次结果不同。"""
    corpus = generate_corpus(count=64, dimension=32, queries=4, clusters=4, seed=37)
    encoder = SyntheticEncoder(corpus)
    first = encoder.encode_texts(["随便一段没有 token 的查询"])
    second = encoder.encode_texts(["随便一段没有 token 的查询"])
    assert np.allclose(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)


def test_encoder_image_paths_share_the_token_contract() -> None:
    corpus = generate_corpus(count=80, dimension=32, queries=4, clusters=4, seed=41)
    encoder = SyntheticEncoder(corpus)
    encoded = encoder.encode_images(["data/images/syn0000017.jpg"])[0]
    assert np.allclose(encoded, corpus.vectors[17], atol=1e-5)


def test_encoder_empty_corpus_falls_back_to_dimension() -> None:
    """语料为空时没有向量可借，编码器必须退回到显式声明的维度而不是崩掉。"""
    empty = generate_corpus(count=1, dimension=16, queries=1, clusters=1, seed=43)
    empty.vectors = empty.vectors[:0]
    encoder = SyntheticEncoder(empty, fallback_dimension=16)
    assert encoder.dimension == 16
    assert encoder.encode_texts(["无 token"])[0].shape == (16,)


def test_query_for_embeds_the_product_id() -> None:
    text = query_for("syn0000042")
    assert "syn0000042" in text
    assert PRODUCT_TOKEN_PATTERN.search(text).group(1) == "0000042"


def test_products_carry_metadata_and_tokens() -> None:
    corpus = generate_corpus(count=30, dimension=16, queries=2, clusters=3, seed=47)
    assert len(corpus.products) == 30
    first = corpus.products[0]
    assert first.id == "syn0000000"
    assert first.id in first.title
    assert first.price > 0
    assert first.source == "synthetic"
    # 价格取对数正态，长尾：应有明显的量级跨度，而不是挤在一起。
    prices = np.array([product.price for product in corpus.products])
    assert prices.max() / prices.min() > 3


def test_write_catalog_round_trips(tmp_path) -> None:
    from shopping_agent.indexing import load_catalog

    corpus = generate_corpus(count=25, dimension=16, queries=2, clusters=3, seed=53)
    path = write_catalog(corpus.products, tmp_path / "catalog_synthetic.jsonl")
    loaded = load_catalog(path)
    assert len(loaded) == 25
    assert loaded[7].id == "syn0000007"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"count": 0}, "count"),
        ({"dimension": 0}, "dimension"),
        ({"clusters": 0}, "clusters"),
        ({"spread": -0.1}, "spread"),
    ],
)
def test_generate_corpus_rejects_bad_arguments(kwargs: dict, message: str) -> None:
    payload = {"count": 32, "dimension": 16, "queries": 2, "clusters": 4}
    payload.update(kwargs)
    with pytest.raises(ValueError, match=message):
        generate_corpus(**payload)
