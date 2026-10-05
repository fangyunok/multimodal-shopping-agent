# 混合检索对照实验报告

生成时间：2026-10-05T04:35:13.715553+00:00

## 结论先行

| 策略 | hit@10 | hit@10（同 category） | MRR | nDCG@10 | P50 延迟 | 相对稠密基线 |
|---|---|---|---|---|---|---|
| `dense_only` | 0.6680 | 0.9620 | 0.4853 | 0.5295 | 0.78 ms | 基线 |
| `bm25_only` | 0.5980 | 0.8160 | 0.3208 | 0.3859 | 0.98 ms | -7.0pt |
| `rrf` | 0.7440 | 0.9540 | 0.5434 | 0.5915 | 3.31 ms | +7.6pt |
| `rrf_rerank` | 0.7440 | 0.9540 | 0.6725 | 0.6903 | 15.12 ms | +7.6pt |

语料：**真实向量**（corpus_with_truth.npy，512 维，5477 条），查询：**真实中文文本查询** 500 条，top_k=10。

两个 hit 口径必须一起读：严口径（真值 = 生成该查询的原商品）与 A1 的 0.668 直接可比；
宽口径把「top-k 里有同品类商品」也算命中，回答的是「用户想找的东西在不在结果里」。

## 延迟分布

| 策略 | P50 | P95 | 均值 |
|---|---|---|---|
| `dense_only` | 0.78 ms | 1.01 ms | 0.81 ms |
| `bm25_only` | 0.98 ms | 1.63 ms | 1.04 ms |
| `rrf` | 3.31 ms | 4.12 ms | 3.36 ms |
| `rrf_rerank` | 15.12 ms | 16.26 ms | 15.76 ms |

## 配置

```json
{
  "generated_at": "2026-10-05T04:35:13.715553+00:00",
  "corpus": 5477,
  "corpus_name": "corpus_with_truth.npy",
  "dimension": 512,
  "queries": 500,
  "queries_skipped_unmapped": 0,
  "queries_file": "queries.npy",
  "top_k": 10,
  "rrf_k": 60,
  "oversample": 5.0,
  "dense_weight": 1.0,
  "bm25_weight": 1.0,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "bm25_build_seconds": 0.99,
  "bm25_summary": {
    "count": 5477,
    "vocabulary_size": 32267,
    "avg_doc_length": 64.3,
    "k1": 1.5,
    "b": 0.75,
    "size_mb": 2.023
  },
  "reranker": "cross-encoder",
  "rerank_candidates": 100,
  "cross_encoder": {
    "model": "BAAI/bge-reranker-base",
    "device": "cpu",
    "batch_size": 16,
    "max_length": 128,
    "queries_scored": 500,
    "latency_mean_ms": 1733.54,
    "latency_p95_ms": 2795.86
  },
  "kinds": [
    "dense_only",
    "bm25_only",
    "rrf",
    "rrf_rerank"
  ],
  "python": "3.13.14",
  "platform": "Windows-11-10.0.26200-SP0"
}
```
