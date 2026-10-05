# 混合检索与重排：把 33pt 缺口拆给两路证据

[QUERY_DISTRIBUTION.md](QUERY_DISTRIBUTION.md) 测出一个结论：真实文本查询 hit@10 = 0.668，
同分布切片查询 0.9982，**差 33 个百分点**；而五种 ANN 索引后端在这个场景下的命中差异只有 ±0.8pt。

那份文档的落点是"瓶颈在表达对齐，不在检索结构"。这个文档接着往下走一步：
**如果只有稠密向量一路证据，缺口能补多少？两路证据分别贡献了什么？剩下的部分归谁？**

## 结论先行

同一批语料（5,477 条真实 Chinese-CLIP 向量）、同一批查询（500 条真实中文文本查询）、
同一套真值，只换检索与排序策略：

| 策略 | hit@10 | hit@10（同 category） | MRR | nDCG@10 | 候选召回 | P50 延迟 |
|---|---|---|---|---|---|---|
| 稠密向量（基线） | 0.6680 | 0.9620 | 0.4853 | 0.5295 | — | 0.78 ms |
| BM25 倒排 | 0.5980 | 0.8160 | 0.3208 | 0.3859 | — | 0.98 ms |
| **RRF 融合两路** | **0.7440** | 0.9540 | 0.5434 | 0.5915 | 0.946 | 3.31 ms |
| RRF + cross-encoder 重排 | 见下方"重排那一档" | | | | | |

三件事可以直接读出来：

1. **单看 BM25 是负收益**（0.598 < 0.668）。它对"精确词项"强，但对"意思相近"完全无能为力——
   中文查询与荷兰语/德语商品标题之间还隔着跨语言鸿沟。这说明它**不是替代品，是补充**。
2. **两路融合后 hit@10 从 0.668 涨到 0.744（+7.6pt）**。BM25 单独更差却能把融合拉高，
   唯一合理的解释是两路的**错误模式不同**：稠密检索失手在"同品类近邻压过真值"，
   BM25 失手在"语义完全不同但词面撞车"，而 66.3% 的稠密失败样本恰好是前者（见 A1 的失败样本分解）。
3. **候选集里其实有 94.6% 的真值，但 top-10 只放出 77%。**
   这个 94.6% 就是重排的天花板——它也是判断"该继续优化召回还是该优化排序"的唯一依据。

## 两路证据各自擅长什么

BM25 的分词是本项目里一个需要解释的选择：中文用「单字 + 相邻二元组」，无词典依赖。

- **单字**保证召回：任何字都能命中，不需要词典就不会有"词典没收录导致整条召回失败"这种失败模式。
- **二元组**保证精度：「机油」比「机」+「油」更能定位，而中文二字词恰好是最小的语义单元。
- **拉丁串整体保留**：`5W-30`、`132x168`、`J7`、`A19` 这类规格型号是最强的区分度信号，
  拆成单字符等于把它们从索引里抹掉。ABO 语料里这类 token 大量存在（标题是荷兰语/德语/英语混排）。

实测分词效果：5,477 条商品建出 32,267 个词项，平均文档长度 64.3，倒排表 2.02 MB，构建 0.98 s。
对比稠密向量矩阵本身是 10.7 MB——**倒排表比向量小 5 倍**，这是"两路并存"在内存上可行的前提。

RRF 选的是名次而不是分数，原因是量纲：BM25 分数无上界（取决于语料与词项分布），
CLIP 余弦落在 [-1, 1]。线性加权（项目原有的 `ScoreFusionRetriever`）要求两者可比，
而 `lexical_weight` 会随语料、随查询长度漂移——换一批数据就要重调，
而调不好时排名的变化其实来自分数尺度而不是相关性。RRF 只有 `rrf_k` 一个常数（取 60）。

## 重排那一档：先看天花板，再看模型

粗排放宽到 100 条候选后：

| 候选窗口 | 真值在候选集里的比例 |
|---|---|
| top-10 | 0.770 |
| **top-100** | **0.946** |

也就是说：**有 17.6pt 的命中藏在候选集里没被排进 top-10。** 这 17.6pt 是重排能争取的全部空间，
也是它争取不到的部分——重排只能在候选集内换顺序，进不到候选集的商品永远出不来。

一个必须记下来的坑：**第一版实验里 cross-encoder 的 hit@10 与 RRF 完全持平（都 0.744），
但 MRR 从 0.5434 升到 0.6725。** 原因是候选收集用了 `search()` 而不是 `search_candidates()`，
而 `search()` 会按 `request.top_k`（=10）截断——于是重排器拿到的是**已经被截到 10 条的集合**，
只能在这 10 条里换顺序，永远换不出更好的东西。

这正是 [rerank.py](../src/shopping_agent/rerank.py) 里把 `search_candidates(request, count)`
定成检索器**必备契约**的原因：它不是"更宽的查询"这种可选优化，而是两阶段检索正确性的前提。
同样的错误在 `RerankRetriever` 的实现里被单测钉住（`test_rerank_requests_wider_candidate_set_than_top_k`）。

顺带一个反直觉结论：**`ExactVectorReranker` 会把 hit@10 打回 0.668。**
它的分数就是向量余弦，与稠密粗排同源，所以重排等于"把 BM25 的贡献丢掉、只按向量重新排序"。
它跑通的是链路，不是增益——这一档保留在脚本里（`--reranker exact-vector`），
作用是提供一个零依赖的降级路径，以及把"重排有没有真的在跑"这件事变成可观测。

## 端到端延迟

| 策略 | P50 | P95 | 说明 |
|---|---|---|---|
| 稠密向量 | 0.78 ms | 0.90 ms | 单次矩阵乘，5,477 × 512 |
| BM25 | 0.98 ms | 1.57 ms | postings 稀疏累加，随查询词分布度增长而非语料条数 |
| RRF | 3.31 ms | 4.23 ms | 两路各取候选 + 名次合并；多出的 2.4 ms 主要在 BM25 的 Python 侧对象组装 |
| cross-encoder 重排（CPU） | 1,733 ms/查询 | 2,796 ms | 100 对 (query, doc) 联合前向，CPU fp32 |

cross-encoder 在 CPU 上是**秒级**的，这个数字必须和它的收益一起读：
它把 P50 从 3.31 ms 抬到秒级，换来的是把候选集里那 17.6pt 往 top-10 里搬。
生产上正确的做法是把它放在算力充足的排序服务后面、只对 top-100 做这一遍，
并按 A/B 实测决定是否值得——本项目的定位是把这笔账算清楚，不是默认它划算。

## 复现

```bash
# 稠密 / BM25 / RRF 三档（纯 CPU，几秒）
python scripts/benchmark_hybrid_retrieval.py \
    --vectors corpus_with_truth.npy --catalog corpus_with_truth_catalog.jsonl \
    --queries-file queries.npy --queries-catalog queries.jsonl \
    --kinds dense_only,bm25_only,rrf --output-dir results/real_text_query

# 叠 cross-encoder（CPU 约 15 分钟：500 查询 × 100 候选）
python scripts/benchmark_hybrid_retrieval.py \
    --vectors corpus_with_truth.npy --catalog corpus_with_truth_catalog.jsonl \
    --queries-file queries.npy --queries-catalog queries.jsonl \
    --kinds dense_only,rrf,rrf_rerank --reranker cross-encoder \
    --rerank-candidates 100 --save-scores results/real_text_query/ce_scores_100.json \
    --output-dir results/real_text_query

# 复用已落盘分数复跑（秒级，不再加载模型）
python scripts/benchmark_hybrid_retrieval.py ... \
    --reranker cross-encoder --rerank-scores results/real_text_query/ce_scores_100.json
```

线上接入：

```bash
# RRF 融合（不需要索引目录）
shopping-agent --retriever-backend rrf --query "5升 5W-30 机油"

# RRF + cross-encoder 重排
shopping-agent --retriever-backend rrf --reranker cross-encoder --cross-rerank-candidates 100 \
    --query "心形仿珍珠耳环"
```

## 两个口径必须一起读

`hit@10`（严口径）把真值定义为"生成该查询的那一行商品"。A1 已经证明这个定义在电商检索里偏严：
未命中样本中 66.3% 的 top-1 与真值**同 category**，在用户视角下并不是错答案。

所以报告里始终并列一列 `hit@10_category`（真值出现，**或** top-10 里有同 category 商品）。
这一列在稠密基线上是 0.962——**"用户想找的东西在不在结果里"这个问题的答案是 96%**，
而 0.668 衡量的是"能不能精确认出用户点名的那一件"。

两个数字都没有错，回答的是两个不同的问题。把 0.668 单独拿出来说"检索系统只行了 67%"是不准确的；
把 0.962 单独拿出来说"检索没问题"同样不准确——用户搜"5 升 5W-30 机油"拿到 6 罐装 1 夸脱的机油，
仍然是错的。

## 产物

- `results/real_text_query/hybrid_retrieval_benchmark.{md,json}` — 四档策略对照
- `results/real_text_query/ce_scores_100.json` — cross-encoder 对 500 × 100 候选的打分（可复现重排档）

相关文档：[QUERY_DISTRIBUTION.md](QUERY_DISTRIBUTION.md)（分布 gap 量化）、
[REAL_CLIP_BENCHMARK.md](REAL_CLIP_BENCHMARK.md)（真实向量 ANN 压测）、
[SCALING.md](SCALING.md)（规模化与分段下推）。
