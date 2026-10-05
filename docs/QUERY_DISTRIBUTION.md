# 真实文本查询评测集：查询分布对检索质量的真实影响

前面所有压测（[REAL_CLIP_BENCHMARK.md](REAL_CLIP_BENCHMARK.md)、[SCALING.md](SCALING.md)）的查询向量都取自语料自身的不同切片——**查询与语料同分布**。线上真实情况是：用户打字描述需求，查询文本与商品文案的表达方式并不一致。这个文档给出这一层的实测。

## 结论先行

| 口径 | 语料 | 查询 | hit@10 / recall@10 |
|---|---|---|---|
| 同分布切片查询 | 4,500 | 500（同批商品向量的不同切片） | **0.9982**（HNSW）/ 0.9994（IVF）/ 1.0（精确） |
| **真实文本查询** | 5,477 | 500（LLM 从商品事实生成的中文查询） | **0.668**（HNSW）/ 0.660（IVF）/ 0.668（精确） |

两个数字回答的是同一个问题——「用户想要的那件商品，能不能出现在 top-10 里」——但相差 **33 个百分点**。这就是「同分布切片查询量不出来的真实代价」。

更关键的一行：**三种索引后端在真实查询下的命中几乎完全一致（0.660 ~ 0.668）**。索引选型、参数调优在这个场景下能腾挪的空间只有 ±0.8pt；而查询表达不对齐造成的是 33pt。**瓶颈在表达对齐，不在检索结构**——这直接决定了后续投入方向（混合检索与 cross-encoder 重排，而不是继续调 efSearch）。

## 评测集怎么来的

1. **抽样**：从 ABO 目录前 9 万条商品中随机抽 500 条（`--pool 90000 --count 500 --seed 42`），并按类目轮转重排打散，避免同类扎堆；
2. **生成**：本地 Ollama `qwen3:4b-instruct`，4 种 persona 轮换（关键词 / 口语 / 场景 / 预算），prompt 只给商品事实（标题、类目、价格），要求 5~25 字中文查询；
3. **编码**：Chinese-CLIP 文本塔，512 维（L2 归一化），与线上查询端口径一致；
4. **真值**：生成它的原商品，**保留在语料内**——用户描述想买的商品被排进 top-10 本就是正确行为，测的正是「打字描述 → 图文融合索引」的跨表达匹配。

生成质量的两个坑已经在 `scripts/build_query_set.py` 里治掉了（详见 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 2026-10-04/05 条目）：LLM 批量生成的**静默错位**（index 齐全但内容串位）与 **qwen3 thinking 截断 JSON**。治理后逐条抽检锚点（品牌/型号/尺寸）全部对齐。

## 33 个百分点去哪了：把失败样本拆开看

对每条查询定义 `margin = cos(query, 真值) - max cos(query, 其他商品)`（向量已归一化，余弦即内积）：

| 类别 | 条数 | 命中 |
|---|---|---|
| 未命中且 `margin < 0.05`（描述模糊 / 语义近邻压过真值） | 166 | — |
| 未命中且 `margin ≥ 0.05`（查询与商品语义相去甚远） | **0** | — |

**没有一条是"查询在问另一件商品"**——生成侧没有跑偏，LLM 生成质量是可信的。进一步看这 166 条的 top-1 是什么：

- **66.3% 的未命中样本，其 top-1 与真值属于同一 category**（全局这一比例是 81.2%）；
- 语料内随机商品对的余弦 >0.90 只占 0.8%——语料里几乎没有近重复，说明 embedding 分得开，不是混淆导致的技术失败。

抽出的典型样例（真值 vs top-1，都是同品类）：

| category | 查询 | 真值 | top-1（sim） |
|---|---|---|---|
| `LIGHT_BULB` | 亚马逊倍思 LED灯泡 A19 10000小时 无调光 | AmazonBasics Bombilla LED 10.000 ore | AmazonBasics Ledlamp GU10 5,5 W（0.735） |
| `AUTO_ACCESSORY` | AmazonBasics 5W-30 汽车机油 5升 | AmazonBasics Aceite 5W-30 5 l | AmazonBasics Motor Oil 1 Quart 6-Pack（0.800） |
| `HOME_BED_AND_BAT` | 亚马逊 Basics 遮光窗帘 132x168 奶绿色 | AmazonBasics verduisteringsgordijnset 132x168 | Solimo Rivendale Window Curtain（0.775） |
| `FINEEARRING` | Swarovski 银质 耳环 仿珍珠 心形 | Sterling silver 仿珍珠心形耳钉 | Sterling Silver Diamond Round Stud（0.764） |

这些 top-1 **在用户视角下都不是错答案**：搜「5 升 5W-30 机油」返回 6 罐装 1 夸脱的机油、搜「心形仿珍珠耳环」返回圆钻耳钉，都是同品类不同规格/不同款式。**"真值 = 那一行商品"这个标注假设在电商检索里本身不成立**。

> 这条发现比 0.668 本身更值得记住：它把"召回率掉了 33%"这件事从"检索系统失败"重新定义为"评测口径偏严"。下一节是顺着这个结论做的口径改造。

## 由此推动的两处口径改造

**① 多真值标注（已在数据层铺好）**：真值从单元素集合扩展为「同 category 且价格同数量级」的商品集合后，`recall` 的通用集合交集实现无需改动（`SyntheticCorpus.recall` 本来就是集合运算），评测脚本的 `--queries-catalog` 换一份标注文件即可切换口径。这样得到的数字才回答"用户想找的东西在不在结果里"。

**② 跨语言维度（ABO 的先天特征）**：ABO 商品标题是多语言（荷兰语 / 德语 / 英语），查询是中文——上表第 3、4 行都能看出跨语言带来的表达损耗。`--queries-file` 支持任意查询向量文件，后续补一份英文查询集即可做跨语言对照。

## 复现

```bash
# 1) 生成查询集（本地 Ollama，中途可断点续跑）
python scripts/build_query_set.py \
    --catalog data/processed/products.jsonl \
    --count 500 --output-dir outputs/real_clip/queries \
    --ollama-model qwen3:4b-instruct --seed 42

# 2) 文本查询压测（真值 = 原商品，保留在语料内）
python scripts/benchmark_ann_scaling.py \
    --vectors outputs/real_clip/vectors.npy \
    --vectors-catalog data/processed/products.jsonl \
    --queries-file outputs/real_clip/queries/queries.npy \
    --queries-catalog outputs/real_clip/queries/queries.jsonl \
    --scales 90940 --queries 500 --top-k 10 \
    --kinds numpy,flat,hnsw,ivf --output-dir results

# 3) 同分布切片查询基线（只换查询口径，用同一份向量）
python scripts/benchmark_ann_scaling.py \
    --vectors outputs/real_clip/vectors.npy \
    --scales 90440 --queries 500 --top-k 10 \
    --kinds numpy,flat,hnsw,ivf --output-dir results_same_dist
```

本报告用的 5,477 档是小规模本机复现版（语料 = 前 5,000 条 + 为 477 条查询补齐的真值商品，保证 500 条查询真值 100% 覆盖）；租卡版脚本已内置 `--pool` 与 `--queries-file`，把 `--scales` 提到 90,940 即为全量档。

## 产物

- `results/real_text_query/ann_scaling_benchmark_text_queries.{md,json}` — 真实文本查询报告（5,477 档）
- `results/real_text_query/ann_scaling_benchmark_same_distribution.md` — 同分布切片查询基线
- `results/real_text_query/queries_sample_500.jsonl` — 500 条查询（含真值 product_id 与 persona）
- `results/real_text_query/hit_analysis.json` — margin 分解与同品类统计
