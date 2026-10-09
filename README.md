# 多模态内容理解与工具编排 Agent

[![CI](https://github.com/fangyunok/multimodal-shopping-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/fangyunok/multimodal-shopping-agent/actions/workflows/ci.yml)

**面向图文内容的可复现多模态 Agent：内容理解 → 检索 → 工具编排 → 可溯源生成。**

把**多模态内容理解（Chinese-CLIP 共享向量空间）、受 JSON Schema 约束的工具编排、可溯源生成与分层离线评测**连成完整链路；检索、规划、工具、回答四层解耦，可独立替换与评测。项目不只展示成功案例，也保留稠密检索退化、Planner 非法输出等负结果。

## 先看结论

| 子系统 | 对照结果 | 结论与边界 |
|---|---|---|
| 多模态内容理解与检索（图文共享向量空间） | Chinese-CLIP 图文索引 Recall@1 `0.1667 → 0.7500` | 100 商品、12 条跨视角 test 查询；95% CI `[0.50, 1.00]`，不可外推到完整商品分布 |
| 属性文本检索与融合 | CLIP 没有全面超过词法基线 | 品牌、型号等精确 token 仍需词法信号，因此保留融合检索 |
| 混合检索与重排 | 真实文本查询 hit@10 `0.668 → 0.770 → 0.918`（RRF 融合 BM25，再叠 cross-encoder 重排） | 两路证据错误模式不同才互补——BM25 单独只有 0.598；重排把候选集（召回 0.946）的 97.0% 捞进 top-10，宽口径 0.986。粗排候选窗口是策略的一部分，引用数字须连窗口一起给；口径见[混合检索文档](docs/HYBRID_RETRIEVAL.md) |
| Prompt 与规划编排 | Qwen2.5-3B 联合准确率 `0.36 → 0.51`、类目准确率 `0.49 → 0.87` | 同一 100 条锁定集；同时产生 10% 无效规划，预算提取仍是短板 |
| 工具编排与可溯源生成 | Schema Dispatcher + 内容 ID 引用 | 模型不能直接编造价格和库存；非法参数可拒绝、引用忠实度可度量 |
| 在线性能 | 规则 Agent 端到端 P50 `0.07 ms` / P95 `0.41 ms`；LLM Planner P50 `188 ms` / P95 `236 ms` | 计时在商品目录与索引预加载后统计，不含进程启动与模型加载 |
| MCP 协议链路 | tools / resources / prompts 三类能力完整暴露，支持 stdio、SSE 与 streamable-http | stdio 吞吐约 `360 req/s` 饱和且不随并发增长，瓶颈在串行管线而非检索；口径见 [MCP 文档](docs/MCP_SERVER.md) |
| 多轮会话与循环控制 | 会话槽位跨轮继承；上下文预算为硬边界；ReAct 轨迹逐步可审计 | 追问「有货吗」直接复用上一轮候选，工具调用 3 次 → 1 次；40 轮会话在 512–4096 tokens 四档预算下峰值均未越界；口径见[运行期文档](docs/AGENT_RUNTIME.md) |
| 规模化检索 | 10 万条 512 维下，HNSW P50 `0.59 ms`（recall@10 `0.997`），numpy 精确 `8.9 ms`；IVF-PQ 索引体积 `9.4 MB`（1/21） | 五种后端可切换，索引类型写在清单里；**装了 faiss 不等于更快**——`IndexFlatIP` 反而比 numpy 慢约 3 倍（单线程 vs BLAS）。口径与失败案例见[规模化文档](docs/SCALING.md) |
| 过滤下推（分段索引路由） | 10 万条 + 1% 放行率下，严苛档集合一致率 IVF `0.78 → 1.0`、HNSW `0.18 → 1.0`；扫描商品占比 `0.0625`（16 段）～`0.0156`（64 段） | 按价位分段建索引，查询时先选段再检索。**对比分段与不分段必须先固定兜底策略**，否则会把"兜底救回来的一次全量扫描"误读成分段的效果；粒度不够时分段只是一个更慢的单索引 |
| 无状态服务化 | 会话外置到 Redis，同 `session_id` 可落任意副本；存活/就绪探针分离 | 写入用 Lua 原子 CAS，冲突计数可见而不是静默覆盖；索引版本支持原子发布与回滚 |
| 后端与工程交付 | FastAPI、Web Demo、MCP Server、Docker、CI、一键 CPU 基线 | 无 GPU 也能验证 API、工具、MCP 协议与评测链路 |

详细口径、失败案例和复现配置见[实验记录](docs/EXPERIMENT_LOG.md)与[项目摘要](docs/PROJECT_SUMMARY.md)。

## 系统架构

```mermaid
flowchart LR
    A[文本/图片请求] --> B[Planner]
    B --> C[Schema Dispatcher]
    C --> D[商品搜索]
    C --> E[库存检查]
    C --> F[商品对比]
    D --> G[词法 + Chinese-CLIP 检索]
    G --> H[带商品ID引用的回答]
    E --> H
    F --> H
    B -. 规则基线 / Qwen2.5-3B .-> B
```

这个仓库聚焦**多模态内容理解与检索、Prompt 规划、Agent 工具编排**；模型后训练和纯推荐排序不是这里的实验变量。

项目文档：[技术摘要](docs/PROJECT_SUMMARY.md)｜[设计决策与验证指南](docs/TECHNICAL_GUIDE.md)｜[完整实验记录](docs/EXPERIMENT_LOG.md)｜[MCP 接口与容量实测](docs/MCP_SERVER.md)｜[会话记忆与 ReAct 循环](docs/AGENT_RUNTIME.md)｜[规模化改造与 ANN 实测](docs/SCALING.md)｜[部署与运维](docs/DEPLOYMENT.md)

Planner 默认使用可复现的规则基线，也可切换到 OpenAI-compatible LLM 后端；配置与公平评测方法见 [LLM Planner 文档](docs/LLM_PLANNER.md)，GPU 复现实验见 [AutoDL 指南](docs/AUTODL_PLANNER.md)。

HTTP 业务接入现支持可替换的外部库存、调用方鉴权与会话隔离，以及 `/chat` 的大模型多步工具决策。
默认模式仍使用演示目录和规则决策；配置方法、库存契约、程序约束和测试边界见 [业务接入文档](docs/BUSINESS_INTEGRATION.md)。

项目包含 [20 条开发集与 100 条锁定 Planner 测试集](docs/PLANNER_DATASET.md)。运行规则基线：

```powershell
.venv\Scripts\shopping-agent --catalog data/products.jsonl --evaluate-planner data/planner_test.jsonl
```

## 5分钟验证

Windows下从干净克隆运行CPU闭环（安装依赖、测试、生成小型benchmark并执行Agent评测）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_cpu_baseline.ps1
```

这条路径不下载Chinese-CLIP，也不调用外部LLM；它用于验证工程链路，不冒充真实多模态模型成绩。

真实多模态实验可通过 `scripts/run_abo_clip_experiment.ps1` 一次复现；参数和所需 ABO 文件见 [数据集说明](docs/ABO_DATASET.md)。

## CPU 基线：不依赖 GPU 也能跑通闭环

本阶段不依赖显卡或外部 API，确保任何人克隆仓库后都能复现：

- 商品文本检索与价格、类目约束；
- 基于 RGB 直方图的轻量图片相似度基线；
- 文本与图片分数融合；
- 搜索、库存检查、商品对比工具；
- 可审计的 Agent 工具调用轨迹；
- 从自然语言抽取预算、类目和比较意图的规则规划基线；
- 可直接提供给 LLM Function Calling 的工具名称、描述和 JSON Schema；
- 检索命中率、工具选择准确率、约束满足率评测；
- 端到端任务成功率和平均工具调用步数；
- 非法工具调用率，以及模型/索引预加载后的 Agent P50/P95 延迟；
- 商品事实回答的结构化引用、引用命中率和 grounded answer rate；
- FastAPI 服务和自动化测试。
- JPEG、PNG、WebP 图片上传接口（5 MB 限制与内容校验）。
- 标准 MCP 接口（tools / resources / prompts）与 stdio、SSE、streamable-http 三种传输。

轻量图片特征只是 CPU 冒烟基线，不把颜色相似误称为语义理解。项目也已接入中文 CLIP，实现共享向量空间中的以文搜图、以图搜图和图文联合检索；两种后端使用同一套 Recall@K/MRR 协议比较。

项目已经提供可选的中文 CLIP 检索后端。首次启用会下载模型，本机 CPU 可以推理但较慢：

```bash
.venv\\Scripts\\python -m pip install -e ".[clip,dev]"
$env:RETRIEVER_BACKEND="clip"
$env:MODEL_DEVICE="cpu"
.venv\\Scripts\\uvicorn shopping_agent.app:app
```

默认 `RETRIEVER_BACKEND=baseline`，自动化测试不下载模型。可设置 `RETRIEVER_BACKEND=fusion` 使用词法与 CLIP 分数融合，并通过 `LEXICAL_WEIGHT` 调整词法权重。使用已下载权重时，可把 `CLIP_MODEL_PATH` 指向本地模型目录，同时保留 `CLIP_MODEL=OFA-Sys/chinese-clip-vit-base-patch16` 作为索引校验使用的标准模型 ID，并设置 `INDEX_DIR` 加载离线索引。

真实数据准备完成后，离线构建并复用商品向量：

```bash
.venv\\Scripts\\shopping-agent \
  --catalog data/products.jsonl \
  --build-index data/index/chinese-clip-base \
  --device cpu
$env:INDEX_DIR="data/index/chinese-clip-base"
```

`--model` 可以是本地模型目录，`--model-id` 用于把标准模型 ID 写入索引。`--image-weight` 可在 0 到 1 之间设置商品图片在离线向量中的融合权重，便于做消融实验。`manifest.json` 会记录商品目录 SHA-256、模型名、融合权重、向量维度和生成时间。商品目录发生变化或模型不一致时，服务拒绝加载旧索引。

## 数据准备

原始商品表支持 JSONL 和 CSV。每条商品必须记录 `source` 与 `license`，本地图片会验证格式、按 SHA-256 去重，并按照商品 ID 稳定切分：

```bash
.venv\\Scripts\\shopping-agent \
  --prepare-data data/raw/products.jsonl \
  --output-catalog data/processed/products.jsonl \
  --image-dir data/processed/images
```

处理后的数据和图片默认不提交 Git，数据规范见 `docs/DATASET_CARD.md`。

首个真实数据源使用 Amazon Berkeley Objects（ABO）的 256px 图片版本。下载并解压官方归档后，可抽取 1000 条商品子集：

```bash
.venv\\Scripts\\shopping-agent --convert-abo D:/datasets/abo --abo-limit 1000
```

如果只下载了 metadata，可按清单并发获取所需的1000张小图：

```bash
.venv\\Scripts\\shopping-agent --fetch-abo-images D:/datasets/abo --abo-limit 1000
```

加入 `--include-other-images` 可同时下载 `other_image_id` 指向的其他视角；随后使用 `--build-cross-view-benchmark D:/datasets/abo` 可生成查询图不同于索引主图的无泄漏图片 benchmark。ABO 不含可靠价格和实时库存，只用于图文检索；数据详情和署名要求见 `docs/ABO_DATASET.md`。

### 复现 Recall@1 `0.1667 → 0.7500`

表中的多模态检索结果由 `scripts/run_abo_clip_experiment.ps1` 一次复现，脚本会依次下载主图/其他视角、转换并稳定划分数据（100 商品、70/18/12）、生成属性与跨视角 benchmark、构建文本-only / 图片-only / 图文索引，并运行词法、RGB、CLIP 与融合对照：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_abo_clip_experiment.ps1 `
  -AboRoot D:/datasets/abo `
  -ModelPath D:/models/chinese-clip-vit-base-patch16 `
  -Limit 100
```

脚本对每个外部命令显式检查退出码，中间任一步失败都会终止。跑完后应观察到下列口径（跨视角 benchmark，12 条查询；原始输出见 [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md)）：

| 方法 | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| RGB 直方图 CPU 基线 | 0.1667 | 0.1667 | 0.2500 | 0.1786 | 0.1944 |
| Chinese-CLIP 图片-only 索引 | 0.6667 | 0.6667 | 0.7500 | 0.6806 | 0.6964 |
| Chinese-CLIP 商品图文等权索引 | 0.7500 | 0.7500 | 0.7500 | 0.7500 | 0.7500 |

> **口径说明**：`0.7500` 基于 12 条跨视角 test 查询，bootstrap 95% CI 为 `[0.50, 1.00]`，样本量小，不作为无偏的最终成绩，也不能外推到完整电商分布。图片-only 低于图文索引，说明商品文本在共享向量空间提供了互补语义。

原始归档与本地模型权重体积较大（ABO listings ≈ 83 MB、images-small ≈ 3 GB），未提交到仓库；请按 [数据集说明](docs/ABO_DATASET.md) 自行准备，再运行上述命令复现全部指标。

检索评测统一输出 Recall@1/5/10、MRR、nDCG@K 与固定随机种子的 bootstrap 95% 置信区间：

```bash
.venv\\Scripts\\shopping-agent --catalog data/processed/products.jsonl --evaluate-retrieval text
.venv\\Scripts\\shopping-agent --catalog data/processed/products.jsonl --evaluate-retrieval image
```

标题或原图检索自身只作为链路检查，不能代表语义检索效果。评测结果还会列出未排在首位的逐条失败案例及实际召回 ID，便于错误分析。

生成不复制商品标题的属性需求评测集，并运行统一指标：

```bash
.venv\\Scripts\\shopping-agent --catalog data/processed/products.jsonl --build-benchmark outputs/retrieval_benchmark.jsonl
.venv\\Scripts\\shopping-agent --catalog data/processed/products.jsonl --benchmark outputs/retrieval_benchmark.jsonl
```

CLIP 后端可在上述命令后增加 `--retriever-backend clip --index-dir data/index/chinese-clip-base --model <本地模型目录>`。自动生成的查询必须经过人工抽查，最终测试集应补充同义改写、属性组合、类目歧义和不同视角图片。

## 快速开始

Windows 下可从干净克隆一条命令安装 CPU 依赖、运行测试、生成公开小型 benchmark 并执行 Agent 评测：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_cpu_baseline.ps1
```

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\python -m pip install -e ".[dev]"
.venv\\Scripts\\python -m pytest
.venv\\Scripts\\shopping-agent --query "白色简约通勤运动鞋" --max-price 500
.venv\\Scripts\\shopping-agent --evaluate data/eval.jsonl
.venv\\Scripts\\uvicorn shopping_agent.app:app --reload
```

服务启动后访问 `http://127.0.0.1:8000/docs`，可以调用 `/search`、`/agent`、`/agent/image` 和 `/tools`。

访问 `http://127.0.0.1:8000/` 可使用内置演示页面，无需Node.js或单独构建前端。

图片请求使用 `multipart/form-data`，不要向服务暴露本机文件路径：

```bash
curl -X POST http://127.0.0.1:8000/agent/image \
  -F "image=@query.png" \
  -F "query=白色简约通勤运动鞋" \
  -F "max_price=500"
```

## Docker 部署

容器默认使用无需GPU和模型下载的CPU基线，并以非root用户运行：

```bash
docker compose up --build -d
curl http://127.0.0.1:8010/health
```

访问 `http://127.0.0.1:8010/docs`。镜像内置健康检查，GitHub Actions会在每次提交后同时执行Python测试和Docker构建。

## MCP Server

同一套检索、比价、库存与端到端追问能力，按 Model Context Protocol 暴露，供 Claude Desktop、Cursor 等任意 MCP 客户端直接接入。业务逻辑没有改动，只把已有的工具协议层（`ToolDefinition`）换了一个标准传输暴露出去。

```bash
.venv\\Scripts\\python -m shopping_agent.mcp_server --transport stdio --catalog data/products.jsonl
```

| 能力类别 | 内容 |
|---|---|
| Tools | `search_products`、`compare_products`、`check_inventory`、`ask_shopping_agent`、`react_query`（多轮会话）、`get_session_state`、`get_runtime_metrics` |
| Resources | `catalog://products`、`catalog://schema`、`catalog://products/{product_id}` |
| Prompts | `grounded_recommendation`（强制引用商品 ID）、`tool_orchestration_plan` |

传输支持 stdio、SSE 与 streamable-http。工程层内建并发闸门、超时、指数退避重试、只读工具幂等重放与逐工具 P50/P95/P99 指标；失败统一转为结构化错误（`invalid_arguments` / `permission_denied` / `not_found` / `timeout` / `tool_failure`），不把异常直接抛给模型。本地图片路径默认拒绝，需显式开启。

有状态的 `react_query` 会**绕开幂等重放**：重复调用本就应该改变会话状态，命中 30 秒重放缓存只会返回陈旧快照并跳过状态更新。

实测结论（每档 best-of-N；Windows、Python 3.11.9、5 条商品目录）：

| 传输 | 并发 1 吞吐 | 并发 8 吞吐 | 并发 1 P50 | 并发 8 P50 |
|---|---|---|---|---|
| 进程内回环 | 1121.4 req/s | 1207.1 req/s | 0.76 ms | 3.67 ms |
| stdio 子进程 | 246.5 req/s | 365.2 req/s | 3.89 ms | 21.84 ms |

**两种传输的吞吐都不随并发增长**（分别在约 1.2k 与约 365 req/s 饱和），P50 随并发近似线性上升 —— 瓶颈在单条串行的请求/响应管线，不在检索逻辑；5 条商品的目录下业务耗时可忽略。因此高扇出场景应改用 streamable-http，把并发落到独立连接而不是在 stdio 单管道里排队。原始结果见 [`results/`](results/)。

复现压测与完整口径：

```bash
.venv\\Scripts\\python scripts/benchmark_mcp.py --transport stdio --concurrency 1,2,4,8
```

接入配置、错误分类表、设计取舍与已知限制见 [MCP Server 文档](docs/MCP_SERVER.md)。

## 多轮会话与 ReAct 循环

单轮 Agent 每次请求都从零开始；真实导购是多轮的——用户会追问「那有货吗」、追加预算、改类目。
这一层由三个解耦模块支撑，全部不依赖外部模型，可在 CPU 干净环境复现。

**会话记忆**（`session.py`）。跨轮累积五类槽位，规则只有两条：**新值覆盖旧值，未提及的继承**。
其中 `last_candidates`（上一次检索返回的候选）收益最直接——用户问「那有货吗」时不再重新检索，
而是直接用记住的候选查库存，**工具调用从 3 次降到 1 次**，也不会因重新检索漂移到别的商品。

**上下文预算**（`session.py`）。`window()` 在 token 预算内把事件流压成 messages，四阶段：
折叠最旧轮次为摘要 → 收缩摘要 → 逐条丢弃 → 截断超额的最近一条。
槽位摘要始终注入上下文头部，因此**压缩不会丢掉已确认的约束**。压缩是就地且持久的，
重复调用不会重复累积摘要，这正是 ReAct 每步重算窗口却不越压越大的前提。

实测（`scripts/benchmark_agent_runtime.py`，40 轮连续会话，不含模型推理开销）：

| 预算 (tokens) | 可用 (tokens) | 轮数 | 最终占用 | 峰值占用 | 首次压缩轮次 | 是否越界 |
|---|---|---|---|---|---|---|
| 512 | 256 | 40 | 192 | 244 | 4 | 否 |
| 1024 | 768 | 40 | 681 | 766 | 3 | 否 |
| 2048 | 1792 | 40 | 1689 | 1790 | 12 | 否 |
| 4096 | 3840 | 40 | 3749 | 3832 | 20 | 否 |

四档预算下峰值均未越界。补阶段四的原因正是一次实测：单条检索观察就可能大于剩余额度，
只丢轮次会让预算被突破几 token——**预算的意义在于它是硬边界**。

**ReAct 循环**（`react.py`）。每一步记下 `thought / action / arguments / observation / error / latency_ms`，
最终答案必须能回溯到具体工具返回。三处工程重点：

- **观察回注**：工具失败不中断循环，而是分类为 `invalid_arguments` / `not_found` / `unknown_tool` / `tool_failure` 参与下一步决策；一个工具抛异常不该让会话崩掉。
- **失控防护**：只有五种终止原因——`final_answer`、`max_steps`、`loop_detected`（同一 `(tool, arguments)` 被重复请求）、`budget_exhausted`、`no_candidates`。`max_steps` 防步数过多，`loop_detected` 防原地打转。
- **预算感知**：每步后重算窗口，压缩后仍超预算即提前收口。上下文管理是循环里的退出条件，不是事后的日志清理。

实测 15 次运行：平均工具调用 3.0 次、单轮最多 4 次，端到端 P50 `2.02 ms` / P95 `3.14 ms`。

```bash
.venv\\Scripts\\python -m shopping_agent.cli --catalog data/products.jsonl --chat
```

**边界**：`RuleReasoner` 是确定性基线，不含真实 LLM 决策器（其步数与延迟是该接口的上界参考）。
会话默认存在进程内（`SESSION_BACKEND=memory`），要跨副本共享需切到 Redis——这一层已经实现并测试，
见下节。设计取舍与完整口径见 [运行期文档](docs/AGENT_RUNTIME.md)。

## 规模化改造：索引可切换、服务无状态、编码可拆分

原始仓库只有个位数商品，`product_vectors @ query_vector` 一次矩阵乘法就结束了。
这带来一个结构性盲区：**所有"规模上去了会怎样"的判断都无法被验证**，只能停留在"理论上该换 ANN"。
所以这一轮做的是把口径补上——先造出可复现的规模化语料与精确真值，再让每个结论都有实测数字。

改造分三层，每层都能独立回退：**索引层**（`ann_index.py` / `faiss_retrieval.py` / `partitioned.py`）、
**服务层**（`session_store.py` / `app.py`）、**编码与版本层**（`encoder_service.py` / `incremental.py`）。

### 索引层：五种后端同一个契约

`build(vectors, config) -> AnnIndex`、`AnnIndex.search(queries, top_k) -> (scores, indices)`。
调用方完全不知道底下是什么，因此"换索引"不会渗透进检索逻辑，也就不会变成不可回退的改造。
索引类型、维度、实际生效参数全部写进**索引清单（v2）**，加载时以清单为准、运行期只覆盖旋钮。

实测（`scripts/benchmark_ann_scaling.py`，512 维 × 128 簇合成语料，CPU，200 条查询）：

| 规模 | 后端 | 构建(s) | 索引体积 | P50 | P99 | 单条 QPS | recall@10 |
|---|---|---|---|---|---|---|---|
| 10 万 | numpy 精确 | 0.14 | 195 MB | 8.93 ms | 10.61 ms | 111 | 1.0 |
| 10 万 | faiss `IndexFlatIP` | 0.07 | 195 MB | 25.97 ms | 31.97 ms | 37 | 1.0 |
| 10 万 | HNSW | 27.0 | 221 MB | **0.59 ms** | 1.14 ms | **1490** | **0.997** |
| 10 万 | IVF | 9.7 | 198 MB | 0.96 ms | 1.52 ms | 984 | **1.0** |
| 10 万 | IVF-PQ | 194.1 | **9.4 MB** | 1.27 ms | 1.78 ms | 755 | 0.097 |

（构建耗时/体积/recall 每次运行都很稳；P50/P95/QPS 跨次运行差异可达数十个百分点，本机没有做 CPU 亲和性绑定。要精确对比 5% 以内的差距必须固定线程数与机器状态。）

三个反直觉但重要的结论：

1. **装了 faiss 不等于更快。** `IndexFlatIP` 在 10 万条上比纯 NumPy 慢 3 倍——FAISS 的精确索引是单线程的，而 NumPy 的 `@` 走多线程 BLAS。FAISS 的价值在**近似**索引，不在精确索引。
2. **内存只有 IVF-PQ 能真正降下来。** HNSW 和 IVF 都是"不压缩"的，体积甚至比原始向量略大（要存图结构 / 桶号）。10 万条的 195 MB 是 `100000 × 512 × 4 B` 的硬账，百万条就是 2 GB/副本。
3. **PQ 单独用的召回不可接受，但作为"粗召回 + 精确重排"的第一级是可行的。** 10 万条下 PQ 的 recall@10 只有 0.097；取 500 个候选再用原始向量精确重排，recall@10 回到 **0.9375**（1 万条下 0.9995），而内存仍只占 9.4 MB。多出的代价是每查 500 次精确内积，512 维下可以忽略。

### 检索层：自适应超采样 + 精确兜底

ANN 和业务过滤（预算、类目）放在一起会出事：ANN 返回的 top-k 是离查询最近的 k 个，
而其中可能**没有一个是符合预算的**，于是过滤后返回 0 条。这里的做法是**自适应超采样**——
候选不够就翻四倍重来，把 `rounds` / `examined` 记进 `last_stats` 让过程可观测。

但超采样有个反直觉的上限，是实测才发现的：**图索引会"交付不足"。** HNSW 返回的是束搜索
**访问到的节点**，不是距离最近的 k 个——实测 `efSearch=64` 时 k=512 以内能足额交付，
k 再大就只能给出约 700 条（faiss 用 `-1` 把结果补齐到 k，不数有效条数根本发现不了）。
这时继续放大 k 是徒劳的，而且拿到的候选集已被"访问到哪些"而非"离得多近"污染——
**哪怕这次凑够了 top_k，结果也已经和精确检索不一致**。

所以判据不是"凑没凑满"，而是"索引有没有交足"：一旦发现交付不足，直接退化到全量精确扫描
（前提是索引目录带 `vectors.npy`，`--no-vectors` 会关掉这条路）。这是在
"宁可这一次慢一点"和"宁可漏掉一个本该出现的便宜商品"之间选了前者，并落在 `last_stats.fallback` 上可见。

过滤压力测试（点名查询，50 条探针）的判据刻意**不是**"是否凑满 top_k"，而是**与精确后端的结果集是否一致**：
在只放行 1% 商品的目录上，精确检索本身也凑不满 top_k，用"凑满"当标准会得出"精确后端也不合格"的荒谬结论。

### 混合检索层：两路证据 + 名次融合 + 重排

[分布 gap 量化](docs/QUERY_DISTRIBUTION.md)用 500 条真实中文查询测出：同分布切片查询 recall@10 = 0.9982，
用户打字描述只有 0.668，**差 33 个百分点**；而五种索引后端在这个场景下的命中差异只有 ±0.8pt。
结论是瓶颈不在检索结构，而在**只有一路证据**——稠密向量擅长"意思相近"，但对精确词项是糊的。

所以补了两条与稠密检索错误模式不同的路径：

- **`bm25.py` 稀疏检索**：中文按"单字 + 相邻二元组"切分（单字保召回、二元组保精度），
  拉丁串整体保留——`5W-30`、`132x168`、`J7` 这类规格型号是最强的区分度信号，拆成单字符会让它们消失。
  打分是 postings 上的稀疏累加，复杂度只随查询词的实际分布度增长，不随语料条数增长。
- **`rrf.py` 名次融合**：`RRF(d) = Σ w_r / (k + rank_r(d))`，**只吃名次不吃分数**。
  BM25 分数无上界、CLIP 余弦落在 [-1, 1]，直接加权等于让 `lexical_weight` 去承担全部量纲对齐的活；
  RRF 天然免疫这个问题，`rrf_k=60` 是唯一的平滑常数。

**重排（`rerank.py`）必须放在"粗排放宽之后"**，这是两阶段检索最容易做错的地方：
粗排就截到 top_k 的话，重排只能在这 k 条里换顺序，永远换不出更好的东西。
`RerankRetriever` 因此要求检索器提供 `search_candidates(request, count)` 这一候选级入口，
并把"重排了多少条、模型推理多少毫秒"写进 `last_stats`。`ExactVectorReranker` 是零依赖降级路径，
`CrossEncoderReranker`（bge-reranker）是能改变排序的那一级——双塔编码器分别编码 query 与 doc，
两者之间没有交互，所以它对词序、规格数字天生不敏感，cross-encoder 有交互。

### 过滤下推：把过滤做进索引结构

超采样解决不了"过滤强度高"这一类查询，根因不在参数上：**价格过滤与向量相似度本来就无关**，
符合条件的前 10 个商品在向量空间里位置是随机的。任何只扫描部分子空间的近似索引都可能压根
没访问到它们——候选池里没有的东西，再怎么重排也变不出来。10 万条 + 1% 放行率下，IVF 的集合一致率
停在 0.78 且**不会**因为兜底而改善，因为它每次都能"足额交付"，只是那 k 个都来自被扫描的桶。

所以正解是让过滤**发生在检索之前**：按价位分段建索引，查询时先用 `max_price` 选段，
只在选中的段里检索——被过滤掉的分段根本不会被查询。严苛档（放行 1%）的集合一致率：

| 规模 | 后端 | 不分段 | 4 段 | 16 段 | 64 段 |
| --- | --- | --- | --- | --- | --- |
| 1 万 | IVF | 0.06 | 0.42 | **1.0** | **1.0** |
| 1 万 | HNSW | 0.00 | 0.34 | 0.92 | **1.0** |
| 10 万 | IVF | 0.78 | 0.84 | 0.98 | **1.0** |
| 10 万 | HNSW | 0.18 | 0.52 | 0.98 | **1.0** |

扫描商品占比（理论下限即过滤的选择性，严苛档约 1%）：4 段 `0.25`、16 段 `0.0625`、64 段 `0.0156`。

三条必须一起读的结论，写在[规模化文档 2.7 节](docs/SCALING.md)：**① 对比分段与不分段必须先固定兜底策略**
（不分段那行的"好"可能来自一次全量精确扫描，混在一起归因会得出"分段有害"的错误结论）；
**② 收益取决于分段粒度与过滤选择性的匹配**（粒度不够时分段只是一个更慢的单索引；匹配上之后扫描占比几乎贴着 1% 的理论下限）；
**③ 分段有下限**，段切细了 IVF-PQ 直接拒绝训练。分段省的是查询不是构建——10 万条 HNSW 切 64 段后构造成本翻倍。

附带一条更干净的判据：**分段之后，精确兜底的开关不再改变任何结果**（不分段时两者差很远：1 万条 HNSW `0.9` vs `0.0`）。
因为分段已经消掉了"候选耗尽"这个失败模式，而兜底本来就是为它准备的——
所以**兜底还在生效，就说明过滤压力还没有被真正下推掉**。

### 服务层：让副本可以随便加

会话原本在进程内存里，副本一旦超过 1 个就会碎：用户在副本 A 说的话，下一个请求被负载均衡丢到副本 B，
B 不知道这个会话，"预算上限 ¥500"凭空消失。表现是"多轮记忆时好时坏"，根因是**状态位置错了**。

`SessionStore` 把"会话存在哪"抽出来（内存 / Redis 双实现），写入用**版本化 CAS**：
只有"我读到的版本 == 存储里的版本"才允许写，否则拒绝陈旧提交并返回 HTTP 409。比较与写入用 Lua 脚本在 Redis 端原子完成，
而不是"GET 出来比一下再 SET"——后者中间有竞态窗口，等于没做。

并发请求可能来自多个页面或网络重试。冲突不会自动覆盖或重放工具；调用方应读取最新状态后重新决策。
成功响应的 `conflicts` 为 0。需要严格顺序时，应按会话串行覆盖读取、执行和提交；
把请求路由到同一副本本身不能保证串行。

服务接口因此变成无状态的：`POST /chat` 的会话由外部存储承载，
`/healthz`（存活，不查下游）与 `/readyz`（就绪，查检索器 / 索引 / 会话存储）分离。
分开的原因很实际：Redis 挂了的时候，就绪探针该让流量绕开这个副本，
但如果接到存活探针上，编排系统会**不断重启一个本来没坏的进程**——而重启治不好一个挂掉的 Redis。

### 编码与版本层：拆掉最贵的一步

模型前向是这个系统里最贵的一步，而它和检索副本的扩缩容诉求正好相反：
编码要"少而强"（一台 GPU 吃满批），检索要"多而廉"（CPU 副本廉价横向扩）。
设了 `ENCODER_BASE_URL` 之后检索副本**镜像里不需要 torch**，走 HTTP 到独立编码服务。

编码服务内部有两层：`DynamicBatcher` 把多个请求的编码合并成一批（低 QPS 等 `max_wait_ms` 就走，
不会为了攒批把尾延迟拖到几百毫秒），`EmbeddingCache` 两级缓存（进程内 LRU + 可选 Redis）。
图片走 base64 而不是文件路径——服务端不该能读客户端文件系统，允许传路径等于把"任意文件读取"做成一个接口。

索引侧用**版本目录 + `CURRENT` 指针**：新版本建好先不接流量，`os.replace` 原子切换指针（同一文件系统内的
rename 是原子的，不存在"读到一半指针"），出问题 `rollback` 把指针改回去，不需要重建。
日常上新走**增量**：变更日志 upsert / delete → 以当前版本为基线产出新版本。
删除会留下空洞，而向量行号必须和商品下标严格一一对应（否则会把 A 的向量返回成 B 的商品），所以删除后要**压实**重排。

```bash
# 建版本（不接流量）→ 原子发布 → 出问题回滚 → 清理旧版本
python -m shopping_agent.cli --build-ann-index data/index/v3 --ann-kind hnsw --catalog-snapshot
python -m shopping_agent.cli --promote-version 3 --index-root data/index
python -m shopping_agent.cli --rollback-index --index-root data/index
python -m shopping_agent.cli --prune-versions 2 --index-root data/index
```

完整方法、容量估算、面试取舍与失败案例见 **[docs/SCALING.md](docs/SCALING.md)**；部署与故障处置见 **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**。

**边界与诚实说明**：`SyntheticEncoder` **不能**用来评估真实图文语义检索质量，它只能评估**检索管线与索引结构**——
合成语料没有真实的语义分布，这里的 recall 说明的是"索引能不能把近邻找回来"，不是"模型懂不懂商品"。
十万级以下的数字都来自本机 CPU 单进程，多副本并发与百万级规模未在此环境验证。

## 能力边界与可迁移性

- **与任务无关、可直接复用**：共享向量检索层（双塔图文编码 + 词法/向量分数融合）、Schema Dispatcher 工具协议层、MCP 协议适配层（tools / resources / prompts 映射 + 超时重试幂等并发闸门）、分层评测框架（Recall@K / MRR / nDCG / 任务成功率 / 参数准确率 / 非法调用率 / 引用忠实度 / P50–P95 延迟）。
- **与本任务绑定**：商品目录与字段 schema、搜索 / 库存 / 比价三类工具、跨视角图片 benchmark。
- **迁移到新的内容场景时**，只需替换后两者，检索层、工具协议层、MCP 适配层与评测协议可直接沿用。

## 工程结构

```text
data/                         示例商品与评测集
src/shopping_agent/
  retrieval.py               CPU 图文混合检索
  semantic_retrieval.py      Chinese-CLIP 共享向量检索
  fusion_retrieval.py        词法 + 向量分数融合（线性加权）
  bm25.py                    稀疏检索层：BM25 倒排索引（中文单字+二元组分词、postings 稀疏打分）
  rrf.py                     RRF 名次融合：多路检索只按名次合并，免疫分数量纲差异
  rerank.py                  重排层：精确向量重排（零依赖降级）+ cross-encoder 重排（可选依赖）
  ann_index.py               ANN 索引层：numpy / flat / hnsw / ivf / ivfpq 五种后端同一契约
  faiss_retrieval.py         ANN 检索后端：自适应超采样、精确兜底、粗召回 + 精确重排、运行期统计
  partitioned.py             过滤下推：按价位/类目分段建索引，查询时先按过滤条件选段再检索
  indexing.py                索引构建与清单（v2）：类型、维度、实际生效参数、商品快照
  incremental.py             增量更新与索引版本：变更日志、压实、版本目录、原子发布/回滚
  synthetic.py               合成规模化语料：簇结构向量 + 精确真值 + 无模型编码器
  encoder_service.py         编码服务化：动态批处理、两级缓存、HTTP 适配
  session_store.py           会话外置：SessionStore 协议、内存/Redis 实现、Lua 原子 CAS
  tools.py                   可调用工具（搜索 / 库存 / 对比）
  agent.py                   可解释的决策与工具编排
  planner.py / llm_planner.py  规则规划基线 / LLM 规划后端
  mcp_server.py              MCP 协议适配层：tools / resources / prompts + 韧性层与指标
  session.py                 会话记忆、跨轮槽位累积与上下文预算
  react.py                   ReAct 循环：多步轨迹、观察回注、循环检测与预算感知
  evaluation.py              离线评测
  app.py                     FastAPI 服务（含 /healthz 存活、/readyz 就绪、/chat 无状态会话、/index 版本状态）
scripts/benchmark_mcp.py      MCP 并发压测（进程内回环 / stdio 子进程两种口径）
scripts/benchmark_agent_runtime.py  会话上下文预算与 ReAct 步数 / 延迟实测
scripts/benchmark_ann_scaling.py    ANN 规模化压测：索引层 + 重排对照 + 端到端 + 过滤压力（含分段下推），产出 results/*.md
scripts/benchmark_hybrid_retrieval.py  混合检索对照：稠密 / BM25 / RRF / RRF+重排 同一批真实查询上的 hit@k 与延迟
scripts/serve_encoder.py      GPU 编码服务（/embed/text、/embed/image、/metrics）
tests/                        单元与接口测试
```

## 后续迭代方向

**已完成**

- M0：仓库、数据契约、测试与 API 骨架
- M1：CPU 图文检索基线、工具调用与评测闭环
- M2：Chinese-CLIP、持久化索引、ABO 数据管线与无泄漏跨视角评测
- M3 基线：规则 Planner、JSON Schema Dispatcher、搜索/库存/对比工具及错误轨迹
- M4 基线：RAG 引用忠实度、工具参数、任务成功率、延迟、置信区间与消融实验
- M5 CPU 版：内置 Web 演示、FastAPI、Docker、CI 和一键复现脚本
- LLM Planner 工程版：Schema 约束、20 条开发集、100 条锁定测试集和成本统计
- M6 MCP Server：tools / resources / prompts 三类能力、stdio 与 SSE / streamable-http 三种传输、超时重试幂等与并发闸门、逐工具延迟指标、双口径压测脚本
- M7 多轮运行期：跨轮会话记忆与槽位继承、token 预算下的上下文压缩（四阶段、硬边界）、ReAct 多步循环（观察回注 / 循环检测 / 预算感知）、MCP `react_query` 会话化与重放冲突处理
- M8 规模化（索引层）：五种后端统一契约、索引清单 v2、自适应超采样与精确兜底、粗召回 + 精确重排（`rerank_candidates`）、过滤下推的分段索引路由（`partitioned.py`）、合成语料与精确真值、跨四个量级的索引/端到端/过滤压力压测；Chinese-CLIP 真实商品向量（91,940 条 ABO 全量）同口径复核（[REAL_CLIP_BENCHMARK.md](docs/REAL_CLIP_BENCHMARK.md)）；**真实文本查询评测集（LLM 按商品事实生成中文查询 + 原商品真值）与分布 gap 量化——hit@10 0.668 vs 同分布切片 0.998，并据此确认瓶颈在表达对齐而非检索结构**（[QUERY_DISTRIBUTION.md](docs/QUERY_DISTRIBUTION.md)）
- M8 规模化（服务层）：会话外置（内存 / Redis）、Lua 原子 CAS 与冲突可见、存活/就绪探针分离、`/chat` 无状态会话接口
- M8 规模化（编码与版本层）：编码服务 HTTP 化（动态批处理 + 两级缓存）、索引版本目录与原子发布/回滚、增量更新与压实
- M9 检索质量：BM25 稀疏检索层（中文单字+二元组分词、postings 稀疏打分）、RRF 名次融合（免疫分数量纲差异）、重排层（精确向量零依赖降级 + cross-encoder，候选级入口 `search_candidates` 作为两阶段检索的正确性契约）、粗排候选窗口扫描（`--sweep-windows`，窗口是策略的一部分）、CLI `--retriever-backend rrf`；同一批 500 条真实文本查询上 hit@10 0.668 → 0.770 → 0.918，宽口径 0.962 → 0.986，cross-encoder 把候选集天花板的 97.0% 搬进 top-10（[HYBRID_RETRIEVAL.md](docs/HYBRID_RETRIEVAL.md)）

**进行中**

- 真实 LLM 对照实验与更大规模分层数据集
- VLM Planner 与细粒度 hard-negative 重排
- Agent 端到端 P50/P95 延迟基准的自动化产出
- 接入真实 LLM 决策器（实现 `Reasoner.decide()`），并用 LLM 步数分布校准当前基线
- MCP：streamable-http 并发容量实测
- 检索质量：cross-encoder 重排的收益/延迟权衡在 GPU 或独立排序服务上复测（CPU 上 17.5 s/查询、5 万次前向耗时 2 h 26 min），并按查询类型分档决定是否只对高价值查询启用
- 评测口径：多真值标注（category + 价格区间口径的等价商品集合）与跨语言查询集（英文查询对照），评测脚本已支持 `--queries-catalog` 换标注文件；报告已并列 `hit@k_category` 宽口径。检索层已把候选集榨到天花板的 97.0%，下一步口径修正的收益大于继续调索引结构
- 规模化：过滤下推与分段在真实向量上的复跑（真实向量 + 商品目录已就绪，见 [docs/REAL_CLIP_BENCHMARK.md](docs/REAL_CLIP_BENCHMARK.md)）
- 规模化：百万级语料与多副本并发的实测（需要 ≥16 GB 内存 / 多机）；分段的动态调整（按查询分布自动选分段边界，而不是按分位数静态切）

## 算力规划

### 负载模型

系统算力需求拆分为三类负载，资源相互隔离、独立扩缩：

| 负载 | 阶段 | 运行模式 | 资源特征 | 失败影响面 |
|---|---|---|---|---|
| 向量索引构建 | M2 | 离线批处理 | 计算密集，吞吐优先，可断点重跑 | 不影响在线服务 |
| Planner 在线推理 | M3 | 常驻服务 | 延迟敏感，长尾并发 | 可回落规则 Planner |
| VLM 后训练 | M3+ | 离线训练 | 显存与计算双密集 | 可跳过，不影响基线结论 |

### 服务等级目标

| 服务路径 | 指标 | 目标 | 当前状态 |
|---|---|---|---|
| 规则 Planner（默认路径，零模型依赖） | 端到端延迟 | P50 ≤ 1 ms / P95 ≤ 5 ms | 实测 0.07 / 0.41 ms ✅ |
| LLM Planner（增强路径） | 端到端延迟 | P50 ≤ 250 ms / P95 ≤ 400 ms | 实测 188 / 236 ms ✅ |
| 检索服务 | 可用性 | 检索后端可配置切换（baseline / clip / fusion），共享同一评测协议 | 已实现 |

> 延迟口径：商品目录与索引预加载后统计，不含进程启动与模型加载时间。

### 显存容量模型

显存预算按 `M ≈ W + A + K` 分解：`W` 为模型权重（参数量 × 位宽），`A` 为激活值（随 batch 与序列长度增长），`K` 为 KV Cache 或训练期框架开销。

| 阶段 | 模型 / 配置 | W | A + K | 显存预算 | 最低规格 |
|---|---|---|---|---|---|
| M0–M1 / M5 | 规则与词法基线，不加载模型 | — | — | — | 纯 CPU |
| M2 索引构建 | Chinese-CLIP ViT-B/16，fp16，输出 512 维向量 | ≈ 0.4 GB | batch 32 @ 224² 约 3 GB | ≥ 4 GB | ≥ 8 GB 单卡 |
| M3 在线推理 | Qwen2.5-3B-Instruct，bf16，`max-model-len 4096` | ≈ 6 GB | KV Cache（`gpu-memory-utilization 0.85`） | ≥ 8 GB | ≥ 16 GB 单卡 |
| M3+ 后训练 | 7B VLM，4-bit NF4 + LoRA + 梯度检查点 | ≈ 4 GB | 随 batch 与分辨率增长 | ≈ 12–20 GB | 24 GB 单卡 |

**并发容量估算（M3）**：Qwen2.5-3B 采用 GQA（2 个 KV 头 × 36 层，bf16），单 token KV 约 36 KB；按 4096 上下文计，单请求 KV 约 144 MB。24 GB 卡在扣除权重与激活后可支撑约 80 路满长度并发，实际短请求场景下并发余量更高。

> **口径说明**：显存与并发为按模型结构、量化位宽与推理配置推算的**预算值**，用于机型选型；实际占用受 CUDA / PyTorch 版本、算子实现与请求长度分布影响，上线前应在目标环境压测校准。

### 降级与运维

- **路径降级**：规则 Planner 为默认路径，零模型调用、微秒级响应；LLM Planner 属增强路径，其异常不应影响核心链路可用性。
- **检索降级**：`RETRIEVER_BACKEND` 支持 baseline / clip / fusion / ann 四种后端可配置切换，GPU 不可用时回退词法通道，功能完整且精度差异可通过同一评测协议量化。ANN 路径另有三级降级：调大 `ef_search` / `nprobe`（热生效、不重建）→ 触碰交付上限时退化精确扫描 → faiss 整体不可用时回落 numpy 精确后端。
- **显存水位**：`gpu-memory-utilization 0.85` 预留 15% 余量覆盖显存碎片与激活峰值；规划监控项包括 GPU 利用率、KV Cache 水位、请求队列深度与 P95 延迟。
- **成本护栏**：推理实例按需启停；索引构建与训练任务错峰执行，离线负载不占用在线显存配额。

### 选型与扩容

- **默认档位**：单卡 24 GB，同时覆盖 M2 编码与 M3 推理 / QLoRA 的显存上界。
- **垂直扩容触发**：放大 batch、关闭梯度检查点，或 QLoRA 改全参数微调时，显存需求上探至 40–80 GB。
- **水平扩展优先**：M2 为离线批处理，商品规模 100 → 1000+ 时优先提升吞吐（更大 batch / 多卡并行编码），而非提高单卡显存。
- **资源解耦收益**：在线推理与离线索引构建不共享显存，任一侧扩容不影响另一侧的容量与延迟基线。

基线阶段（M0–M5）全部在本机 CPU 完成；GPU 用于缩短耗时与支撑更大模型，**不是跑通链路的前提**。

### 索引侧内存模型

检索侧的内存不取决于 GPU，而取决于**商品数 × 维度 × 副本数**，是一条更硬的账：

| 商品数 | 原始向量 | HNSW | IVF | IVF-PQ |
|---|---|---|---|---|
| 10 万 | 195 MB | 221 MB | 198 MB | **9.4 MB** |
| 100 万 | 1.91 GB | 2.16 GB | 1.94 GB | **92 MB** |
| 1000 万 | 19.1 GB | 21.6 GB | 19.4 GB | **0.92 GB** |

（512 维 float32；HNSW 额外存图结构、IVF 额外存桶号，所以都比原始向量略大。）

三条结论：

1. **先算这条账再谈副本数。** 多副本时每副本各持一份索引，`副本数 × 索引体积` 才是总内存。
   百万条 HNSW 下每副本 2.2 GB，10 个副本就是 22 GB——这时候要么加内存，要么换索引，要么改用「中心化检索服务 + 轻量副本」。
2. **IVF-PQ 是唯一能真正省内存的档位**，约 1/21。但它的 recall@10 会掉到 0.097，**必须配精确重排**才能用
   （取 500 候选重排可回到 0.9375）；直接用会静默漏掉大量本该返回的商品。
3. **索引可以共享，编码不该跟着复制。** 检索副本只是 CPU + 内存，扩起来便宜；编码是 GPU 侧瓶颈，
   扩检索副本只会让它排队更长。这两件事的扩容指标完全不同（QPS/延迟 vs GPU 利用率/队列深度）。

> **口径说明**：索引体积来自 `results/ann_scaling_benchmark.md` 的实测（faiss 序列化字节数），
> 百万 / 千万级是按线性外推；百万级以上的实际表现未在本机验证，上线前应在目标环境压测校准。

## 安全与复现

- `.env`、模型权重、生成索引和输出目录均不提交；
- 示例数据为自建小数据，仅用于测试；
- 后续真实数据会记录来源、许可证、版本和清洗脚本；
- 所有模型结论必须与规则/轻量基线对照，并记录随机种子和配置。

## License

MIT，详见 [LICENSE](LICENSE)。

