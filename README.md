# 多模态内容理解与工具编排 Agent

[![CI](https://github.com/fangyunok/multimodal-shopping-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/fangyunok/multimodal-shopping-agent/actions/workflows/ci.yml)

**面向图文内容的可复现多模态 Agent：内容理解 → 检索 → 工具编排 → 可溯源生成。**

把**多模态内容理解（Chinese-CLIP 共享向量空间）、受 JSON Schema 约束的工具编排、可溯源生成与分层离线评测**连成完整链路；检索、规划、工具、回答四层解耦，可独立替换与评测。项目不只展示成功案例，也保留稠密检索退化、Planner 非法输出等负结果。

## 先看结论

| 子系统 | 对照结果 | 结论与边界 |
|---|---|---|
| 多模态内容理解与检索（图文共享向量空间） | Chinese-CLIP 图文索引 Recall@1 `0.1667 → 0.7500` | 100 商品、12 条跨视角 test 查询；95% CI `[0.50, 1.00]`，不可外推到完整商品分布 |
| 属性文本检索与融合 | CLIP 没有全面超过词法基线 | 品牌、型号等精确 token 仍需词法信号，因此保留融合检索 |
| Prompt 与规划编排 | Qwen2.5-3B 联合准确率 `0.36 → 0.51`、类目准确率 `0.49 → 0.87` | 同一 100 条锁定集；同时产生 10% 无效规划，预算提取仍是短板 |
| 工具编排与可溯源生成 | Schema Dispatcher + 内容 ID 引用 | 模型不能直接编造价格和库存；非法参数可拒绝、引用忠实度可度量 |
| 在线性能 | 规则 Agent 端到端 P50 `0.07 ms` / P95 `0.41 ms`；LLM Planner P50 `188 ms` / P95 `236 ms` | 计时在商品目录与索引预加载后统计，不含进程启动与模型加载 |
| MCP 协议链路 | tools / resources / prompts 三类能力完整暴露，支持 stdio、SSE 与 streamable-http | stdio 吞吐约 `360 req/s` 饱和且不随并发增长，瓶颈在串行管线而非检索；口径见 [MCP 文档](docs/MCP_SERVER.md) |
| 多轮会话与循环控制 | 会话槽位跨轮继承；上下文预算为硬边界；ReAct 轨迹逐步可审计 | 追问「有货吗」直接复用上一轮候选，工具调用 3 次 → 1 次；40 轮会话在 512–4096 tokens 四档预算下峰值均未越界；口径见[运行期文档](docs/AGENT_RUNTIME.md) |
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

项目文档：[技术摘要](docs/PROJECT_SUMMARY.md)｜[设计决策与验证指南](docs/TECHNICAL_GUIDE.md)｜[完整实验记录](docs/EXPERIMENT_LOG.md)｜[MCP 接口与容量实测](docs/MCP_SERVER.md)｜[会话记忆与 ReAct 循环](docs/AGENT_RUNTIME.md)

Planner 默认使用可复现的规则基线，也可切换到 OpenAI-compatible LLM 后端；配置与公平评测方法见 [LLM Planner 文档](docs/LLM_PLANNER.md)，GPU 复现实验见 [AutoDL 指南](docs/AUTODL_PLANNER.md)。

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

**边界**：`RuleReasoner` 是确定性基线，不含真实 LLM 决策器（其步数与延迟是该接口的上界参考）；
会话在内存中，进程重启即丢失，多实例部署需要外部存储；高并发下的会话行为未做压测。
设计取舍与完整口径见 [运行期文档](docs/AGENT_RUNTIME.md)。

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
  fusion_retrieval.py        词法 + 向量分数融合
  tools.py                   可调用工具（搜索 / 库存 / 对比）
  agent.py                   可解释的决策与工具编排
  planner.py / llm_planner.py  规则规划基线 / LLM 规划后端
  mcp_server.py              MCP 协议适配层：tools / resources / prompts + 韧性层与指标
  session.py                 会话记忆、跨轮槽位累积与上下文预算
  react.py                   ReAct 循环：多步轨迹、观察回注、循环检测与预算感知
  evaluation.py              离线评测
  app.py                     FastAPI 服务
scripts/benchmark_mcp.py      MCP 并发压测（进程内回环 / stdio 子进程两种口径）
scripts/benchmark_agent_runtime.py  会话上下文预算与 ReAct 步数 / 延迟实测
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

**进行中**

- 真实 LLM 对照实验与更大规模分层数据集
- VLM Planner 与细粒度 hard-negative 重排
- Agent 端到端 P50/P95 延迟基准的自动化产出
- 接入真实 LLM 决策器（实现 `Reasoner.decide()`），并用 LLM 步数分布校准当前基线
- MCP：streamable-http 并发容量实测；会话状态外部化（Redis）以支持多实例部署

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
- **检索降级**：`RETRIEVER_BACKEND` 支持 baseline / clip / fusion 三种后端可配置切换，GPU 不可用时回退词法通道，功能完整且精度差异可通过同一评测协议量化。
- **显存水位**：`gpu-memory-utilization 0.85` 预留 15% 余量覆盖显存碎片与激活峰值；规划监控项包括 GPU 利用率、KV Cache 水位、请求队列深度与 P95 延迟。
- **成本护栏**：推理实例按需启停；索引构建与训练任务错峰执行，离线负载不占用在线显存配额。

### 选型与扩容

- **默认档位**：单卡 24 GB，同时覆盖 M2 编码与 M3 推理 / QLoRA 的显存上界。
- **垂直扩容触发**：放大 batch、关闭梯度检查点，或 QLoRA 改全参数微调时，显存需求上探至 40–80 GB。
- **水平扩展优先**：M2 为离线批处理，商品规模 100 → 1000+ 时优先提升吞吐（更大 batch / 多卡并行编码），而非提高单卡显存。
- **资源解耦收益**：在线推理与离线索引构建不共享显存，任一侧扩容不影响另一侧的容量与延迟基线。

基线阶段（M0–M5）全部在本机 CPU 完成；GPU 用于缩短耗时与支撑更大模型，**不是跑通链路的前提**。

## 安全与复现

- `.env`、模型权重、生成索引和输出目录均不提交；
- 示例数据为自建小数据，仅用于测试；
- 后续真实数据会记录来源、许可证、版本和清洗脚本；
- 所有模型结论必须与规则/轻量基线对照，并记录随机种子和配置。

## License

MIT，详见 [LICENSE](LICENSE)。
