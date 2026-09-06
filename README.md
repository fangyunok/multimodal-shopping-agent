# 多模态电商购物 Agent

一个面向求职作品集的、可复现的电商智能体项目：图文混合检索（RAG）、商品搜索/库存/对比工具调用，以及端到端离线评测。

## 当前阶段：CPU MVP

本阶段不依赖显卡或外部 API，确保任何人克隆仓库后都能复现：

- 商品文本检索与价格、类目约束；
- 基于 RGB 直方图的轻量图片相似度基线；
- 文本与图片分数融合；
- 搜索、库存检查、商品对比工具；
- 可审计的 Agent 工具调用轨迹；
- 从自然语言抽取预算、类目和比较意图的规则规划基线；
- 可直接提供给 LLM Function Calling 的工具名称、描述和 JSON Schema；
- 检索命中率、工具选择准确率、约束满足率评测；
- FastAPI 服务和自动化测试。
- JPEG、PNG、WebP 图片上传接口（5 MB 限制与内容校验）。

轻量图片特征只是 CPU 冒烟基线，不把颜色相似误称为语义理解。下一阶段将接入中文 CLIP，实现真正的以文搜图、以图搜图和图文联合检索；再接入视觉语言模型完成商品属性理解。

项目已经提供可选的中文 CLIP 检索后端。首次启用会下载模型，本机 CPU 可以推理但较慢：

```bash
.venv\\Scripts\\python -m pip install -e ".[clip,dev]"
$env:RETRIEVER_BACKEND="clip"
$env:MODEL_DEVICE="cpu"
.venv\\Scripts\\uvicorn shopping_agent.app:app
```

默认 `RETRIEVER_BACKEND=baseline`，自动化测试不下载模型。

真实数据准备完成后，离线构建并复用商品向量：

```bash
.venv\\Scripts\\shopping-agent \
  --catalog data/products.jsonl \
  --build-index data/index/chinese-clip-base \
  --device cpu
$env:INDEX_DIR="data/index/chinese-clip-base"
```

`manifest.json` 会记录商品目录 SHA-256、模型名、向量维度和生成时间。商品目录发生变化或模型不一致时，服务拒绝加载旧索引。

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

ABO 不含可靠价格和实时库存，只用于图文检索；数据详情和署名要求见 `docs/ABO_DATASET.md`。

检索评测统一输出 Recall@1/5/10 与 MRR：

```bash
.venv\\Scripts\\shopping-agent --catalog data/processed/products.jsonl --evaluate-retrieval text
.venv\\Scripts\\shopping-agent --catalog data/processed/products.jsonl --evaluate-retrieval image
```

标题或原图检索自身只作为链路检查，不能代表语义检索效果。

生成不复制商品标题的属性需求评测集，并运行统一指标：

```bash
.venv\\Scripts\\shopping-agent --catalog data/processed/products.jsonl --build-benchmark outputs/retrieval_benchmark.jsonl
.venv\\Scripts\\shopping-agent --catalog data/processed/products.jsonl --benchmark outputs/retrieval_benchmark.jsonl
```

自动生成的查询必须经过人工抽查，最终测试集应补充同义改写、属性组合、类目歧义和不同视角图片。

## 快速开始

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

## 工程结构

```text
data/                         示例商品与评测集
src/shopping_agent/
  retrieval.py               CPU 图文混合检索
  tools.py                   可调用购物工具
  agent.py                   可解释的决策与工具编排
  evaluation.py              离线评测
  app.py                     FastAPI 服务
tests/                        单元与接口测试
```

## 路线图

- [x] M0：仓库、数据契约、测试与 API 骨架
- [x] M1：CPU 图文检索基线、工具调用与评测闭环
- [ ] M2：中文 CLIP 向量化、持久化索引、真实商品图文数据管线（均已接入，待选数据集）
- [ ] M3：视觉语言模型属性抽取和基于 schema 的 LLM 工具调用
- [ ] M4：RAG 忠实度、工具参数、任务成功率、延迟与消融实验
- [ ] M5：Gradio 演示、Docker、CI 和完整实验报告

## GPU 规划

M0-M1 只使用本机 CPU。M2 的向量离线编码可先租单卡 RTX 4090（24GB），性价比通常最好；M3 若采用 7B 视觉语言模型的 QLoRA，优先选单卡 RTX 4090 24GB，只有在出现显存不足时再升到 A800 40GB/80GB。实际租用前需依据当时 AutoDL 的地区库存与每小时价格再决定。

## 安全与复现

- `.env`、模型权重、生成索引和输出目录均不提交；
- 示例数据为自建小数据，仅用于测试；
- 后续真实数据会记录来源、许可证、版本和清洗脚本；
- 所有模型结论必须与规则/轻量基线对照，并记录随机种子和配置。
