# 多模态电商购物 Agent

一个面向求职作品集的、可复现的电商智能体项目：图文混合检索（RAG）、商品搜索/库存/对比工具调用，以及端到端离线评测。

## 当前阶段：CPU MVP

本阶段不依赖显卡或外部 API，确保任何人克隆仓库后都能复现：

- 商品文本检索与价格、类目约束；
- 基于 RGB 直方图的轻量图片相似度基线；
- 文本与图片分数融合；
- 搜索、库存检查、商品对比工具；
- 可审计的 Agent 工具调用轨迹；
- 检索命中率、工具选择准确率、约束满足率评测；
- FastAPI 服务和自动化测试。

轻量图片特征只是 CPU 冒烟基线，不把颜色相似误称为语义理解。下一阶段将接入中文 CLIP，实现真正的以文搜图、以图搜图和图文联合检索；再接入视觉语言模型完成商品属性理解。

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

服务启动后访问 `http://127.0.0.1:8000/docs`，可以直接调用 `/search` 和 `/agent`。

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
- [ ] M2：中文 CLIP 向量化、FAISS 索引、真实商品图文数据
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

