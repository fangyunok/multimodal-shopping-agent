# 简历项目描述

## 推荐标题

多模态电商购物 Agent｜Chinese-CLIP、图文 RAG、Tool Calling、离线评测

## 三条简历表述

- 构建 CPU-first 多模态购物 Agent，基于 Chinese-CLIP 建立 512 维图文共享向量索引，并融合词法召回；通过 FastAPI 提供文字/图片检索、库存查询与商品对比工具，使用 Pydantic JSON Schema 校验工具参数并保留可审计调用轨迹。
- 基于 Amazon Berkeley Objects（CC BY 4.0）构建 100 商品真实图文子集与稳定 train/validation/test 划分，设计“其他视角查询图 vs 主图索引”的无泄漏评测；Chinese-CLIP 图文索引 Recall@1 从 RGB 基线 0.1667 提升至 0.7500（12 条 test 查询，bootstrap 95% CI [0.50, 1.00]）。
- 建立检索、工具与 Agent 分层评测，覆盖 Recall@K、MRR、nDCG、工具选择/参数准确率、非法调用率、任务成功率、引用忠实度和 P50/P95 延迟；使用 Docker、GitHub Actions 和一键 CPU 脚本保证测试与演示可复现。

## 面试时必须主动说明

- 0.75 来自 12 条跨视角测试查询，样本量较小，不能声称达到生产水平；
- 文本属性困难集上，Chinese-CLIP 没有全面超过词法基线，说明精确品牌/材质词仍需稀疏召回；
- 已实现可切换的规则/LLM Planner 和 Schema 约束，但在真实模型实验完成前，不应描述成已证明 LLM 自主规划效果；
- ABO 不含可靠实时价格和库存，因此检索实验与业务工具评测使用不同数据口径。

## 一句话介绍

我没有只做一个能聊天的购物页面，而是把多模态检索、受 Schema 约束的工具调用、证据引用和可复现实验串成了完整闭环，并专门设计跨视角测试避免原图检索的数据泄漏。
