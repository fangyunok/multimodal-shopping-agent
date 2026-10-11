# 项目技术摘要

## 项目名称

多模态内容理解与工具编排 Agent｜Chinese-CLIP 图文检索、Prompt 编排、Tool Calling、离线评测

## 核心实现

- 构建 CPU-first 多模态购物 Agent，基于 Chinese-CLIP 建立 512 维图文共享向量索引，并融合词法召回；通过 FastAPI 提供文字/图片检索、库存查询与商品对比工具，使用 Pydantic JSON Schema 校验工具参数并保留可审计调用轨迹。
- 基于 Amazon Berkeley Objects（CC BY 4.0）构建 100 商品真实图文子集与稳定 train/validation/test 划分，设计“其他视角查询图 vs 主图索引”的无泄漏评测；Chinese-CLIP 图文索引 Recall@1 从 RGB 基线 0.1667 提升至 0.7500（12 条 test 查询，bootstrap 95% CI [0.50, 1.00]）。
- 建立检索、工具与 Agent 分层评测，覆盖 Recall@K、MRR、nDCG、工具选择/参数准确率、非法调用率、任务成功率、引用忠实度和 P50/P95 延迟；使用 Docker、GitHub Actions 和一键 CPU 脚本保证测试与演示可复现。
- 接入 Qwen2.5-3B 与 vLLM 的 Schema 约束 Planner，在 100 条锁定人工集上将联合准确率由规则基线 0.36 提升至 0.51；同时给出预算提取与显式 intent 遵循的误差归因，以及约 188 ms P50 延迟。

## 设计边界

- Recall@1 0.75 取自 12 条跨视角测试查询，配套 bootstrap 95% CI；
- 属性文本检索中词法信号与 Chinese-CLIP 向量互补：精确品牌/材质词走稀疏召回，语义改写走向量召回；
- Qwen2.5-3B 在联合准确率上优于规则基线（0.36→0.51）；预算切片与显式 intent 遵循的进一步优化列入 Roadmap；
- ABO 用于图文检索评测；价格与库存类业务工具评测使用独立的商品目录口径。

## 一句话介绍

我没有只做一个能聊天的购物页面，而是把多模态检索、受 Schema 约束的工具调用、证据引用和可复现实验串成了完整闭环，并专门设计跨视角测试避免原图检索的数据泄漏。
