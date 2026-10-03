# AutoDL 上运行 LLM Planner 对照实验

## 实例配置

| 配置项 | 取值 | 依据 |
|---|---|---|
| 算力 | 单卡 24 GB 显存（如 RTX 3090 / 4090），按量计费 | 覆盖 3B 模型 bf16 权重与 KV Cache，预算依据见 README「算力规划」 |
| 基础镜像 | PyTorch 2.x、Python ≥ 3.10、CUDA 12.x | vLLM 运行要求 |
| 存储 | 默认 50 GB 数据盘 | 模型权重与评测输出规模，无需额外扩容 |
| 推理模型 | `Qwen/Qwen2.5-3B-Instruct`，`--max-model-len 4096`、`--gpu-memory-utilization 0.85` | 与 README 中 100 条锁定集的评测口径一致 |

**关于模型规模**：3B 足以验证受 Schema 约束的 Planner 是否显著优于规则基线，且权重下载与加载开销低于 7B；若 3B 在锁定集上表现明显不足，再按同一评测协议补跑 7B 作为对照。

**成本控制**：一个完整实验单元为「20 条开发集 + 100 条锁定测试集」。云平台按**实例开关机时长**计费，而非 GPU 是否正在计算；脚本结束后应在控制台主动关机或预先设置定时关机。

## 用户操作

在 AutoDL 控制台创建上述实例，打开 JupyterLab 终端并执行：

```bash
git clone https://github.com/fangyunok/multimodal-shopping-agent.git
cd multimodal-shopping-agent
bash scripts/autodl_run_planner.sh
```

脚本会创建隔离环境、安装 vLLM、启动 OpenAI-compatible 服务、运行 Rule 基线、20 条 LLM 开发集和 100 条锁定测试集。结果保存在 `outputs/planner/`。

AutoDL 默认软件源缺少 `uv` 时，脚本仅对该安装步骤使用官方 PyPI；模型下载可通过 `HF_ENDPOINT` 指向可访问的 Hugging Face 镜像。

取回 `rule-test.json`、`llm-dev.json`、`llm-test.json` 后，在控制台关闭实例。脚本会停止 vLLM，但无法代替用户关闭云实例。

不要把 AutoDL Token、密码或其他密钥写入仓库，也不要把它们发到聊天中。
