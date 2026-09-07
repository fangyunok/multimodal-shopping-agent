# AutoDL 上运行 LLM Planner 对照实验

## 推荐实例

- 按量计费，单卡 RTX 3090 24GB；
- PyTorch 2.x、Python 3.10 或以上、CUDA 12.x 基础镜像；
- 使用默认 50GB 数据盘，不购买额外扩容；
- 模型为 `Qwen/Qwen2.5-3B-Instruct`；
- 总预算 30 元，建议先充值 10 元并设置 6 小时定时关机。

3B 模型足以验证结构化 Planner 是否明显优于规则基线，下载和加载成本也低于 7B。只有 3B 在锁定集上表现明显不足时，才在剩余预算内补跑 7B。

## 用户操作

在 AutoDL 控制台创建上述实例，打开 JupyterLab 终端并执行：

```bash
git clone https://github.com/fangyunok/multimodal-shopping-agent.git
cd multimodal-shopping-agent
bash scripts/autodl_run_planner.sh
```

脚本会创建隔离环境、安装 vLLM、启动 OpenAI-compatible 服务、运行 Rule 基线、20 条 LLM 开发集和 100 条锁定测试集。结果保存在 `outputs/planner/`。

AutoDL 默认软件源缺少 `uv` 时，脚本仅对该安装步骤使用官方 PyPI；模型下载可通过 `HF_ENDPOINT` 指向可访问的 Hugging Face 镜像。

下载 `rule-test.json`、`llm-dev.json` 和 `llm-test.json` 后，立刻在 AutoDL 控制台点击“关机”。脚本会停止 vLLM，但无法代替用户关闭云实例。计费以实例开关机时间为准，而不是 GPU 是否正在运算。

不要把 AutoDL Token、密码或其他密钥写入仓库，也不要把它们发到聊天中。
