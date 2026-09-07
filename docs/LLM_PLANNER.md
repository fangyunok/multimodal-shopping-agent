# LLM Planner 配置与评测

## 设计

`ShoppingAgent` 接受统一 `Planner` 协议，默认使用无需网络的 `RulePlanner`。`LLMPlanner` 调用 OpenAI-compatible `/v1/chat/completions` 接口，要求模型按 `Plan` JSON Schema 返回 `intent`、`max_price` 和 `category`。工具执行仍经过本项目 Dispatcher，LLM 不直接产生价格或库存。

客户端使用 Python 标准库，不绑定特定云厂商 SDK。请求固定 `temperature=0`，并拒绝以下输出：

- 非 JSON 或缺少响应字段；
- Schema 以外的字段、负数预算或未知 intent；
- 商品目录中不存在的类目；
- 与用户显式 intent 冲突的规划。

## 启用方式

不要把真实密钥写入仓库。复制 `.env.example` 的变量到当前终端或本机 `.env`：

```powershell
$env:PLANNER_BACKEND="llm"
$env:LLM_BASE_URL="http://127.0.0.1:8001"
$env:LLM_MODEL="Qwen/Qwen2.5-3B-Instruct"
$env:LLM_API_KEY=""
.venv\Scripts\uvicorn shopping_agent.app:app --port 8010
```

本地 vLLM 通常不需要 API Key；云端兼容接口应通过环境变量注入。CLI 可使用 `--planner-backend llm --llm-base-url ... --llm-model ...`。在真实模型服务启动前，自动化测试只使用注入的模拟 transport，不发起网络请求，也不产生费用。

模型服务启动后可一条命令运行公平对照：

```powershell
$env:LLM_API_KEY="仅在云接口需要时设置"
powershell -ExecutionPolicy Bypass -File scripts/run_planner_comparison.ps1 `
  -BaseUrl http://127.0.0.1:8001 `
  -Model Qwen/Qwen2.5-3B-Instruct
```

输出包含分项/联合准确率、非法输出率、按标签切片、失败案例、P50/P95 延迟和接口返回的 prompt/completion/total token 数。

## 公平对照

Rule 与 LLM Planner 必须使用同一锁定测试集。Prompt 只允许在开发集修改；正式测试比较工具选择、参数 exact match、非法规划率、任务成功率、平均步骤、P50/P95 延迟及 token/费用。模型或 Prompt 版本必须写入实验记录。
