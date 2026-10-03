# MCP Server：接口协议与容量实测

本项目除 HTTP 服务与 CLI 之外，还把同一套能力按 **Model Context Protocol（MCP）** 暴露出去，
使 Claude Desktop、Cursor 等任意 MCP 客户端可以直接接入检索、比价、库存与端到端追问能力。

一句话定位：**业务逻辑一行没改，只是把已有的工具协议层（`ToolDefinition`）换了一个标准传输暴露出去**。
`tools.py` 里的 `ToolDefinition(name, description, parameters=JSON Schema)` 三个字段与 MCP 的
tool 规格天然同构，因此这一层的实现重点是协议完整性、错误语义与运行期韧性，而不是重写业务。

- 实现：`src/shopping_agent/mcp_server.py`
- 依赖：`mcp>=2.3,<3`（2.x 中 `FastMCP` 已更名为 `MCPServer`，本实现按 2.x API 编写）
- 压测脚本：`scripts/benchmark_mcp.py`

## 一、暴露的能力

### Tools（5 个）

| 工具 | 入参 | 语义 | 幂等 |
|---|---|---|---|
| `search_products` | `query`、`max_price`、`category`、`top_k`、`image_path` | 文本/图片检索并施加预算与类目约束 | ✅ 只读 |
| `compare_products` | `product_ids`（2–10 个） | 按 ID 对比价格、评分与库存 | ✅ 只读 |
| `check_inventory` | `product_id` | 查询库存可用性 | ✅ 只读 |
| `ask_shopping_agent` | `query`、`intent`、`max_price`、`category`、`top_k`、`product_ids` | 端到端追问，返回带商品 ID 引用的回答与完整调用轨迹 | ✅ 只读 |
| `get_runtime_metrics` | — | 返回逐工具调用量、错误、重试、幂等命中与 P50/P95/P99 延迟 | ✅ 只读 |

工具入参由函数签名生成扁平 JSON Schema，可直接用于任意支持 Function Calling 的模型；
业务侧再用 Pydantic 做第二层校验（`top_k ≤ 20`、`product_ids` 数量下限等）。

### Resources（2 静态 + 1 模板）

| URI | 内容 |
|---|---|
| `catalog://products` | 商品目录全量快照 |
| `catalog://schema` | 商品数据契约（`Product` 的 JSON Schema），供客户端校验与生成表单 |
| `catalog://products/{product_id}` | 单个商品的结构化事实；不存在时返回结构化 `not_found` 而非抛错 |

### Prompts（2 个）

| 提示词 | 作用 |
|---|---|
| `grounded_recommendation` | 生成受预算与类目约束、**必须引用商品 ID**、价格与库存只能来自工具返回的推荐模板 |
| `tool_orchestration_plan` | 针对一个目标产出工具编排计划：调用顺序、入参来源、失败回退路径与终止条件 |

## 二、传输与接入

支持三种传输形态：

```bash
# stdio（MCP 客户端默认链路，Claude Desktop / Cursor 用这个）
python -m shopping_agent.mcp_server --transport stdio --catalog data/products.jsonl

# SSE（长连接、服务端推送）
python -m shopping_agent.mcp_server --transport sse --host 127.0.0.1 --port 8770

# Streamable HTTP（多连接并发，适合高扇出场景）
python -m shopping_agent.mcp_server --transport streamable-http --host 127.0.0.1 --port 8770
```

安装包后也可直接用入口命令：`shopping-agent-mcp --transport stdio`。

### 接入 Claude Desktop / Cursor

在客户端的 MCP 配置文件中加入（Windows 示例，路径按本机实际位置替换）：

```json
{
  "mcpServers": {
    "multimodal-shopping-agent": {
      "command": "D:/multimodal-shopping-agent/.venv/Scripts/python.exe",
      "args": [
        "-m", "shopping_agent.mcp_server",
        "--transport", "stdio",
        "--catalog", "D:/multimodal-shopping-agent/data/products.jsonl"
      ]
    }
  }
}
```

## 三、工程层设计

四类横切能力集中实现在 `ToolRuntime`，业务代码（`tools.py` / `agent.py`）完全不需要感知：

| 能力 | 实现 | 取舍说明 |
|---|---|---|
| **并发闸门** | `asyncio.Semaphore(max_concurrency)` | 上限默认 8，避免下游检索被打满；同步 handler 通过 `asyncio.to_thread` 卸载到线程池，因此超时可达、且并发请求能真正并行 |
| **超时** | `asyncio.wait_for(timeout_seconds)` | 默认 2s。CPU 密集 handler 被取消后其线程仍会跑完 —— 超时保护的是调用方，不保证回收计算 |
| **重试** | 指数退避 + 抖动，默认 1 次 | 只对瞬时故障重试。`PermissionError` 虽然是 `OSError` 子类但被显式排除：重试不会改变权限判定结果 |
| **幂等重放** | `sha256(tool + 规范化入参)` 作键，TTL 30s | 抑制客户端重试造成的重复计算。只缓存成功结果；TTL 设为 0 可关闭 |
| **指标** | 线程安全计数 + 最近秩法分位数 | 通过 `get_runtime_metrics` 暴露，便于客户端或运维侧拉取 |

失败一律转换为结构化错误，不把异常直接抛给模型；错误被分类，客户端可据此决定重试还是终止：

| 类型 | 触发场景 |
|---|---|
| `invalid_arguments` | 入参越界或数量不足（如 `top_k=999`、`product_ids` 少于 2 个） |
| `permission_denied` | 访问被策略禁止的资源（默认禁止本地图片路径） |
| `not_found` | 商品 ID 不存在 |
| `timeout` | 超过 `timeout_seconds` 未返回 |
| `tool_failure` | 其他未预期异常 |

## 四、容量实测

脚本：`scripts/benchmark_mcp.py`；原始结果（含完整 JSON）见
[`results/mcp_benchmark_inprocess.md`](../results/mcp_benchmark_inprocess.md) 与
[`results/mcp_benchmark_stdio.md`](../results/mcp_benchmark_stdio.md)。
环境：Windows，Python 3.11.9，商品目录 5 条，每个并发档重复多轮取吞吐最高的一轮
（best-of-N，用于抑制调度噪声）。计时口径为「客户端发起 → 编解码 → 参数校验 → 业务处理 → 响应返回」。

### 传输层对照（工具 `search_products`，每档 200 次请求）

| 传输 | 并发 1 吞吐 | 并发 8 吞吐 | 并发 1 P50 | 并发 8 P50 | 结论 |
|---|---|---|---|---|---|
| 进程内回环 | 1121.4 req/s | 1207.1 req/s | 0.76 ms | 3.67 ms | 协议 + 处理成本上限 |
| stdio 子进程 | 246.5 req/s | 365.2 req/s | 3.89 ms | 21.84 ms | 生产客户端实际链路 |

### 三条实测结论

1. **两种传输的吞吐都不随并发增长**，分别在约 1.2k req/s 与约 365 req/s 饱和，而 P50 随并发近似线性上升。
   → 瓶颈在单条串行的请求/响应管线，**不在检索逻辑**；在 5 条商品的目录上，业务处理耗时可忽略。
2. **stdio 相对进程内回环多出约 3.1 ms 固定开销**（并发 1 时 P50 由 0.76 ms 升至 3.89 ms），吞吐下降约 3.3 倍。
   这是进程间管道与流式编解码的成本，属于协议本身的物理开销，不随调参消失。
3. **幂等缓存对本场景吞吐几乎无贡献**（场景 A 与场景 B 同档位基本持平）。原因是首个请求写入缓存之前，
   并发请求已经发出。该机制的价值是**正确性层面的**——抑制客户端重试导致的重复计算，而不是提速。

### 因此优化方向在传输层

- 需要高扇出或服务端多客户端并发时，改用 **streamable-http**，让并发落到独立连接上，而不是在 stdio 单管道里排队；
- 单次交互想降低往返次数，可通过 `tool_orchestration_plan` 提示词让模型一次规划、批量取数，而不是逐轮试探；
- 若检索后端换成 CLIP 或更大目录，业务耗时会上升，届时瓶颈才会从传输层转移到计算层 —— 届时先扩 `max_concurrency`，再考虑批量编码。

## 五、安全边界

- **默认拒绝本地图片路径**：`search_products` 的 `image_path` 在默认配置下直接返回 `permission_denied`，
  需显式设置 `MCP_ALLOW_LOCAL_IMAGE_PATH=1` 或传 `--allow-local-image-path` 才放行。
  这一条与 HTTP 服务「不接受本地路径、只接受上传」的约定保持一致。
- 商品价格与库存只从工具返回，提示词中已明确禁止模型凭记忆生成。
- 服务不读取 `.env`、模型权重与索引目录之外的内容；密钥仅从环境变量读取。

## 六、已知限制

- 幂等缓存是**进程内、无持久化**的，多实例部署时无法跨实例去重；
- 超时无法中断已经进入执行阶段的 CPU 密集任务（Python 线程不可取消），只保证调用方及时拿到错误；
- `ask_shopping_agent` 目前是单轮规划（Planner → Tool → 回答），**没有多轮会话记忆与 ReAct 循环**，
  因此不支持「那第二个呢」这类依赖上下文的追问；
- 压测在本地单机完成，未覆盖真实网络 RTT 与多客户端场景；`streamable-http` 的并发容量尚未实测。
