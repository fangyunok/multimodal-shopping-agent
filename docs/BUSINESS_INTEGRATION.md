# 业务接入：库存、调用方身份与大模型多轮决策

HTTP 服务已支持三项独立开关。默认仍为可离线运行的演示模式：规则决策、商品文件库存、无鉴权。
生产接入必须显式配置下面的服务地址和凭据。仓库不包含任何业务系统地址或真实密钥。

| 能力 | 配置 | 生效范围 |
| --- | --- | --- |
| 外部库存 | `INVENTORY_BACKEND=http` | HTTP `/agent`、`/agent/image`、`/chat` 的库存检查与商品对比 |
| 多步模型决策 | `REASONER_BACKEND=llm` | HTTP `/chat`；每步读取工具结果再选择下一动作 |
| 调用方鉴权 | `AUTH_MODE=api_key` | HTTP 业务接口、接口文档与状态查询；首页和两种健康探针公开 |

CLI 与 MCP 默认保持目录库存和规则决策；不能把 HTTP 的鉴权、会话隔离或外部库存能力推断为 MCP 已具备。
`PLANNER_BACKEND=llm` 仍控制单轮入口的需求提取，和 `/chat` 的 `REASONER_BACKEND` 分开配置。

## 1. 库存契约

设置 `INVENTORY_BACKEND=http` 与 `INVENTORY_BASE_URL`。服务端请求：

```http
GET /inventory?product_id=shoe-001
Authorization: Bearer <INVENTORY_API_KEY>
```

响应必须为以下 JSON 对象，`stock` 是非负整数，`observed_at` 是带时区的实际查询时间：

```json
{"product_id":"shoe-001","stock":8,"observed_at":"2026-10-09T13:00:00Z"}
```

示例时间只说明格式；服务必须返回查询时的时间。该字段表示上游何时确认这个库存值，不能用商品最后编辑时间代替。
适配自己的 ERP/WMS 时，可在业务网关实现这个契约。它是只读查询，不预占、不扣减库存、不创建订单。

| 配置 | 默认 | 作用 |
| --- | --- | --- |
| `INVENTORY_BACKEND` | `catalog` | `catalog` 或 `http` |
| `INVENTORY_BASE_URL` | 空 | 库存服务根地址；可包含固定路径前缀 |
| `INVENTORY_API_KEY` | 空 | 上游 Bearer 凭据，可按服务要求设置 |
| `INVENTORY_TIMEOUT_SECONDS` | `3` | 单次 HTTP 超时 |
| `INVENTORY_MAX_AGE_SECONDS` | `30` | 允许的库存响应最大年龄；未来偏差最多 5 秒 |

服务不会缓存外部库存，每次追问重新查询。商品 ID 不符、时间过旧、数量类型错误、超时、HTTP 错误和重定向均拒绝。
失败不会悄悄使用目录库存，也不会把失败解释为 0 件：单轮接口返回 503，多轮接口将错误交给决策循环并说明库存未知。
部分候选查询失败时，回答只对已确认的候选标注有货/无货，其余标注库存未知。

库存工具返回 `source` 与 `observed_at`。单轮响应的 `inventory_source`、多轮响应的
`context.inventory_source` 标识 `http` 或 `catalog_demo`；网页也区分这两种来源。
商品价格目前仍来自目录，外部库存查询不等于实时价格、下单承诺或库存锁定。

## 2. 调用方鉴权与会话归属

```bash
export AUTH_MODE=api_key
export API_KEYS_JSON="$(python -c 'import json,secrets; print(json.dumps({"shop-web":secrets.token_urlsafe(32)}))')"
docker compose up --build -d --scale shopping-agent=3
```

将生成的密钥通过部署平台的秘密配置分发给调用方。配置对象的键是稳定调用方名称，值是至少 32 个 ASCII 字符的独立密钥。
不要把密钥提交到仓库。HTTP 请求使用 `Authorization: Bearer <token>`；对外访问由入口网关终止 TLS。
生产应用应由自己的后端持有服务密钥，不应把共享服务密钥打包到公开网页中。仓库演示页的令牌输入仅适合受控测试，页面不持久化令牌。

未认证/错误凭据返回 401；启用鉴权却缺失、重复或非法的配置返回 503，`/readyz` 也报告不可用。
`/healthz`、`/readyz` 和首页公开，以便现有网关健康检查持续工作；`/docs`、`/openapi.json`、`/tools`、`/index` 等同样受鉴权保护。

会话存储键由「已认证调用方 + session_id」生成。A、B 两个调用方传相同 `session_id` 不会共享记忆；
轮换密钥但保持调用方名称时，会话归属保持不变。所有副本须使用一致的配置，配置变更后按部署方式滚动重启。
这提供的是服务调用方隔离，不是完整终端用户登录；同一调用方内的用户归属、权限和会话 ID 分配应由业务后端控制。

## 3. 大模型多轮决策

配置 `REASONER_BACKEND=llm`、`LLM_BASE_URL`、`LLM_MODEL`，按提供方要求设置 `LLM_API_KEY`。
模型服务须支持 Chat Completions 风格消息和 `response_format=json_schema`；URL 可以是服务根地址或以 `/v1` 结尾。
`LLM_TIMEOUT_SECONDS` 默认 10 秒，`LLM_MAX_INPUT_TOKENS` 默认 8192（启发式估算），单次生成最多 512 tokens。
网络响应上限 256 KiB，禁止跟随重定向，不自动重试模型请求。

模型每步只能输出一个已登记工具动作，或 `finish` 加已确认的商品 ID。执行器负责：

- 将会话预算、类目和排除项施加到检索，并限制每轮候选最多 3 个；本地图片路径不允许由模型指定。
- 验证工具名称和参数 Schema，拒绝不符合约束的库存、对比目标。
- 检查最终推荐的商品确实来自本轮检索且已成功确认有货。模型不能直接生成价格/库存事实。
- 用工具结果组装回答及商品 ID 引用；库存追问和商品对比也有引用。
- 延续最大步数、重复工具调用检测和会话历史预算；完整模型请求另有输入预算校验。

无效 JSON、未登记动作、不可靠的最终推荐、网络错误或模型输入超限，会在本轮切换为规则决策器；
`context.fallback_used=true` 明确披露降级，同一轮不会反复请求故障模型。`context.reasoner_backend=llm` 表示配置的决策后端。
服务不会把原始上游错误响应、地址或密钥放入用户回答。

已确认槽位仍先由规则规划器提取，模型可利用压缩历史和工具观察补充检索描述。复杂偏好理解的效果必须用真实模型另行评测，
不能从契约测试的通过率推断。已有单次网络超时与步数上限，不提供强制终止本地运算的端到端截止时间；部署时要协调客户端和网关超时。

## 4. 验证范围

```bash
pip install -e '.[dev,ann,redis]'
python -m pytest tests/test_business_runtime.py
```

测试启动本地 HTTP 服务，走真实请求、响应、凭据传递与 Schema 校验，覆盖库存变化/过期/错商品/超时/重定向，
模型多步检索与库存追问、故障降级、约束校验、非法输出、调用方隔离及密钥轮换。它们不调用真实大模型或真实业务库存系统。
CI 另外生成临时调用方凭据并启动三个 Docker 副本，验证未认证拒绝、跨副本会话连续性、跨调用方隔离以及 Redis 故障恢复。

接入真实库存服务后仍需进行合同一致性测试；接入真实模型后仍需测任务成功率、约束遵守率、延迟和费用。
