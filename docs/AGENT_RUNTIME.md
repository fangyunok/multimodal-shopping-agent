# 会话记忆、上下文预算与 ReAct 循环

这份文档说明本项目**多轮**能力的三块实现：状态怎么跨轮累积、上下文怎么不爆、一次请求怎么
展开成多步工具调用。三块都在 `src/shopping_agent/` 下，彼此解耦，可以单独替换。

| 模块 | 文件 | 职责 |
|---|---|---|
| 会话记忆 | `session.py` | 跨轮累积槽位、沉淀事件流 |
| 上下文预算 | `session.py` | 在 token 预算内把事件流压成 messages |
| ReAct 循环 | `react.py` | Thought → Action → Observation 的显式多步执行 |

单轮 Agent（`agent.py`）保持不变：它仍是确定性规划基线，多轮能力是叠加在它之上的一层。

## 1. 会话记忆：新的覆盖旧的，没提的继承

多轮对话里最容易出错的不是"记不住"，而是**记错**——把上一轮的预算套用到这一轮，
或者用户改口后还守着旧约束。这里的规则只有两条：

1. **新值覆盖旧值**：用户这一轮明确说了预算或类目，就覆盖。
2. **未提及的继承**：这一轮没说，就沿用上一轮已确认的值。

`SessionState` 累积五类槽位：

| 槽位 | 来源 | 用途 |
|---|---|---|
| `max_price` | Planner 解析 | 检索时作为预算约束 |
| `category` | Planner 解析 | 检索时作为类目约束 |
| `last_candidates` | 上一次检索返回 | 「有货吗」「对比一下」直接复用，不必重新检索 |
| `excluded_product_ids` | 用户说「不要 / 除了 / 排除」 | 从候选里剔除 |
| `preferences` | 逐轮补充的偏好 | 注入上下文头部 |

`last_candidates` 是收益最直接的一个：用户问「那有货吗」时，Agent 不需要重新检索，
而是直接用会话里记住的候选去查库存。**追问从 3 次工具调用降到 1 次**，且不会因为
重新检索而漂移到别的商品上（见 `tests/test_react.py::test_follow_up_question_reuses_remembered_candidates`）。

槽位会被渲染成一行摘要，**始终**注入上下文头部：

```
[会话状态] 预算上限 ¥500；类目 运动鞋；最近候选 shoe-001、shoe-002。
```

这一行的意义在于：即使历史事件被压缩掉了，已确认的约束也不会丢。

## 2. 上下文预算：压缩是持久且幂等的

`ContextBudget` 把 token 分成两段：`max_tokens` 是上限，`reserve_tokens` 预留给模型回复，
两者之差是历史可用的 `usable_tokens`。

`SessionMemory.window()` 在预算内产出 messages，分四个阶段：

| 阶段 | 动作 | 目的 |
|---|---|---|
| 一 | 把最旧的轮次折叠成摘要 | 保留信息密度，丢掉冗余 |
| 二 | 从最久远的摘要开始收缩 | 摘要本身也会吃掉预算 |
| 三 | 逐条丢弃更早的事件 | 硬性收缩 |
| 四 | 截断超额的最近一条 | 单条观察可能比剩余额度还大 |

阶段四是踩过坑之后补上的。最初的实现只做前三个阶段，并"硬保底保留最后一条事件"——
但一次检索的观察可能有几百 token，比剩余额度还大，结果预算被突破 2–5 token。
**预算的意义就在于它是硬边界**，所以最后一条允许被截断（加省略号），而不是被整体保留。

另一个容易忽略的点：`window()` 的压缩是**就地且持久**的——被折叠的轮次会从 `turns`
移出并进入 `_summaries`。如果每轮重新从完整历史里现算，摘要会重复累积，
上下文反而越压越大。ReAct 每一步都会重算窗口，这个性质是它成立的前提。

### 实测：预算确实是硬边界

由 `scripts/benchmark_agent_runtime.py` 在本机 CPU 上实测（40 轮连续会话，目录
`data/products.jsonl`，检索走词法基线、ReAct 走 RuleReasoner，不含模型推理开销）：

| 预算 (tokens) | 可用 (tokens) | 轮数 | 最终占用 | 峰值占用 | 首次压缩轮次 | 累计丢弃 | 是否越界 |
|---|---|---|---|---|---|---|---|
| 512 | 256 | 40 | 192 | 244 | 4 | 35 | 否 |
| 1024 | 768 | 40 | 681 | 766 | 3 | 36 | 否 |
| 2048 | 1792 | 40 | 1689 | 1790 | 12 | 21 | 否 |
| 4096 | 3840 | 40 | 3749 | 3832 | 20 | 24 | 否 |

四档预算下峰值都没越界。原始数据见 `results/agent_runtime_benchmark.{json,md}`，
可用 `python scripts/benchmark_agent_runtime.py --turns 40` 复现。

**口径**：token 用启发式估算（CJK 每字 1 token，其余每 4 字符 1 token），
不做真实 BPE 分词。绝对值与具体模型会有偏差，**趋势与边界结论有效**。

## 3. ReAct 循环：显式轨迹 + 失控防护

`ReActAgent.run()` 把一次请求展开成多步，每一步记下
`thought / action / arguments / observation / error / latency_ms`，最终答案必须能回溯到
具体工具返回。`RuleReasoner` 是确定性决策器（离线可测），`Reasoner` 是接口，LLM 决策器可直接替换。

三件事是这个循环能不能上线的关键：

**观察回注。** 工具失败不中断循环，而是作为 observation 参与下一步决策。
权限错误、参数非法、商品不存在、未知工具分别被分类成 `invalid_arguments` / `not_found` /
`unknown_tool` / `tool_failure`，决策器据此选择重试、降级还是收口。
一个工具抛异常不应该让整个会话崩掉。

**失控防护。** 循环只会有五种终止原因：

| `stop_reason` | 触发条件 |
|---|---|
| `final_answer` | 正常收口 |
| `max_steps` | 达到步数上限 |
| `loop_detected` | 同一个 `(tool, arguments)` 被重复请求 |
| `budget_exhausted` | 上下文压缩后仍超预算 |
| `no_candidates` | 检索无结果，属于业务性终止 |

`max_steps` 与 `loop_detected` 是两回事：前者防"步数太多"，后者防"原地打转"。
LLM 决策器最容易犯的错就是反复请求同一个参数的动作，签名去重能直接掐住。

**预算感知。** 每一步后重算上下文窗口，压缩后仍超预算就带 `budget_exhausted` 提前收口。
上下文管理不是事后的日志清理，而是循环里的一个退出条件。

实测（15 次运行，三个查询各 5 次）：平均工具调用 3.0 次，单轮最多 4 次，
端到端 P50 `2.02 ms`、P95 `3.14 ms`，终止原因全部为 `final_answer`。

## 4. 在三处入口暴露

| 入口 | 用法 |
|---|---|
| CLI | `shopping-agent --catalog data/products.jsonl --chat` |
| MCP | `react_query(query, session_id)`，会话状态由服务端按 `session_id` 持有 |
| 代码 | `ReActAgent(tools, session=SessionMemory()).run(query)` |

MCP 侧有一个必须处理的冲突：服务端的**幂等重放**会按 `(tool, arguments)` 缓存 30 秒的返回，
但会话工具重复调用**本就应该改变状态**——命中重放会返回陈旧快照并跳过状态更新。
因此 `ToolRuntime.run(..., idempotent=False)` 让有状态工具绕开重放缓存，
`tests/test_mcp_server.py::test_stateful_react_query_bypasses_the_idempotency_cache` 守着这一点。

会话在服务端按 `session_id` 分片持有，受 `max_sessions` 上限约束（超出按插入顺序淘汰），
可用 `get_session_state` 读取某个会话的槽位与上下文占用。

## 5. 边界：这套实现不证明什么

- **不含真实 LLM 决策器**。`RuleReasoner` 是确定性基线，步数与延迟是接口的上界参考，
  不是"用了大模型的效果"。接入 LLM 只需实现 `Reasoner.decide()`。
- **不证明长会话的检索质量**。目录只有 5 个商品，本模块测的是上下文管理与循环控制，
  不是推荐效果。
- **没有跨进程持久化**。会话在内存里，进程重启即丢失；多实例部署需要外部存储
  （Redis 等），当前未实现。
- **没有做并发压测**。会话字典靠锁保护，但高并发下的行为未实测。
