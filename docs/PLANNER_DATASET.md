# Planner 人工评测集数据卡

## 文件与用途

- `data/planner_dev.jsonl`：20 条开发用例，仅用于 Prompt、解析和错误处理调试；
- `data/planner_test.jsonl`：100 条锁定测试用例，只用于最终对照，不应根据其失败案例继续修改 Prompt 后重复宣称测试成绩。

每条数据均逐条设计并标注自然语言请求、用户显式 intent、期望 `Plan` 和场景标签。当前目录限定为公开示例商品支持的 `运动鞋`、`箱包`、`服装`。

## 测试集分布

| 维度 | 数量 |
| --- | ---: |
| search | 56 |
| compare | 22 |
| inventory | 22 |
| budget | 26 |
| synonym | 50 |
| explicit_intent | 10 |
| underspecified | 13 |
| adversarial | 5 |

标签可以重叠，例如一条请求可以同时属于 `compare`、`budget` 和 `synonym`。测试保证每条数据只有一个主 intent 标签，开发集与测试集的 query 不重复，ID 全局唯一。

## 标注字段

- `query`：用户原始表达；
- `requested_intent`：API 显式 intent，默认 `auto`；
- `expected.intent`：`search`、`compare` 或 `inventory`；
- `expected.max_price`：明确的最高预算，未提及时为 `null`；
- `expected.category`：规范化目录类目，无法判断时为 `null`；
- `tags`：用于切片分析的挑战类型。

## 规则基线

100 条锁定集上，Rule Planner 的 intent/max_price/category/joint accuracy 分别为 0.75/0.87/0.49/0.36，非法输出率为 0。`exact` 联合准确率为 0.9524，而 `synonym` 为 0，主要错误来自类目同义词、中文数字预算、隐式比较和隐式库存表达。

该集合评测规划结构，不评测商品排序和最终回答；端到端指标继续使用 `data/eval.jsonl`。当前用例由项目作者单人标注，尚未计算多人标注一致性，面试时需主动说明这一限制。
