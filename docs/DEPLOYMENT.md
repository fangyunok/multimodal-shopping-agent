# 部署与运维

这份文档只回答一个问题：**这套东西从"本机跑通"变成"上线扛量"之间，差了哪些东西。**

配套阅读：`docs/SCALING.md`（为什么这么改、实测数据）、`docs/ARCHITECTURE.md`（模块分层）。

---

## 一、三种部署形态

| 形态 | 会话存储 | 编码 | 适用 | 命令 |
| --- | --- | --- | --- | --- |
| 单机 demo | 进程内 | 本进程 | 演示、开发、CI | `uvicorn shopping_agent.app:app` |
| 无状态多副本 | Redis | 本进程（每副本一份模型） | 中等规模；副本数少、模型小 | `docker compose up --scale shopping-agent=3` |
| 编码服务分离 | Redis | 独立 GPU 服务 | 模型大 / 副本多 / 要省显存 | 见第五节 |

关键分界线是**状态放哪**：只要会话还在进程内存里，副本就不能>1，滚动发布就会丢会话。
所以 `SESSION_BACKEND=redis` 不是"优化"，而是"能不能扩容"的前置条件。

---

## 二、进程与端口

| 进程 | 端口 | 说明 |
| --- | --- | --- |
| 检索服务 | 8000 | FastAPI，可水平扩副本 |
| 编码服务 | 8100 | `scripts/serve_encoder.py`，只此一份（或少数几份）GPU |
| Redis | 6379 | 会话状态；不持久化也够用 |

### 检索服务的探针

| 端点 | 语义 | 用途 |
| --- | --- | --- |
| `GET /healthz` | 存活：进程还在就 200 | `livenessProbe` / Docker `HEALTHCHECK` |
| `GET /readyz` | 就绪：检索器已加载 + 索引可读 + 会话存储可达 | `readinessProbe`，不 ready 就从 Service 摘掉 |
| `GET /index` | 当前索引版本、可回滚版本列表 | 发布后确认、排查"是不是发了旧版本" |
| `GET /metrics` | 编码服务的缓存命中率、平均批次大小 | 编码服务专用 |

**为什么两个探针必须分开。** Redis 挂了的时候，`/readyz` 应该返回 degraded 让流量绕开这个副本；
但如果把它接到存活探针上，编排系统会**不断重启进程**——而重启一个没坏的进程治不好一个挂掉的 Redis，
只会让情况更糟（所有副本反复重启、连接风暴）。

```bash
curl -s localhost:8000/healthz   # {"status":"alive"}
curl -s localhost:8000/readyz    # {"status":"ready","checks":{...}}
```

`/readyz` 的返回里逐项列出 `retriever`（商品数、索引摘要）与 `session_store`（后端类型、是否可达、TTL）。
排查线上"5xx 变多"时，先看这个端点，比翻日志快。

---

## 三、Docker Compose 快速起一套

```bash
docker compose up --build
# 起 3 个检索副本，验证无状态
docker compose up --build --scale shopping-agent=3
```

Compose 里已经配好 `SESSION_BACKEND=redis` 与 `REDIS_URL`，并且 `depends_on: condition: service_healthy`——
检索副本会等 Redis 真正能 `PING` 通了才启动，避免启动瞬间 `/readyz` 全红。

要启用 ANN 后端，镜像需要带 faiss：

```bash
docker build --build-arg INSTALL_EXTRAS=ann -t shopping-agent:ann .
```

然后把 `RETRIEVER_BACKEND=ann`、`INDEX_DIR=/app/data/index/v3` 传进去（索引目录挂载进容器）。

### 验证"无状态"确实成立

```bash
# 同一个 session_id 连续打两次，第二次故意打到另一个副本
curl -s -X POST localhost:8010/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-demo","query":"预算 500 以内的通勤运动鞋"}'
curl -s -X POST localhost:8010/chat -H 'Content-Type: application/json' \
  -d '{"session_id":"s-demo","query":"有货吗"}'
```

第二轮能接上一轮的上下文（`context.kept_turns` 递增），说明状态在 Redis 而不是在某个副本的内存里。
响应里的 `conflicts` 字段是**版本冲突计数**——正常应为 0；持续 >0 说明同一个 `session_id`
在被多个客户端并发驱动，正确修法是在网关按 `session_id` 做一致性路由，而不是调大重试次数。

---

## 四、环境变量

### 检索

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RETRIEVER_BACKEND` | `baseline` | `baseline` / `clip` / `fusion` / `ann` / `partitioned` |
| `CATALOG_PATH` | `data/products.jsonl` | 商品目录 |
| `INDEX_DIR` | — | 索引目录；`ann` 与 `partitioned` 后端必填 |
| `ANN_KIND` | `hnsw` | 运行期只用来定位**索引类型**；维度与类型以清单为准 |
| `ANN_EF_SEARCH` | `64` | HNSW 查询宽度。**改了立刻生效，不需要重建索引** |
| `ANN_NPROBE` | `32` | IVF 扫描桶数。同上，可热调 |
| `ANN_RERANK_CANDIDATES` | `0` | 粗召回 + 精确重排的候选数，`0` 表示关闭。开启后分数改为用 `vectors.npy` 重算，因此**索引必须保留向量** |
| `PLANNER_BACKEND` | `rule` | `rule` / `llm` |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | — | `llm` 规划器必填 |

> 召回率不够时的第一反应应该是**调大 `ANN_EF_SEARCH` / `ANN_NPROBE`**，而不是重建索引。
> 这两个值都被 faiss 序列化进索引文件，加载后可以任意覆盖。

> 注意这里刻意**读不到** HNSW 的 `m` / `efConstruction`、IVF 的 `nlist`——它们是**构建期**参数，
> 改了必须重建索引。把它们做成环境变量只会制造"我改了配置为什么没变化"的假问题。

**`ann` 与 `partitioned` 的选择**：`ann` 是单份索引，`partitioned` 是按价位/类目切成的多份分段索引
（过滤下推）。后者只在过滤**有选择性**时才有收益——`max_price` 高过全库最高价时所有分段都会被选中，
比单索引还慢。它解决的是"价格过滤与向量相似度无关"这一类问题，不是通用提速手段。
判据与取舍见 [SCALING.md](SCALING.md)。

### 会话与运行期

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `SESSION_BACKEND` | `memory` | `memory` / `redis` |
| `REDIS_URL` | `redis://localhost:6379/0` | |
| `SESSION_TTL_SECONDS` | `3600` | 会话过期；每次写入自动续期 |
| `CONTEXT_BUDGET` | `2048` | 单会话上下文 token 上限 |
| `MAX_STEPS` | `6` | ReAct 单轮最大步数 |
| `INDEX_ROOT` | — | 索引版本根目录；设置后 `/index` 才返回版本状态 |

### 编码服务

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `ENCODER_BASE_URL` | — | 设置后检索副本走远端编码，**本进程不再加载模型** |
| `ENCODER_DEVICE` | `cuda:0`（有则用） | 编码服务侧 |
| `ENCODER_MAX_BATCH` | `256` | 动态批处理上限 |
| `ENCODER_MAX_WAIT_MS` | `8` | 攒批最长等待；高 QPS 下攒得满，低 QPS 下不拖尾 |
| `EMBEDDING_CACHE_SIZE` | `4096` | 进程内 LRU 容量 |

---

## 五、编码服务分离（阶段三）

模型前向是这个系统里最贵的一步，而且它和检索副本的扩缩容诉求**正好相反**：
编码要"少而强"（一台 GPU 吃满批），检索要"多而廉"（CPU 副本廉价横向扩）。

```bash
# GPU 机器
pip install -e ".[clip,service]"
ENCODER_DEVICE=cuda:0 python scripts/serve_encoder.py --host 0.0.0.0 --port 8100

# 检索副本
export ENCODER_BASE_URL=http://encoder.internal:8100
```

拆开之后检索副本的镜像里**没有 torch**，镜像体积和冷启动时间都会明显下降。

### 三个接口

```
POST /embed/text   {"texts": ["..."]}        -> {"vectors": [[...]], "cache_hit_ratio": 0.42, ...}
POST /embed/image  {"images_base64": ["..."]} -> {"vectors": [[...]]}
GET  /healthz                                 -> 编码器与批处理统计
```

**图片走 base64 而不是文件路径**，因为服务端不应该能读客户端文件系统。允许传路径等于把
"任意文件读取"直接做成一个接口。这个项目在图片上传接口上已经踩过同一个坑，这里保持一致。

### 批处理与缓存

- `DynamicBatcher`：多个请求的编码任务合并成一批提交给模型。低 QPS 时等 `max_wait_ms` 就走，
  不会为了攒批把单请求尾延迟拖到几百毫秒；高 QPS 时自然攒满 `max_batch`。
- `EmbeddingCache`：两级缓存（进程内 LRU + 可选 Redis）。电商场景里"同一本书被反复搜"很常见，
  命中缓存直接省掉一次前向。

---

## 六、发布与回滚（索引版本）

索引重建期间服务不能中断，所以要能"新版本建好 → 原子切换 → 出问题回滚"。

```bash
# 1. 建新版本（写到 v4，不动线上流量）
python -m shopping_agent.cli --build-ann-index data/index/v4 --ann-kind hnsw \
    --catalog-snapshot --catalog data/products.jsonl

# 2. 校验：清单、条目数、维度是否自洽
python -m shopping_agent.cli --index-status --index-root data/index

# 3. 原子发布：改 CURRENT 指针（os.replace 是原子操作，不会出现"半发布"）
python -m shopping_agent.cli --promote-version 4 --index-root data/index

# 4. 出问题回滚到上一版
python -m shopping_agent.cli --rollback-index --index-root data/index

# 5. 清理旧版本，只留最近 2 个
python -m shopping_agent.cli --prune-versions 2 --index-root data/index
```

目录结构：

```
data/index/
├── CURRENT              # 内容就是版本号，例如 "3"
├── v3/
│   ├── manifest.json    # 索引清单（v2：含 index_type / backend / ann 参数 / effective）
│   ├── ann.index        # faiss 索引（numpy 后端没有这个文件）
│   ├── vectors.npy      # 原始向量（用于精确兜底；--no-vectors 可省）
│   └── products.jsonl   # 商品快照（--catalog-snapshot 才有）
└── v4/ ...
```

指针用 `os.replace` 切换：同一文件系统内的 rename 是原子的，所以不存在"读到一半指针"的状态。
回滚就是 `promote(上一版)`，把指针改回去——不需要重新构建。

### 分段索引（过滤下推）

当"价格过滤与语义相似度无关"导致 ANN 精度崩掉时（见 [SCALING.md](SCALING.md) 2.7 节），
正解不是继续调参，而是把过滤做进索引结构：按价位分段建索引，查询时先按 `max_price` 选段。

```bash
python -m shopping_agent.cli --build-partitioned-index data/index/v4 \
    --partition-by price --partition-buckets 8 --ann-kind ivf \
    --catalog-snapshot --catalog data/products.jsonl
```

发布与回滚流程与单索引**完全一致**（`--promote-version` / `--rollback-index` 不需要区分）：
分段索引同样会写一份标准 `manifest.json`，否则它就会被排除在原子发布链路之外。
目录多一层：

```
data/index/v4/
├── manifest.json        # 标准清单：index_type=partitioned，含 partition_by / partition_count
├── partitions.json      # 分段布局：每段的目录、商品数、价格区间、类目集合
├── part-0000/
│   ├── ann.index
│   ├── members.npy      # 本段商品在全局目录中的下标
│   └── vectors.npy
└── part-0001/ ...
```

`members.npy` 不是可有可无的附属品：每个分段有**自己的局部下标**，而返回给调用方的必须是
全局商品下标。少了这层映射，检索会把 A 的向量返回成 B 的商品——这是正确性问题，不是性能问题。

用 `RETRIEVER_BACKEND=ann` 加载分段目录会**直接报错**并提示正确入口。把 N 个分段当成一份索引读，
只会得到一个溯源不出来的维度错误。

### 开启精确重排

```bash
# 索引必须保留 vectors.npy（构建时不要加 --no-vectors）
RETRIEVER_BACKEND=ann ANN_RERANK_CANDIDATES=500 python -m shopping_agent.cli --query "轻便运动鞋"
```

重排是"粗召回 + 精确重排"的后半段：先取 500 个候选，再用原始向量重算分数。
它让 IVF-PQ 的压缩收益真正兑现（recall@10 从 0.097 回到 0.9375），代价是这次查询多算 500 次内积。

配了 `ANN_RERANK_CANDIDATES` 却没有 `vectors.npy` 时**直接报错**，不会静默降级——
静默降级的代价不是"白配了"，而是让人误以为"重排没用"，进而把已经有效的优化删掉。

### 增量更新

全量重建十万级索引要几十秒到几分钟。日常上新不该每次都付这个代价：

```bash
# 1. 写入变更日志（upsert / delete）
cat data/changes/2026-10-05.jsonl
# {"op":"upsert","product":{...},"seq":1}
# {"op":"delete","product_id":"p-123","seq":2}

# 2. 以当前发布版本为基线，产出新版本并直接发布
python -m shopping_agent.cli --apply-changes data/changes/2026-10-05.jsonl \
    --from-current --index-root data/index --publish
```

变更应用后会**压实（compaction）**：删除会留下空洞，而向量行号必须和商品下标严格一一对应，
否则检索会把 A 的向量返回成 B 的商品。所以删除后要把向量与商品列表一起重排。

---

## 七、常见故障与处置

| 现象 | 可能原因 | 处置 |
| --- | --- | --- |
| `/readyz` 报 `session_store.reachable=false` | Redis 不通 | 修 Redis；流量会被就绪探针自动摘掉，不用重启进程 |
| `/chat` 返回 409 | 同一 `session_id` 被并发驱动 | 网关按 `session_id` 做一致性路由；不是靠加重试解决 |
| `/chat` 响应里 `conflicts>0` | 同上，但重试成功了 | 观察是否持续；持续出现说明路由有问题 |
| 召回明显偏低 | `ef_search` / `nprobe` 配小了 | 热调大这两个值，先不动索引 |
| 内存超限 | IVF/HNSW 都是**不压缩**的 | 换 `ivfpq`（约 1/10 体积），代价是召回下降；见 `docs/SCALING.md` 实测 |
| 启动即失败、报维度不一致 | 索引清单与 `ANN_DIMENSION` 对不上 | 以清单为准；清单是唯一事实来源 |
| 发布后结果异常 | 发到了半成品版本 | `--index-status` 看 CURRENT，必要时 `--rollback-index` |

---

## 八、扩容时先算什么

容量估算的完整模型在 `docs/SCALING.md` 第八节。这里只给三条结论：

1. **先看索引内存**，不是 CPU。十万条 512 维 = 195 MB；百万条 ≈ 2 GB。
   多副本时每副本各一份，所以 `副本数 × 索引体积` 才是总内存。
2. **再用 `ivfpq` 换内存**。十万条从 198 MB 降到 9.4 MB，但召回从 1.0 掉到 0.097——
   这个数字在业务上通常不可接受，所以 PQ 只在"内存真的放不下"时才用，而不是默认。
3. **编码服务不要跟着检索副本一起扩**。它是 GPU 侧瓶颈，扩检索副本只会让它排队更长。
   检索副本的扩容依据是 QPS 与延迟，编码服务的扩容依据是 GPU 利用率与队列长度。
