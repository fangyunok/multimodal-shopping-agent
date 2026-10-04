# 真实 CLIP 向量上的 ANN 压测

## 为什么单独做一次

`scripts/benchmark_ann_scaling.py` 的默认语料是合成向量（高斯混合，簇结构由 `synthetic.py` 构造）。
它的价值在于把索引结构、自适应超采样、过滤下推这些机制**逐一隔离**出来观察——每条查询都有确定性真值，
不掺模型噪声。

真实商品 embedding 的分布是否同样"分簇、簇内紧密"，是另一件事。
把编码器换成 Chinese-CLIP 的商品向量再跑一遍**同一套判据**，两份报告就能逐列对比。

判据完全一致：同一个脚本、同一组索引参数、同一个 `recall@k` 定义。差别只在语料。

## 前置条件

| 项 | 建议 | 说明 |
|---|---|---|
| 算力 | 单卡 24 GB（RTX 3090 / 4090 等），按量计费 | Chinese-CLIP ViT-B/16 本身很小，余量留给图片解码与批处理 |
| 磁盘 | 数据盘 ≥ 50 GB | 全量 92,320 条对应约 8 万张商品图（实测 80,310 张、<1 GB）；`ABO_ROOT` 建议指到数据盘 |
| 网络 | 可达 AWS S3 与 Hugging Face（或配 `HF_ENDPOINT` 走镜像） | 脚本默认 `HF_ENDPOINT=https://hf-mirror.com` |

**耗时与成本看两个真实读数**：商品图片下载（受带宽影响）与图像塔编码。
编码的实际耗时会被写进 `outputs/real_clip/vectors.meta.json` 的 `encode_seconds` 与 `seconds_per_item`，
按这份读数估算成本，不需要预估。

## 一键运行

```bash
git clone https://github.com/fangyunok/multimodal-shopping-agent.git
cd multimodal-shopping-agent
bash scripts/autodl_run_real_clip_benchmark.sh
```

脚本依次完成六步：

1. 环境检查（Python、torch、CUDA 可用性）
2. 安装依赖 `.[clip,ann,dev]`
3. 下载 ABO 元数据（listings 分片 + `images.csv.gz`）
4. 下载商品图（`--fetch-abo-images`）
5. 转换商品目录（`--convert-abo` → `--prepare-data`）
6. Chinese-CLIP 编码 → ANN 压测 → 打包结果

## 数据来源与规模

商品数据来自 **Amazon Berkeley Objects (ABO)**（CC BY 4.0），归档口径为 147,702 条商品 listing、398,212 张目录图。

**实测全库规模**：listings 分片实际只发布到 `listings_0` ~ `listings_9` 共 10 片，合计 **92,320 条**。
10 万档物理上不存在，最大可行档位 ≈ 92,320 − 缺图跳过 − 查询段（默认配置下约 90,900）。

ABO 在 S3 上按**与归档相同的目录结构**开放，因此不需要下载 3 GB 的 `abo-images-small.tar`：
只需 `listings/metadata/listings_*.json.gz`（每片约 5 MB）与 `images/metadata/images.csv.gz`（约 6 MB），
商品图按需逐张取——`--abo-limit` 决定取多少条。

转换期还有一个必须满足的前提：**`--convert-abo` 只接受图片已在本地磁盘的商品**（`abo.py` 的路径解析基于本地目录），
所以"先转换、后补图"行不通，第 4 步下载必须一次取全。

## 可调参数（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `ABO_LIMIT` | 自动 = 最大档 + 查询数 | 取多少条商品，直接决定图片下载量；封顶即全量 92,320 |
| `SCALES` | `10000,90000` | 压测规模档位（最大档 ≤ 编码条数 − 查询数） |
| `QUERIES` | `200` | 每档查询条数（本轮实测取 1000） |
| `IMAGE_WEIGHT` | `0.5` | 图文融合比例，与线上索引构建保持一致 |
| `ABO_ROOT` | `data/abo` | ABO 数据目录，建议指到数据盘 |
| `KEEP_ABO` | `0` | 设 `1` 保留原始图片，便于重复实验 |

## 口径：查询向量是怎么切出来的

导出的 `vectors.npy` 是一份连续的商品向量。压测时按 **前缀作语料、紧随其后的 N 条作查询** 切开，
两段不重叠——`benchmark_ann_scaling.py --vectors` 就是这么做的。

这一条必须明确：真实商品向量自带强自相关，若让查询命中自身，recall 会被抬到一个没有意义的高位。
仓库在跨视角评测里拒绝过"原图搜原图"式的数据泄漏，这里是同一条原则。

因此导出条数要**比最大压测规模多出查询段**（默认 200 条），否则压测会直接报错提醒重新导出。

## 产物

| 文件 | 内容 |
|---|---|
| `results/ann_scaling_benchmark.md` | 可读报告：索引层表格 + 重排对照 |
| `results/ann_scaling_benchmark.json` | 同上的结构化数据 |
| `outputs/real_clip/vectors.meta.json` | 向量元数据：模型、维度、`image_weight`、条数、目录 SHA-256、编码耗时 |
| `real_clip_benchmark_<时间戳>.tar.gz` | 上述文件打包，下载回本机即可 |

## 两份报告怎么逐列对比

表格列完全一致，按列读：

- **构建耗时 / 索引体积**：只取决于向量条数与维度，两套语料应当接近——差异大说明导出口径有问题。
- **P50 / P99**：图索引的延迟对数据分布的**局部密度**敏感。合成语料是各向同性的高斯簇，
  真实 embedding 的簇内结构不同，这一列最可能出现差异。
- **recall@k**：IVF 的表现很大程度上取决于 `nlist` 与实际有效簇数的匹配程度。
  这一列是判断"合成语料上的结论能否外推"的主要依据。
- **重排对照**：压缩索引的候选集质量同样受分布影响，`candidates` 取多少才够，两套语料可以互相印证。

## 跑完的两件事

1. **下载 `real_clip_benchmark_<时间戳>.tar.gz` 回本机**，把产物落进仓库。
2. **到控制台关机**——云平台按实例开机时长计费，脚本不会替你关机。

## 实测结果（2026-10-04，单卡 RTX 3090）

完整报告：[results/real_clip/ann_scaling_benchmark.md](../results/real_clip/ann_scaling_benchmark.md)（真实向量），
对照 [results/ann_scaling_benchmark.md](../results/ann_scaling_benchmark.md)（合成语料）。

### 运行读数

| 环节 | 读数 |
|---|---|
| 实例 | AutoDL RTX 3090（24 GB），1.48 元/小时，按秒计费 |
| ABO 全量图片 | 92,320 条 listing → 91,940 条转换成功（380 条缺图跳过），80,310 张图落盘 |
| 图片下载 | 24 并发，约 37 张/秒，全量约 35 分钟 |
| 编码 | 91,940 条 × 512 维，fp32、batch 64，**1088.8 s（84.5 条/秒，0.0118 s/条）**，全部商品均解析到本地图片 |
| 压测 | 档位 10,000 / 90,940，查询 1000 条/档，top-10 |
| 总成本 | 实例从开机到关机约 2.5 小时 ≈ **4 元** |

`encode_seconds` / `seconds_per_item` 由 `vectors.meta.json` 自动记录，换卡后按该读数重估成本即可。

### 编码精度对比：fp32 vs fp16（同卡同数据，2026-10-04）

编码器支持 `--dtype float16/bfloat16`（`ChineseClipEncoder(dtype=...)`，浮点输入按模型精度搬运、输出统一 float32）。
同卡同目录数据全量复跑：

| 指标 | fp32 | fp16 | 结论 |
|---|---|---|---|
| 全量编码（91,940 条） | 1088.8 s（84.5 条/s） | **579.3 s（158.8 条/s）** | **1.88× 提速**，编码耗时直接减半 |
| 向量一致性（逐行余弦） | — | mean 0.999999 / min 0.999918 | 与 fp32 向量几乎逐位一致 |
| recall@10 @10,000（hnsw / ivf） | 0.9994 / 0.9971 | 0.9996 / 0.9971 | 持平 |
| recall@10 @10,000（flat / ivfpq） | 0.9999 / 0.753 | 1.0 / 0.7595 | 持平（差异在噪声内） |

**结论：fp16 编码是纯收益**——吞吐翻倍、召回不动、向量与 fp32 余弦 ≥ 0.9999。线上索引构建与压测导出默认用 fp16 即可；
fp32 保留作对照基准。完整对比数据：[results/real_clip/ann_scaling_benchmark_fp16_10k.md](../results/real_clip/ann_scaling_benchmark_fp16_10k.md)。

```bash
# fp16 编码一行开关
python scripts/build_real_clip_vectors.py --catalog data/processed/products.jsonl \
    --output outputs/real_clip/vectors.npy --device cuda --dtype float16
```

### 索引层（真实向量，规模 90,940）

| 索引 | 构建(s) | 体积(MB) | P50(ms) | P99(ms) | recall@10 |
|---|---|---|---|---|---|
| numpy 全量精确 | 0.10 | 177.6 | 1.79 | **71.9** | 1.0 |
| flat | 0.16 | 177.6 | 9.75 | 10.3 | 0.9999 |
| **hnsw** | 36.7 | 201.2 | **0.33** | **0.58** | **0.9996** |
| ivf (nlist=1024) | 25.2 | 180.3 | 0.65 | 0.92 | 0.9983 |
| ivfpq | 66.3 | 8.8 | 0.25 | 0.33 | 0.6841 |

### 粗召回 + 精确重排（规模 90,940）

| 索引 | 候选 10 | 候选 100 | 候选 500 |
|---|---|---|---|
| ivfpq | 0.6841 | **0.994** | **0.9984** |
| ivf | 0.9983 | 0.9983 | 0.9983 |

### 两条结论（与合成语料对照读）

1. **选型结论在真实分布上成立**。HNSW 仍是延迟与召回的双优解（P50 0.33 ms / recall 0.9996）；
   IVF 在自适应 nlist 下召回 0.9983；IVF-PQ 原始分数不可用但候选集质量好——**取 100 候选重排就把召回从 0.684 拉到 0.994**（合成语料上同口径是 0.30 → 0.9995），真实商品分布下 PQ 码本分得更开，候选反而更干净。
2. **真实分布给全量精确扫描引入了合成语料上不存在的尾延迟**：numpy 后端 P50 1.79 ms，P99 却冲到 71.9 ms（放大 40 倍）；HNSW 的 P99 稳在 0.58 ms。也就是说**规模越大，精确检索的长尾越危险，近似索引反而越必要**——这条在合成语料（各向同性高斯簇）上是读不出来的，属于本轮最有价值的新事实。

> 注：两份报告的绝对延迟不可直接比（合成报告在本机 Windows CPU、真实报告在云上 Linux CPU 产出），可比的是**同一报告内的相对关系**与 recall 列。

## 延伸方向

- **查询分布对齐**：当前查询向量与语料同分布（同一批商品向量的不同切片）。
  要对齐线上真实的查询分布，可在 `build_real_clip_vectors.py` 里追加一段"自然语言查询文本编码"的导出，
  `benchmark_ann_scaling.py --vectors-query-count` 已预留拆分位置。
- **更接近线上形态的语料**：`--image-weight` 已经与线上索引一致，可再扫若干档位看融合比例对近邻结构的影响。
- **百万级**：ABO 在 S3 上实际开放 92,320 条 listing，已在本轮用满；往上一档需要换更大的数据源（如 Amazon Reviews / McAuley 商品图文库）。
  `--scales` 已支持任意档位，内存需求随条数线性增长。
