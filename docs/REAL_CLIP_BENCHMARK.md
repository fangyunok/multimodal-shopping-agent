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
| 磁盘 | 数据盘 ≥ 50 GB | 10 万张商品图的落盘空间；`ABO_ROOT` 建议指到数据盘 |
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

商品数据来自 **Amazon Berkeley Objects (ABO)**（CC BY 4.0），147,702 条商品 listing、398,212 张目录图。

ABO 在 S3 上按**与归档相同的目录结构**开放，因此不需要下载 3 GB 的 `abo-images-small.tar`：
只需 `listings/metadata/listings_*.json.gz`（每片约 5 MB）与 `images/metadata/images.csv.gz`（约 6 MB），
商品图按需逐张取——`--abo-limit` 决定取多少条。

## 可调参数（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `ABO_LIMIT` | `100000` | 取多少条商品，直接决定图片下载量 |
| `SCALES` | `10000,100000` | 压测规模档位 |
| `QUERIES` | `200` | 每档查询条数 |
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

## 延伸方向

- **查询分布对齐**：当前查询向量与语料同分布（同一批商品向量的不同切片）。
  要对齐线上真实的查询分布，可在 `build_real_clip_vectors.py` 里追加一段"自然语言查询文本编码"的导出，
  `benchmark_ann_scaling.py --vectors-query-count` 已预留拆分位置。
- **更接近线上形态的语料**：`--image-weight` 已经与线上索引一致，可再扫若干档位看融合比例对近邻结构的影响。
- **百万级**：ABO 商品总量约 14.8 万，往上一档需要换更大的数据源；`--scales` 已支持任意档位，
  内存需求随条数线性增长。
