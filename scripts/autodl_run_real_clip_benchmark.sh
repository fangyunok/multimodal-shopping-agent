#!/usr/bin/env bash
# 在 GPU 机器（AutoDL / 任意 CUDA 环境）上一键跑「真实 Chinese-CLIP 向量」的 ANN 压测。
#
# 为什么要在 GPU 上做：benchmark_ann_scaling.py 默认吃合成语料，而合成语料的簇结构是构造出来的。
# 换成真实模型编码的商品向量，才能回答"真实 embedding 分布下索引还是这个表现吗"。
#
# 用法（在 GPU 实例的终端里）：
#     git clone https://github.com/fangyunok/multimodal-shopping-agent.git
#     cd multimodal-shopping-agent
#     bash scripts/autodl_run_real_clip_benchmark.sh
#
# 常用覆盖（环境变量）：
#     ABO_LIMIT=95000    取多少条商品（决定下载图片量；ABO 全库实测 92,320 条 listing，封顶即全量）
#     SCALES=10000,90000 压测规模档位（最大档 ≤ 编码条数 − 查询数）
#     ABO_ROOT=/root/autodl-tmp/abo  把数据放到数据盘，避免系统盘写满
#     KEEP_ABO=1         保留 ABO 原始数据（默认跑完删图片省盘）
#
# 跑完产物：outputs/real_clip/ 下的 vectors.npy、vectors.meta.json、
#           results/ann_scaling_benchmark.{json,md}，以及一个可直接下载的 tar.gz。

set -euo pipefail

WORKDIR="${WORKDIR:-$PWD}"
ABO_ROOT="${ABO_ROOT:-$WORKDIR/data/abo}"
# ABO 全库实测共 10 个 listings 分片、92,320 条商品，10 万档物理上不存在。
# 缺图的少量商品（本机全量跑为 380 条）会在转换时跳过，因此最大档留了余量。
SCALES="${SCALES:-10000,90000}"
QUERIES="${QUERIES:-200}"
# 压测脚本的切分规则是「语料取向量文件前缀 [0:count]、查询取紧接其后的 queries 条」，
# 因此导出的向量条数必须 ≥ 最大规模 + 查询数，否则第 6 步会被校验直接拦下。
# （试跑时踩过：导出 10000 条却要 10000 语料 + 200 查询。）这里按 SCALES/QUERIES 自动推导，
# 想手工覆盖仍可显式设 ABO_LIMIT。
MAX_SCALE="$(printf '%s' "$SCALES" | tr ',' '\n' | grep -E '^[0-9]+$' | sort -n | tail -1)"
ABO_LIMIT="${ABO_LIMIT:-$((MAX_SCALE + QUERIES))}"
TOP_K="${TOP_K:-10}"
IMAGE_WEIGHT="${IMAGE_WEIGHT:-0.5}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LISTING_SHARDS="${LISTING_SHARDS:-8}"
WORKERS="${WORKERS:-16}"
KEEP_ABO="${KEEP_ABO:-0}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONUNBUFFERED=1
# pip install -e 装出来的控制台脚本常落在 ~/.local/bin，AutoDL 默认不在 PATH 里。
export PATH="$HOME/.local/bin:$PATH"

ABO_S3="${ABO_S3:-https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com}"
OUT_DIR="$WORKDIR/outputs/real_clip"

log() { printf '\n=== %s ===\n' "$*"; }

cd "$WORKDIR"

log "0/6 环境检查"
python -c "import sys; print('python', sys.version.split()[0])"
if [ "$DEVICE" = "cuda" ]; then
  python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" \
    || { echo "torch 不可用；若确认要用 CPU，请设 DEVICE=cpu"; exit 1; }
fi

log "1/6 安装依赖（clip + ann + dev）"
python -m pip install -q -e ".[clip,ann,dev]"

log "2/6 下载 ABO 元数据（listings 分片 + images.csv.gz）"
mkdir -p "$ABO_ROOT/listings/metadata" "$ABO_ROOT/images/metadata" "$ABO_ROOT/images/small"
for i in $(seq 0 $((LISTING_SHARDS - 1))); do
  target="$ABO_ROOT/listings/metadata/listings_$i.json.gz"
  if [ -f "$target" ]; then echo "已存在 listings_$i"; continue; fi
  if curl -fsSL --retry 3 -o "$target" "$ABO_S3/listings/metadata/listings_$i.json.gz"; then
    echo "已下载 listings_$i"
  else
    echo "listings_$i 不存在，停止（分片到此为止）"; rm -f "$target"; break
  fi
done
if [ ! -f "$ABO_ROOT/images/metadata/images.csv.gz" ]; then
  curl -fsSL --retry 3 -o "$ABO_ROOT/images/metadata/images.csv.gz" "$ABO_S3/images/metadata/images.csv.gz"
fi
ls -la "$ABO_ROOT/listings/metadata" | head
echo "images.csv.gz: $(du -h "$ABO_ROOT/images/metadata/images.csv.gz" | cut -f1)"

log "3/6 下载商品图片（limit=$ABO_LIMIT，workers=$WORKERS）"
shopping-agent --fetch-abo-images "$ABO_ROOT" --abo-limit "$ABO_LIMIT" --download-workers "$WORKERS"
echo "已落盘图片：$(find "$ABO_ROOT/images/small" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l) 张"

log "4/6 转换为商品目录"
mkdir -p data/raw data/processed
shopping-agent --convert-abo "$ABO_ROOT" --abo-limit "$ABO_LIMIT" --abo-output data/raw/abo.jsonl
shopping-agent --prepare-data data/raw/abo.jsonl \
  --output-catalog data/processed/products.jsonl \
  --image-dir data/processed/images
echo "目录条数：$(wc -l < data/processed/products.jsonl)"

log "5/6 用 Chinese-CLIP 编码成真实向量（image_weight=$IMAGE_WEIGHT）"
mkdir -p "$OUT_DIR"
python scripts/build_real_clip_vectors.py \
  --catalog data/processed/products.jsonl \
  --output "$OUT_DIR/vectors.npy" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --image-weight "$IMAGE_WEIGHT"

log "6/6 跑 ANN 压测（真实向量模式）"
python scripts/benchmark_ann_scaling.py \
  --vectors "$OUT_DIR/vectors.npy" \
  --scales "$SCALES" \
  --queries "$QUERIES" \
  --top-k "$TOP_K" \
  --kinds numpy,flat,hnsw,ivf,ivfpq \
  --output-dir results

log "打包结果"
STAMP="$(date +%Y%m%d-%H%M)"
ARCHIVE="$WORKDIR/real_clip_benchmark_$STAMP.tar.gz"
FILES=(results/ann_scaling_benchmark.json results/ann_scaling_benchmark.md)
if [ -f "$OUT_DIR/vectors.meta.json" ]; then FILES+=("$OUT_DIR/vectors.meta.json"); fi
if [ -f data/processed/products.jsonl ]; then FILES+=("data/processed/products.jsonl"); fi
tar czf "$ARCHIVE" "${FILES[@]}"
ls -lh "$ARCHIVE"

if [ "$KEEP_ABO" != "1" ]; then
  log "清理 ABO 原始图片（设 KEEP_ABO=1 可保留）"
  rm -rf "$ABO_ROOT/images/small"
fi

echo
echo "完成。把下面这个文件下载回本机："
echo "  $ARCHIVE"
echo "注意：脚本不会替你关机——AutoDL 按开机时长计费，跑完请到控制台关机。"
