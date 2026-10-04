"""把真实商品目录编码成 ANN 压测用的向量文件。

为什么要单独一个脚本
--------------------
``benchmark_ann_scaling.py`` 默认吃合成语料，而合成语料的簇结构是**构造出来**的。
要回答"换成真实 embedding 分布，索引还是这个表现吗"，就必须拿真实模型产出的向量去跑。
这个脚本负责产出那份向量。

口径必须与线上一致
------------------
直接复用 ``indexing.encode_products``——也就是服务构建索引时用的同一段图文融合逻辑
（文本向量与图像向量按 ``image_weight`` 加权后 L2 归一化）。压测用的向量若是另一套口径，
跑出来的数字就说明不了上线后的表现。

用法
----
    # GPU 上编码（catalog 里的 image_path 能解析到本地图时才会融合图像）
    python scripts/build_real_clip_vectors.py \
        --catalog data/processed/products.jsonl \
        --output data/processed/real_clip_vectors.npy \
        --device cuda --batch-size 64 --image-weight 0.5

    # 再压测（脚本会从向量文件末尾切查询，与语料不重叠）
    python scripts/benchmark_ann_scaling.py \
        --vectors data/processed/real_clip_vectors.npy --scales 10000,100000 --queries 200

注意：导出的向量条数要**比最大压测规模多出查询段**（默认 200 条），否则压测会直接报错
提醒你重新导出——这是有意为之，避免查询命中自身把 recall 抬到没有意义的高位。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopping_agent.encoders import ChineseClipEncoder  # noqa: E402
from shopping_agent.indexing import encode_products, load_catalog  # noqa: E402

DEFAULT_MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"


def catalog_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="导出真实 CLIP 商品向量（供 ANN 压测使用）")
    parser.add_argument("--catalog", required=True, help="商品目录 jsonl；image_path 可解析时才融合图像")
    parser.add_argument("--output", required=True, help="输出 vectors.npy")
    parser.add_argument("--meta", default=None, help="元数据 json；默认是输出同目录的 .meta.json")
    parser.add_argument("--model-id", default=DEFAULT_MODEL, help="Hugging Face 模型名（--model 未给时用它）")
    parser.add_argument("--model", default=None, help="本地模型目录；给了就不联网下载")
    parser.add_argument("--device", default="cuda", help="cuda / cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-weight", type=float, default=0.5, help="与线上索引保持一致（默认 0.5）")
    parser.add_argument("--limit", type=int, default=0, help="只编码前 N 条；0 表示全部")
    args = parser.parse_args()

    catalog = Path(args.catalog).resolve()
    if not catalog.exists():
        raise SystemExit(f"商品目录不存在：{catalog}")
    products = load_catalog(catalog)
    if args.limit:
        products = products[: args.limit]
    if not products:
        raise SystemExit("商品目录为空，无法编码")

    images = 0
    for product in products:
        resolved = product.resolved_image(catalog)
        if resolved is not None and resolved.exists():
            images += 1
    print(
        f"待编码商品 {len(products)} 条，其中 {images} 条能解析到本地图片"
        f"（image_weight={args.image_weight}，device={args.device}）"
    )
    if args.image_weight > 0 and images == 0:
        print("警告：没有任何商品能解析到图片，本次实际只编码了文本向量（等于 image_weight=0）")

    encoder = ChineseClipEncoder(
        model_name=args.model or args.model_id, device=args.device, batch_size=args.batch_size
    )
    start = time.perf_counter()
    vectors = np.asarray(encode_products(products, catalog, encoder, args.image_weight), dtype=np.float32)
    elapsed = time.perf_counter() - start

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, vectors)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vectors": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "image_weight": args.image_weight,
        "products_with_images": images,
        "products_total": len(products),
        "model": args.model or args.model_id,
        "device": args.device,
        "batch_size": args.batch_size,
        "catalog": str(catalog),
        "catalog_sha256": catalog_sha256(catalog),
        "encode_seconds": round(elapsed, 2),
        "seconds_per_item": round(elapsed / max(len(products), 1), 4),
        "environment": {
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
        },
    }
    meta_path = Path(args.meta).resolve() if args.meta else output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    size_mb = output.stat().st_size / (1024 * 1024)
    print(f"已写入 {output}（{meta['vectors']} × {meta['dimension']}，{size_mb:.1f} MB，耗时 {elapsed:.1f}s）")
    print(f"元数据 {meta_path}")
    print("提示：压测时 --queries 要留足余量，脚本会从向量文件末尾切查询、与语料不重叠")


if __name__ == "__main__":
    main()
