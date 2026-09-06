# Amazon Berkeley Objects 数据适配

本项目选用 Amazon Berkeley Objects（ABO）作为首个真实商品图文数据源。ABO 官方页面描述其包含 147,702 条商品 listing 和 398,212 张唯一目录图片，并提供多语言标题、描述、品牌、颜色、材质、风格和商品类型等字段。

## 许可证与署名

ABO 官方归档页面和归档内 `README.md` / `LICENSE-CC-BY-4.0.txt` 标明数据使用 CC BY 4.0。使用数据时必须保留以下署名：

- 数据及图片：Amazon.com；
- 数据集构建者：Matthieu Guillaumin、Thomas Dideriksen、Kenan Deng、Himanshu Arora、Jasmine Collins、Shubham Goel、Jitendra Malik 等；
- 论文：*ABO: Dataset and Benchmarks for Real-World 3D Object Understanding*, CVPR 2022。

AWS Open Data Registry 页面当前显示的许可标签与官方归档页存在不一致。实际使用前应再次阅读下载归档中随附的许可证文件；本项目以归档内许可证为准并保留该文件。

## 下载范围

初版只需要：

- `abo-listings.tar`：约 83 MB；
- `abo-images-small.tar`：约 3 GB，最长边不超过 256 像素。

不下载 110 GB 原图、360 度图片或 3D 模型。

官方下载页：<https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/index.html>

## 转换流程

将两个归档解压到同一个 ABO 根目录后执行：

```bash
shopping-agent \
  --fetch-abo-images D:/datasets/abo \
  --abo-limit 1000 \
  --download-workers 8

shopping-agent \
  --convert-abo D:/datasets/abo \
  --abo-limit 1000 \
  --abo-output data/raw/abo.jsonl

shopping-agent \
  --prepare-data data/raw/abo.jsonl \
  --output-catalog data/processed/products.jsonl \
  --image-dir data/processed/images
```

适配器优先选择 `zh_CN`，其次选择 `en_US`，并关联 `main_image_id` 与 256px 图片。ABO 不提供可靠价格和实时库存，因此转换数据中的价格设为 0、库存设为 1，仅用于图文检索实验；价格和库存工具评测仍使用独立的可控数据。
