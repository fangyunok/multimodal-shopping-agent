# 数据集说明

## 当前状态

仓库中的 `data/products.jsonl` 是自建冒烟样例，只用于测试接口，不用于报告真实模型效果。首个真实数据源选定为 Amazon Berkeley Objects（ABO），详细说明见 `docs/ABO_DATASET.md`。

## 数据准入要求

- 数据来源可追溯，许可证允许研究和作品集展示；
- 不包含用户身份、订单、地址等个人数据；
- 每个商品具有稳定 ID、文本字段和可选图片；
- 训练、验证、测试切分由商品 ID 哈希确定，重复运行结果一致；
- 图片按内容 SHA-256 去重；
- 原始数据与处理后数据分离。

## 处理命令

```bash
shopping-agent \
  --prepare-data data/raw/products.jsonl \
  --output-catalog data/processed/products.jsonl \
  --image-dir data/processed/images
```

输出汇总包含商品数量、唯一图片数量、重复图片数量，以及 train/validation/test 数量。

## 待补充

下载 ABO 后需要补充实际下载日期、归档 SHA-256、转换规模、类别分布和语言分布。
