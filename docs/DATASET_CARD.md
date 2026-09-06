# 数据集说明

## 当前状态

仓库中的 `data/products.jsonl` 是自建冒烟样例，只用于测试接口，不用于报告真实模型效果。最终实验数据集尚未选定。

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

选择真实数据集后，需要记录：数据集名称、发布者、原始链接、许可证版本、下载日期、字段映射、过滤规则、切分规模和已知偏差。

