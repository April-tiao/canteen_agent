# 智慧食堂 50ms 网关

## 时延口径

`p50_ms / p95_ms / p99_ms / max_ms` 都是单条 query 从进入网关到输出结果的端到端耗时分布，包含：

- 文本 normalize
- query 领域纠错
- 时间提取
- 二分类打分
- 阈值决策

不是 300 条数据总耗时。

## 当前推荐上线结构

生产链路：

```text
query
 -> normalize
 -> domain correction
 -> rule time extraction
 -> ONNX INT8 Chinese-RoBERTa/TinyBERT binary classifier
 -> in_domain / out_domain
```

推荐二分类底座：

```text
uer/chinese_roberta_L-4_H-512
```

上线产物：

```text
artifacts/domain_classifier_int8.onnx
artifacts/domain_classifier_hf/
```

## 训练并导出 ONNX INT8

先安装依赖：

```powershell
pip install -r requirements-prod.txt
```

训练并导出：

```powershell
python train_export_domain_classifier.py --data canteen_test_data_300.json
```

导出成功后运行生产评测：

```powershell
python production_gateway.py --data canteen_test_data_300.json
```

## 本地无模型时的烟测

仅用于验证代码链路，不代表模型效果：

```powershell
python production_gateway.py --data canteen_test_data_300.json --allow-rule-fallback
```

## 上线前必须替换的数据

`canteen_test_data_300.json` 是假数据，只能用于链路测试。上线前至少需要：

- 1万到2万条食堂正样本
- 1万到2万条非食堂负样本
- 3000到5000条混淆负样本
- 独立测试集，不参与训练

上线门槛建议：

```text
domain_accuracy >= 90%
in_domain_recall >= 90%
out_domain_precision >= 90%
p95_latency <= 50ms
```
