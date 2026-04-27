# Canteen Agent Gateway

智慧食堂 50ms 前置网关原型与生产化示例，覆盖：

- 意图二分类：判断用户 query 是否属于智慧食堂业务范围
- 时间提取：规则抽取今天、明天、餐段、几点等表达
- 用户 query 纠正：领域词典纠正常见错别字、菜品名、食堂名、支付词
- 端到端时延统计：输出单条 query 的 avg、p50、p95、p99、max

当前项目包含假数据、规则 demo、模型训练导出脚本、ONNX INT8 生产推理入口。

## 项目结构

```text
.
├── canteen_gateway_demo.py              # 规则版 demo，便于快速理解流程
├── canteen_test_data_300.json           # 300 条假测试数据
├── generate_canteen_data.py             # 生成 300 条假数据
├── production_gateway.py                # 生产入口，加载 ONNX INT8 模型二分类
├── train_export_domain_classifier.py    # 微调中文小模型并导出 ONNX / INT8 ONNX
├── train_and_eval.ps1                   # Windows 一键安装依赖、训练、评测
├── requirements-prod.txt                # 训练和推理依赖
├── README_PRODUCTION.md                 # 生产部署补充说明
└── artifacts/
    ├── domain_classifier.onnx           # 未量化 ONNX 模型
    ├── domain_classifier_int8.onnx      # INT8 量化 ONNX 模型，生产默认加载
    ├── vocab.txt                        # tokenizer 词表备份
    └── domain_classifier_hf/            # HuggingFace tokenizer/model 配置
```

## 任务边界

本项目只做 50ms 内前置处理：

```text
query
 -> 文本规范化
 -> 领域 query 纠错
 -> 时间规则提取
 -> 二分类模型判断 in_domain / out_domain
 -> 输出结果和时延
```

不包含：

- 图片理解 / VLM
- 100 个二级意图精分类
- 多意图拆分
- 上下文 summary
- 复杂槽位填充

## 模型方案

二分类模型使用推荐的小型中文 BERT/RoBERTa：

```text
uer/chinese_roberta_L-4_H-512
```

训练脚本会：

1. 读取 `canteen_test_data_300.json`
2. 使用 `corrected` 字段作为训练文本
3. 将 `domain` 转成二分类标签
4. 微调 `AutoModelForSequenceClassification`
5. 导出 `artifacts/domain_classifier.onnx`
6. 动态量化成 `artifacts/domain_classifier_int8.onnx`

生产推理默认加载：

```text
artifacts/domain_classifier_int8.onnx
artifacts/domain_classifier_hf/
```

## 运行环境

推荐 Python 3.10+。

在当前 Codex 桌面环境里，使用的是内置 Python：

```powershell
C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

普通机器上如果 `python` 和 `pip` 已在 PATH，可以直接用 `python`。

## 一键运行

Windows PowerShell：

```powershell
.\train_and_eval.ps1
```

脚本会依次执行：

```powershell
python -m pip install -r requirements-prod.txt
python train_export_domain_classifier.py --data canteen_test_data_300.json
python production_gateway.py --data canteen_test_data_300.json
```

如果本机没有全局 `python`，请打开 `train_and_eval.ps1`，把 `$Python` 改成你的 Python 路径。

## 分步运行

安装依赖：

```powershell
python -m pip install -r requirements-prod.txt
```

生成假数据：

```powershell
python generate_canteen_data.py
```

训练并导出模型：

```powershell
python train_export_domain_classifier.py --data canteen_test_data_300.json
```

运行生产推理评测：

```powershell
python production_gateway.py --data canteen_test_data_300.json
```

没有模型产物时，仅做规则 fallback 烟测：

```powershell
python production_gateway.py --data canteen_test_data_300.json --allow-rule-fallback
```

注意：`--allow-rule-fallback` 只能用于本地链路测试，不能用于生产。

## 输出示例

当前 300 条假数据、ONNX INT8 模型版评测结果：

```text
=== Production Gateway Evaluation ===
test_size: 300
domain_accuracy:     100.00% (300/300)
time_accuracy:       100.00% (300/300)
correction_accuracy: 100.00% (300/300)

=== Per-query Latency ===
avg_ms: 4.1550
p50_ms: 3.8720
p95_ms: 5.3543
p99_ms: 6.3168
max_ms: 6.3415
under_50ms: 100.00%
```

这里的 `p50_ms / p95_ms / p99_ms / max_ms` 是单条 query 的端到端耗时分布，包含：

- normalize
- query 纠错
- 时间提取
- ONNX 二分类推理
- 阈值决策

不是 300 条数据的总耗时。

## 数据格式

`canteen_test_data_300.json` 中每条数据格式如下：

```json
{
  "q": "今天中午二食堂有宫爆鸡丁吗",
  "domain": "in_domain",
  "time": true,
  "corrected": "今天中午二食堂有宫保鸡丁吗"
}
```

字段说明：

- `q`：原始用户 query
- `domain`：期望二分类标签，`in_domain` 或 `out_domain`
- `time`：是否期望抽取到时间
- `corrected`：期望纠错后的 query

## 生产接口说明

核心入口在 `production_gateway.py`：

```python
from pathlib import Path
from production_gateway import OnnxBertBinaryClassifier, ProductionCanteenGateway

classifier = OnnxBertBinaryClassifier(
    Path("artifacts/domain_classifier_int8.onnx"),
    Path("artifacts/domain_classifier_hf"),
)
gateway = ProductionCanteenGateway(classifier, threshold=0.5)

result = gateway.handle("今天中午二食堂有宫爆鸡丁吗")
print(result)
```

返回字段：

```text
domain              # in_domain / out_domain
domain_score        # 模型输出的 in-domain 概率
original_query      # 原始 query
corrected_query     # 纠错后 query
correction_applied  # 是否发生纠错
corrections         # 纠错明细
time                # 时间提取结果，没有则为 None
latency_ms          # 单条端到端耗时
```

## 上线注意事项

`canteen_test_data_300.json` 是假数据，只能证明代码链路和时延，不代表真实线上准确率。

真实上线前至少需要：

- 1 万到 2 万条食堂业务正样本
- 1 万到 2 万条非食堂负样本
- 3000 到 5000 条混淆负样本
- 独立测试集，不能参与训练
- 按食堂、档口、菜名、支付、退款、投诉、营业时间等业务域分层评估

建议上线门槛：

```text
domain_accuracy >= 90%
in_domain_recall >= 90%
out_domain_precision >= 90%
p95_latency <= 50ms
```

## 重要说明

当前仓库里的模型是用 300 条假数据训练出来的演示模型。它适合验证：

- 工程链路是否完整
- ONNX INT8 推理是否可用
- 端到端时延是否满足 50ms
- 数据格式和评测方式是否清晰

它不适合作为最终线上业务模型。上线时请用真实标注数据重新训练并替换 `artifacts/` 下的模型产物。
