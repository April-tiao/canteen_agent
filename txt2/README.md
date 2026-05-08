# txt2 文本处理项目

`txt2` 是最终可打包交付的文本处理项目，覆盖三件事：

- query 纠错
- 时间段提取
- 文本意图分类，也就是判断是否属于智慧食堂业务域

意图分类模型已使用新数据 `data/canteen_query_dataset.json` 从基础模型重新训练。纠错词典也从新数据 train split 的 `corrections` 标注生成。

## 目录结构

```text
txt2/
  data/
    canteen_query_dataset.json
  src/
    build_correction_lexicon.py
    train_intent_classifier.py
    evaluate_intent_classifier.py
    evaluate_text_pipeline.py
    predict_intent.py
    text_processing.py
    test_time_ranges.py
  artifacts/
    correction_lexicon.json
    domain_classifier_hf/
    domain_classifier_int8.onnx
    domain_classifier_eval_metrics.json
    intent_eval_test.json
    intent_eval_all.json
    text_pipeline_eval_test.json
    text_pipeline_eval_all.json
    vocab.txt
  requirements.txt
  README.md
```

## 标签定义

| label | intent | 含义 |
|---:|---|---|
| 0 | out_domain | 不属于智慧食堂业务 |
| 1 | in_domain | 属于智慧食堂业务 |

## 本次训练方案

- 数据：`data/canteen_query_dataset.json`
- 基础模型：`uer/chinese_roberta_L-4_H-512`
- 训练集：800 条
- 测试集：200 条
- epoch：5
- batch size：32
- max length：64
- learning rate：2e-5
- 导出：HF 模型、ONNX、动态量化 int8 ONNX

这版分类器是从基础中文 RoBERTa 重新训练，不是从旧项目的 `artifacts/domain_classifier_hf` 继续训练。

## 完整效果

完整 pipeline 顺序：

```text
原始 query -> normalize -> query 纠错 -> 时间段提取 -> 意图分类
```

| 测评范围 | 样本数 | 意图分类准确率 | 时间段准确率 | query 纠错准确率 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| test | 200 | 100.00% | 100.00% | 85.50% | 120 | 80 | 0 | 0 |
| all | 1000 | 100.00% | 99.90% | 83.20% | 600 | 400 | 0 | 0 |

纯意图分类模型效果：

| 测评范围 | 样本数 | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| test | 200 | 100.00% | 1.0000 | 1.0000 |
| all | 1000 | 100.00% | 1.0000 | 1.0000 |

完整 pipeline 时延，包含 normalize、query 纠错、时间段提取和 ONNX int8 意图分类：

| 测评范围 | avg | p50 | p95 | p99 | max | <=50ms |
|---|---:|---:|---:|---:|---:|---:|
| test | 9.10ms | 8.93ms | 13.14ms | 17.25ms | 20.40ms | 100.00% |
| all | 7.17ms | 6.77ms | 9.91ms | 13.64ms | 35.00ms | 100.00% |

纯意图分类时延，只包含 ONNX int8 文本分类：

| 测评范围 | avg | p50 | p95 | p99 | max | <=50ms |
|---|---:|---:|---:|---:|---:|---:|
| test | 8.02ms | 7.04ms | 13.05ms | 16.36ms | 18.86ms | 100.00% |
| all | 6.15ms | 5.81ms | 8.85ms | 17.29ms | 22.06ms | 100.00% |

## 复现步骤

安装依赖：

```powershell
pip install -r requirements.txt
```

生成纠错词典：

```powershell
python src/build_correction_lexicon.py `
  --data data/canteen_query_dataset.json `
  --split train `
  --output artifacts/correction_lexicon.json
```

重新训练意图分类模型：

```powershell
python src/train_intent_classifier.py `
  --data data/canteen_query_dataset.json `
  --base-model uer/chinese_roberta_L-4_H-512 `
  --output-dir artifacts `
  --epochs 5 `
  --batch-size 32 `
  --max-length 64
```

纯意图分类测评：

```powershell
python src/evaluate_intent_classifier.py `
  --data data/canteen_query_dataset.json `
  --split test `
  --model artifacts/domain_classifier_int8.onnx `
  --tokenizer artifacts/domain_classifier_hf `
  --output artifacts/intent_eval_test.json
```

完整 pipeline 测评：

```powershell
python src/evaluate_text_pipeline.py `
  --data data/canteen_query_dataset.json `
  --split test `
  --model artifacts/domain_classifier_int8.onnx `
  --tokenizer artifacts/domain_classifier_hf `
  --correction-lexicon artifacts/correction_lexicon.json `
  --output artifacts/text_pipeline_eval_test.json
```

时间段规则回归测试：

```powershell
python src/test_time_ranges.py
```

## 单条预测

```powershell
python src/predict_intent.py "今添食唐消费统记"
```

输出会同时包含纠错、时间段和意图分类：

```json
{
  "original_query": "今添食唐消费统记",
  "corrected_query": "今天食堂消费统计",
  "time": {
    "raw": "今天",
    "granularity": "day",
    "start": "2026-05-08",
    "end": "2026-05-08"
  },
  "intent": "in_domain",
  "in_domain_score": 0.995695
}
```

## 交付说明

`txt2` 目录已经包含运行所需的数据、代码、纠错词典、训练后的模型和评估结果。打包给别人时保留整个 `txt2` 文件夹即可。

说明：仓库版保留 `domain_classifier_int8.onnx` 和 tokenizer，用于推理与评估。接近 100MB 的 fp32 ONNX 和 HF `model.safetensors` 未上传到 GitHub，可通过训练脚本重新生成。
