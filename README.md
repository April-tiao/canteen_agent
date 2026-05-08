# 智慧食堂图文意图识别项目

本项目用于判断一条“文本 + 图片”请求是否属于智慧食堂业务域。

最终业务输出是二分类：

| label | intent | 含义 |
|---:|---|---|
| 0 | `out_domain` | 不属于智慧食堂业务 |
| 1 | `in_domain` | 属于智慧食堂业务 |

## 1. 项目目标

输入是一条同时包含文本和图片的请求：

```json
{
  "query": "今添食唐消费统记",
  "image": "图片文件或图片二进制"
}
```

系统内部同时执行两条链路：

```text
文本 query
  -> query 纠错
  -> 时间段提取
  -> 文本意图二分类
  -> text_intent

图片 image
  -> 图像三分类模型
  -> 0_other / 1_food / 2_chart
  -> 映射为 image_intent
```

图文融合规则：

```text
final_intent = in_domain
  if text_intent == in_domain or image_intent == in_domain

final_intent = out_domain
  if text_intent == out_domain and image_intent == out_domain
```

也就是：文本或图片任一模态出现智慧食堂业务证据，整条请求就判定为 `in_domain`。

图像类别到业务意图的映射：

| 图像三分类类别 | 业务意图 |
|---|---|
| `0_other` | `out_domain` |
| `1_food` | `in_domain` |
| `2_chart` | `in_domain` |

## 2. 完整目录结构

```text
canteen_agent/
  README.md
  .gitignore

  txt2/
    README.md
    requirements.txt

    data/
      canteen_query_dataset.json

    src/
      build_correction_lexicon.py
      train_intent_classifier.py
      evaluate_intent_classifier.py
      evaluate_text_pipeline.py
      predict_intent.py
      test_time_ranges.py
      text_processing.py

    artifacts/
      correction_lexicon.json
      domain_classifier_int8.onnx
      vocab.txt
      domain_classifier_eval_metrics.json
      intent_eval_test.json
      intent_eval_all.json
      text_pipeline_eval_test.json
      text_pipeline_eval_all.json

      domain_classifier_hf/
        config.json
        special_tokens_map.json
        tokenizer.json
        tokenizer_config.json
        vocab.txt

  cv3/
    README.md
    EXPERIMENT_USAGE.md
    image_classification_report.md
    requirements.txt

    prepare_image_dataset_3class.py
    canteen_3class_experiment.py
    text_image_intent_fusion.py
    image_dataset_3class_manifest.json

    outputs_mnv3_3class_160/
      best_model.pt
      final_3class_report.json
      paired_text_image_intent_fusion_eval.json
      paired_text_image_intent_fusion_eval_opencv_reduced4.json
      text_image_intent_fusion_eval.json
```

说明：

- `txt2/` 是文本侧项目，包含 query 纠错、时间段提取、文本二分类模型、评估脚本和结果。
- `cv3/` 是图像侧和图文融合项目，包含图像三分类模型、图像类别映射、图文成对融合评估脚本和结果。
- 原始图像数据集 `cv3/image_dataset_3class/` 未上传，因为体积约数 GB；仓库保留了 manifest 和数据准备脚本。
- 文本侧保留可直接推理的 `domain_classifier_int8.onnx`。接近 100MB 的 fp32 ONNX 和 HF `model.safetensors` 未上传，可通过训练脚本重新生成。

## 3. 环境安装

建议 Python 3.10+。

文本侧依赖：

```powershell
cd txt2
pip install -r requirements.txt
```

图像侧依赖：

```powershell
cd cv3
pip install -r requirements.txt
```

如果在 Codex 桌面环境中运行，也可以使用内置 Python：

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install -r txt2/requirements.txt
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install -r cv3/requirements.txt
```

## 4. 文本侧可复现步骤

### 4.1 生成 query 纠错词典

纠错词典由新数据 train split 中的 `corrections` 标注生成。

```powershell
cd txt2
python src/build_correction_lexicon.py `
  --data data/canteen_query_dataset.json `
  --split train `
  --output artifacts/correction_lexicon.json
```

输出：

```text
artifacts/correction_lexicon.json
```

### 4.2 重新训练文本意图分类模型

本次文本分类模型从基础模型 `uer/chinese_roberta_L-4_H-512` 重新训练，不是沿用旧模型。

```powershell
cd txt2
python src/train_intent_classifier.py `
  --data data/canteen_query_dataset.json `
  --base-model uer/chinese_roberta_L-4_H-512 `
  --output-dir artifacts `
  --epochs 5 `
  --batch-size 32 `
  --max-length 64
```

训练输出：

```text
artifacts/domain_classifier_hf/
artifacts/domain_classifier.onnx
artifacts/domain_classifier_int8.onnx
artifacts/domain_classifier_eval_metrics.json
```

仓库中已保留 `domain_classifier_int8.onnx` 和 tokenizer，可直接用于推理和评估。

### 4.3 纯文本意图分类评估

```powershell
cd txt2
python src/evaluate_intent_classifier.py `
  --data data/canteen_query_dataset.json `
  --split test `
  --model artifacts/domain_classifier_int8.onnx `
  --tokenizer artifacts/domain_classifier_hf `
  --output artifacts/intent_eval_test.json
```

全量评估：

```powershell
python src/evaluate_intent_classifier.py `
  --data data/canteen_query_dataset.json `
  --split all `
  --model artifacts/domain_classifier_int8.onnx `
  --tokenizer artifacts/domain_classifier_hf `
  --output artifacts/intent_eval_all.json
```

### 4.4 完整文本 pipeline 评估

完整文本 pipeline 包含：

```text
normalize -> query 纠错 -> 时间段提取 -> 文本意图分类
```

测试集：

```powershell
cd txt2
python src/evaluate_text_pipeline.py `
  --data data/canteen_query_dataset.json `
  --split test `
  --model artifacts/domain_classifier_int8.onnx `
  --tokenizer artifacts/domain_classifier_hf `
  --correction-lexicon artifacts/correction_lexicon.json `
  --output artifacts/text_pipeline_eval_test.json
```

全量：

```powershell
python src/evaluate_text_pipeline.py `
  --data data/canteen_query_dataset.json `
  --split all `
  --model artifacts/domain_classifier_int8.onnx `
  --tokenizer artifacts/domain_classifier_hf `
  --correction-lexicon artifacts/correction_lexicon.json `
  --output artifacts/text_pipeline_eval_all.json
```

### 4.5 单条文本预测

```powershell
cd txt2
python src/predict_intent.py "今添食唐消费统记"
```

示例输出：

```json
{
  "original_query": "今添食唐消费统记",
  "normalized_query": "今添食唐消费统记",
  "corrected_query": "今天食堂消费统计",
  "time": {
    "raw": "今天",
    "granularity": "day",
    "start": "2026-05-08",
    "end": "2026-05-08"
  },
  "label": 1,
  "intent": "in_domain",
  "in_domain_score": 0.995695
}
```

## 5. 图像侧可复现步骤

### 5.1 图像数据结构

图像三分类数据目录应为：

```text
image_dataset_3class/
  train/
    0_other/
    1_food/
    2_chart/
  test/
    0_other/
    1_food/
    2_chart/
```

图像标签：

| label | class | 含义 |
|---:|---|---|
| 0 | `0_other` | 其他图片，不属于智慧食堂业务 |
| 1 | `1_food` | 食物图片，属于智慧食堂业务 |
| 2 | `2_chart` | 图表图片，属于智慧食堂业务 |

### 5.2 扫描数据集

```powershell
cd cv3
python canteen_3class_experiment.py --data-dir image_dataset_3class scan
```

### 5.3 重新训练图像三分类模型

```powershell
cd cv3
python canteen_3class_experiment.py `
  --data-dir image_dataset_3class `
  train `
  --epochs 5 `
  --batch-size 64 `
  --image-size 160 `
  --output-dir outputs_mnv3_3class_160
```

输出：

```text
outputs_mnv3_3class_160/best_model.pt
outputs_mnv3_3class_160/metrics.json
```

仓库中已保留当前最佳模型：

```text
cv3/outputs_mnv3_3class_160/best_model.pt
```

### 5.4 图像三分类评估

```powershell
cd cv3
python canteen_3class_experiment.py `
  --data-dir image_dataset_3class `
  evaluate `
  --checkpoint outputs_mnv3_3class_160/best_model.pt `
  --image-size 160
```

### 5.5 图像推理时延评估

```powershell
cd cv3
python canteen_3class_experiment.py `
  --data-dir image_dataset_3class `
  latency `
  --checkpoint outputs_mnv3_3class_160/best_model.pt `
  --samples 120 `
  --image-size 160
```

## 6. 图文成对融合可复现步骤

当前任务设定为：一次请求同时包含文本和图片。

融合脚本：

```text
cv3/text_image_intent_fusion.py
```

运行：

```powershell
cd cv3
python text_image_intent_fusion.py `
  --text-project ../txt2 `
  --text-split test `
  --image-data-dir image_dataset_3class `
  --image-checkpoint outputs_mnv3_3class_160/best_model.pt `
  --image-backend opencv `
  --decode-reduction 4 `
  --cv-threads 1 `
  --max-image-bytes 100000000 `
  --max-image-pixels 200000000 `
  --pair-mode cycle `
  --output outputs_mnv3_3class_160/paired_text_image_intent_fusion_eval_opencv_reduced4.json
```

说明：

- 当前没有人工标注的真实图文成对 manifest。
- 本次使用 `cycle` 方式构造图文对：图像 test 702 张全部参与，文本 test 200 条循环配对。
- 成对标签按 OR 规则生成：文本或图片任一为 `in_domain`，整条图文请求即为 `in_domain`。

如果以后有真实图文成对标注，可以使用：

```powershell
python text_image_intent_fusion.py `
  --pair-mode manifest `
  --pair-manifest path/to/pair_manifest.json
```

## 7. 测评结果

### 7.1 文本侧完整 pipeline

| split | 样本数 | 意图分类准确率 | 时间段准确率 | query 纠错准确率 |
|---|---:|---:|---:|---:|
| test | 200 | 100.00% | 100.00% | 85.50% |
| all | 1000 | 100.00% | 99.90% | 83.20% |

文本完整 pipeline 时延：

| split | avg | p50 | p95 | p99 | max | <=50ms |
|---|---:|---:|---:|---:|---:|---:|
| test | 9.10ms | 8.93ms | 13.14ms | 17.25ms | 20.40ms | 100.00% |
| all | 7.17ms | 6.77ms | 9.91ms | 13.64ms | 35.00ms | 100.00% |

### 7.2 纯文本意图分类

| split | 样本数 | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| test | 200 | 100.00% | 1.0000 | 1.0000 |
| all | 1000 | 100.00% | 1.0000 | 1.0000 |

### 7.3 图像三分类

图像模型：MobileNetV3-Small  
输入尺寸：`160x160`  
训练轮数：5 epochs  
测试集：702 张，三类均衡，每类 234 张。

| 指标 | 结果 |
|---|---:|
| Accuracy | 98.15% |
| Macro Precision | 98.16% |
| Macro Recall | 98.15% |
| Macro F1 | 98.15% |

三分类混淆矩阵：

```text
[
  [228, 4,   2],
  [1,   230, 3],
  [1,   2,   231]
]
```

图像模型推理时延：

| 指标 | 时延 |
|---|---:|
| Mean | 8.92ms |
| P50 | 8.82ms |
| P95 | 11.23ms |
| P99 | 13.90ms |

### 7.4 图文成对融合

图文融合二分类结果：

| 测试对象 | 样本数 | 准确率 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 文本单模态 | 200 | 100.00% | 120 | 80 | 0 | 0 |
| 图像单模态，PIL 基线 | 702 | 98.86% | 466 | 228 | 6 | 2 |
| 图文成对融合，PIL 基线 | 702 | 99.72% | 610 | 90 | 2 | 0 |
| 图像单模态，OpenCV reduced4 | 702 | 98.15% | 465 | 224 | 10 | 3 |
| 图文成对融合，OpenCV reduced4 | 702 | 99.29% | 610 | 87 | 5 | 0 |

PIL 基线图文融合时延：

| 项目 | avg | p50 | p95 | p99 | max | <=50ms |
|---|---:|---:|---:|---:|---:|---:|
| 文本 pipeline | 6.26ms | 6.20ms | 7.67ms | 8.32ms | 9.40ms | 100.00% |
| 图像端到端 | 60.28ms | 16.31ms | 191.89ms | 526.40ms | 1457.61ms | 73.65% |
| 图像模型推理 | 9.02ms | 8.95ms | 10.46ms | 13.35ms | 22.90ms | 100.00% |
| 图文串行端到端 | 70.77ms | 27.18ms | 209.04ms | 548.59ms | 1384.81ms | 70.09% |
| 图文并行估算 | 63.43ms | 19.95ms | 201.54ms | 540.91ms | 1377.04ms | 72.65% |

OpenCV reduced4 优化后图文融合时延：

| 项目 | avg | p50 | p95 | p99 | max | <=50ms |
|---|---:|---:|---:|---:|---:|---:|
| 文本 pipeline | 6.25ms | 6.22ms | 7.29ms | 8.62ms | 9.26ms | 100.00% |
| 图像端到端 | 28.68ms | 11.08ms | 88.63ms | 306.43ms | 528.38ms | 84.05% |
| 图像模型推理 | 8.33ms | 8.39ms | 9.47ms | 10.51ms | 12.71ms | 100.00% |
| 图文串行端到端 | 43.88ms | 21.26ms | 131.18ms | 335.22ms | 549.29ms | 73.65% |
| 图文并行估算 | 37.21ms | 14.86ms | 121.79ms | 328.48ms | 542.49ms | 76.78% |
| 图文真实并行 | 39.05ms | 18.18ms | 115.33ms | 305.55ms | 529.53ms | 75.36% |

说明：

- 图文串行端到端表示先跑文本再跑图片。
- 图文并行估算表示文本和图片同时推理，整条请求耗时近似取两者最大值。
- 图像端到端时延包含文件读取、PIL 解码、resize、normalize 和模型推理。
- 测试集中存在超大图片，PIL 触发过 `DecompressionBombWarning`，因此图像端到端 p99/max 有明显长尾。
- 图像模型本体推理稳定在 10ms 左右，主要耗时来自图片读取、解码和预处理。
- OpenCV reduced4 优化将图文真实并行平均时延降到 39.05ms，但 p95/p99 仍受超大图片解码影响。
- 若线上必须保证完整端到端 50ms，需要配合输入约束，例如文件大小不超过 5-8MB、总像素不超过 2M-4M、客户端上传前压缩最长边。

## 8. 关键实现说明

### 8.1 query 纠错

当前 query 纠错是词典替换法：

```text
从 train split 的 corrections 标注中提取 wrong -> correct
-> 生成 correction_lexicon.json
-> 运行时按词典做字符串替换
```

示例：

```text
今添食唐消费统记
-> 今天食堂消费统计
```

### 8.2 时间段提取

当前时间段提取是增强规则法，不是模型。

已覆盖：

- 今天、昨天、明天
- 上周、本周、下周
- 上一个星期、上一周、前一周
- 上个月、上一个月、前一个月
- 最近一周、最近一个月、过去 N 天、近 N 周、近 N 个月
- 本季度、上季度、下季度
- 显式年月，如 `2026年1月`
- 显式季度，如 `2026年第一季度`
- 一句话多个时间表达时，默认取 query 中最早出现的时间表达

长期泛化建议：升级为“时间 span 识别模型 + 日期归一化规则”。

### 8.3 图文融合

当前不是一个单独的多模态大模型，而是：

```text
文本模型 + 图像模型 + 规则融合器
```

优点：

- 推理快
- 可解释
- 可单独替换文本或图像模型
- 方便做端到端时延拆分

### 8.4 实时图文并行优化

本次新增的优化均在 `cv3/text_image_intent_fusion.py` 中：

- `--image-backend opencv`：用 OpenCV bytes 解码替代 PIL。
- `--decode-reduction 4`：JPEG 降采样解码，减少超大图解码和 resize 成本。
- `--cv-threads 1`：控制 OpenCV 线程，减少线程调度抖动。
- `--max-image-bytes`：限制图片文件大小。
- `--max-image-pixels`：限制图片总像素。
- 真实图文并行计时：用两个线程同时执行文本和图片链路。

推荐实时服务方向：

```text
HTTP image bytes 输入，不落盘
-> 图片大小/像素数校验
-> OpenCV/turbojpeg 解码
-> resize/normalize 快路径
-> 文本和图片真正并行推理
-> OR 融合输出
```

当前代码已经支持上述核心评估逻辑。若要 p95/p99 稳定进入 50ms，必须在真实服务入口限制超大图，否则单靠模型优化无法消除图片解码长尾。

## 9. 注意事项

1. 原始图像数据集没有上传到仓库，因为体积过大。
2. 文本 fp32 ONNX 和 HF `model.safetensors` 没有上传，因为 GitHub API 不接受接近 100MB 的 blob。
3. 仓库中保留了当前可直接推理和评估所需的 int8 ONNX、tokenizer、图像模型和评估结果。
4. 如需完整重训文本模型，运行 `txt2/src/train_intent_classifier.py`。
5. 如需完整重训图像模型，需要先准备 `cv3/image_dataset_3class/` 数据目录。
