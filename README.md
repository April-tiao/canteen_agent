# Canteen Agent Gateway

智慧食堂前置网关生产化示例，在线阶段只做低延迟推理：

- 食堂业务相关性二分类
- 用户 query 领域纠错
- 日期级时间区间提取
- 单条 query 端到端时延统计

当前线上入口是 `production_gateway.py`，二分类默认加载离线训练并量化好的 ONNX INT8 模型。

## 项目结构

```text
.
├── canteen_core.py                       # 生产共享逻辑：纠错、时间区间、规则 fallback、统计工具
├── production_gateway.py                 # 生产推理入口：ONNX 二分类 + 纠错 + 时间区间
├── train_export_domain_classifier.py     # 离线训练并导出 ONNX / ONNX INT8
├── generate_canteen_data.py              # 生成 300 条测试数据
├── test_time_ranges.py                   # 时间区间规则测试
├── train_and_eval.ps1                    # Windows 一键训练与评测脚本
├── requirements-prod.txt                 # 训练和推理依赖
├── canteen_test_data_300.json            # 300 条测试数据
└── artifacts/
    ├── domain_classifier_int8.onnx       # 生产默认加载的 INT8 二分类模型
    ├── domain_classifier.onnx            # 未量化 ONNX，便于对比评测
    ├── vocab.txt                         # 词表备份
    └── domain_classifier_hf/             # tokenizer 配置目录
```

## 在线链路

```text
用户 query
  -> normalize
  -> 领域 query 纠错
  -> 日期级时间区间提取
  -> ONNX INT8 二分类模型
  -> in_domain / out_domain
```

生产入口不会实时训练模型。模型训练、导出、量化都在离线阶段完成。

## 二分类模型

当前训练脚本默认使用：

```text
uer/chinese_roberta_L-4_H-512
```

离线训练后导出：

```text
artifacts/domain_classifier.onnx
artifacts/domain_classifier_int8.onnx
artifacts/domain_classifier_hf/
```

在线推理默认使用：

```text
artifacts/domain_classifier_int8.onnx
artifacts/domain_classifier_hf/
```

## 时间区间输出

时间提取依赖在线当天日期，即 `date.today()`。输出只保留年月日，不输出时分秒。

统一输出格式：

```json
{
  "raw": "上个月",
  "granularity": "month",
  "start": "2026-03-01",
  "end": "2026-03-31"
}
```

以当天为 `2026-04-27` 为例：

```text
今天       -> 2026-04-27 - 2026-04-27
昨天晚餐   -> 2026-04-26 - 2026-04-26
上周       -> 2026-04-20 - 2026-04-26
本周午餐   -> 2026-04-27 - 2026-05-03
上个月     -> 2026-03-01 - 2026-03-31
本月       -> 2026-04-01 - 2026-04-30
上个季度   -> 2026-01-01 - 2026-03-31
本季度     -> 2026-04-01 - 2026-06-30
去年       -> 2025-01-01 - 2025-12-31
今年       -> 2026-01-01 - 2026-12-31
```

餐段词如“早餐、午餐、晚餐”只影响 `granularity` 和 `meal_period`，日期仍然输出区间：

```json
{
  "raw": "本周午餐",
  "granularity": "meal_period",
  "meal_period": "lunch",
  "start": "2026-04-27",
  "end": "2026-05-03"
}
```

## 数据格式

`canteen_test_data_300.json` 每条数据：

```json
{
  "q": "今天中午二食堂有宫爆鸡丁吗",
  "domain": "in_domain",
  "time": true,
  "corrected": "今天中午二食堂有宫保鸡丁吗"
}
```

字段说明：

- `q`：原始 query
- `domain`：期望二分类标签，`in_domain` 或 `out_domain`
- `time`：是否期望抽取到时间
- `corrected`：期望纠错后的 query

## 安装依赖

推荐 Python 3.10+。

```powershell
python -m pip install -r requirements-prod.txt
```

在 Codex 桌面环境中可以使用内置 Python：

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install -r requirements-prod.txt
```

## 离线训练与导出

```powershell
python train_export_domain_classifier.py --data canteen_test_data_300.json
```

训练脚本会：

1. 读取测试数据
2. 使用 `corrected` 文本训练二分类模型
3. 输出验证集指标
4. 导出 ONNX
5. 量化为 ONNX INT8

## 生产推理评测

```powershell
python production_gateway.py --data canteen_test_data_300.json
```

本地没有模型产物时可以做规则 fallback 烟测：

```powershell
python production_gateway.py --data canteen_test_data_300.json --allow-rule-fallback
```

`--allow-rule-fallback` 不能用于生产。

## 一键运行

Windows PowerShell：

```powershell
.\train_and_eval.ps1
```

## 生产接口示例

```python
from pathlib import Path

from production_gateway import OnnxBertBinaryClassifier, ProductionCanteenGateway

classifier = OnnxBertBinaryClassifier(
    Path("artifacts/domain_classifier_int8.onnx"),
    Path("artifacts/domain_classifier_hf"),
)
gateway = ProductionCanteenGateway(classifier, threshold=0.5)

result = gateway.handle("上个月饭卡消费明细")
print(result.domain)
print(result.time)
```

返回核心字段：

```text
domain              # in_domain / out_domain
domain_score        # in-domain 概率
original_query      # 原始 query
corrected_query     # 纠错后的 query
correction_applied  # 是否发生纠错
corrections         # 纠错明细
time                # 日期级时间区间，未命中则为 None
latency_ms          # 单条端到端时延
```

## 测试

时间区间测试：

```powershell
python test_time_ranges.py
```

生产入口评测：

```powershell
python production_gateway.py --data canteen_test_data_300.json
```

当前 300 条测试数据的 ONNX INT8 生产入口结果：

```text
domain_accuracy:     100.00% (300/300)
time_accuracy:       100.00% (300/300)
correction_accuracy: 100.00% (300/300)
p50_ms: 5.3387
p95_ms: 6.0502
p99_ms: 6.4429
under_50ms: 100.00%
```

这些时延是单条 query 的端到端耗时分布，不是总耗时。

## 上线注意

`canteen_test_data_300.json` 是测试数据，只用于验证工程链路和时延。正式上线请使用真实标注数据重新训练并替换 `artifacts/` 下的模型。

建议上线门槛：

```text
domain_accuracy >= 90%
in_domain_recall >= 90%
out_domain_precision >= 90%
p95_latency <= 50ms
```

如果加入图像分类、OCR 或 VLM，请将视觉模型作为离线训练、在线推理模块接入，并单独统计视觉推理耗时。

## 图像/图文实验

新增实验数据：

```text
train/chart  -> 默认 in_domain
train/food   -> 默认 in_domain
train/fruit  -> 默认 out_domain
ai_agent_records_202604280912.json -> 文本正样本
canteen_test_data_300.json         -> 文本负样本来源
```

构建统一 manifest：

```powershell
python multimodal_dataset.py
```

默认输出：

```text
artifacts/multimodal_manifest.json
```

当前 manifest 统计：

```text
images: 94
texts: 404
image_positive: 86
image_negative: 8
text_positive: 284
text_negative: 120
```

### 方案一：MobileNetV3-Small

轻量图像分类候选，最适合 50ms 线上路径。

```powershell
python run_image_mobilenetv3_experiment.py --epochs 6 --image-size 160 --latency-repeats 100
```

当前 CPU 结果：

```text
accuracy: 100.00%

end_to_end_image_open_preprocess_model:
p50_ms: 10.79
p95_ms: 95.51
under_50ms: 92.00%

model_only_preprocessed_tensor:
p50_ms: 6.79
p95_ms: 7.97
under_50ms: 100.00%
```

结论：模型本体满足 50ms；端到端尾延迟主要来自图片打开、解码和预处理。线上若必须 50ms，需要优化图片输入链路，例如固定尺寸上传、服务侧高性能解码、预处理线程池、ONNX/TensorRT/OpenVINO。

### 方案二：MobileCLIP2-S0 Frozen Image Encoder + Linear Probe

图文双塔小模型对照方案，当前只冻结 image encoder，再训练线性分类器。

```powershell
python run_mobileclip_linear_probe.py --latency-repeats 40
```

当前 CPU 结果：

```text
accuracy: 100.00%

end_to_end_image_open_preprocess_model:
p50_ms: 79.98
p95_ms: 167.25
under_50ms: 0.00%

model_only_preprocessed_tensor:
p50_ms: 77.78
p95_ms: 107.94
under_50ms: 0.00%
```

结论：MobileCLIP2-S0 准确率可以作为对照，但 CPU 推理不满足 50ms。若要上线，需要 GPU、ONNX 优化或更小的图像 encoder。

### 方案三：FastVLM-0.5B + 二分类头对照组

FastVLM 只作为 VL 小模型准确率上限/统一图文入口对照，不作为 50ms 线上候选。

当前脚本：

```powershell
python run_fastvlm_contrast.py --sample-size 4 --trust-remote-code
```

当前 CPU 极小样本结果：

```text
p50_ms: 6000.61
p95_ms: 6155.43
under_50ms: 0.00%
```

结论：CPU 上完全不满足 50ms。后续如果要严肃评估准确率上限，应离线训练 FastVLM hidden state 的二分类头，而不是让模型生成 `in_domain/out_domain` 字符串。

### 汇总结果

```powershell
python summarize_multimodal_results.py
```

输出：

```text
artifacts/multimodal_experiment_summary.json
```

当前线上建议：

```text
50ms 主路径：文本 ONNX + MobileNetV3-Small 图像分类
准确率/泛化对照：MobileCLIP2-S0
VL 上限对照：FastVLM-0.5B
```
