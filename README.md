# Canteen Agent Gateway

智慧食堂前置网关生产化实验项目。目标是在线阶段低延迟完成：

- 文本业务相关性二分类
- 用户 query 领域纠错
- 日期级时间区间提取
- 轻量图像分类前置拒识
- 单条请求端到端时延统计

当前结论：文本 ONNX 网关满足 50ms；图像侧在不改训练数据、不改 MobileNetV3-Small 模型、不引入 OCR/VL 的前提下，通过 OpenCV bytes 解码、合并 resize/normalize、ONNX Runtime 单线程控制，`bytes -> preprocess -> model` 端到端 P95 已进入 50ms。

## Project Layout

```text
.
├── canteen_core.py
├── production_gateway.py
├── train_export_domain_classifier.py
├── generate_canteen_data.py
├── test_time_ranges.py
├── multimodal_dataset.py
├── run_image_mobilenetv3_experiment.py
├── run_mobilenetv3_onnx_opencv_benchmark.py
├── run_mobileclip_linear_probe.py
├── run_fastvlm_contrast.py
├── summarize_multimodal_results.py
├── requirements-prod.txt
├── canteen_test_data_300.json
├── ai_agent_records_202604280912.json
├── train/
└── artifacts/
```

## Text Gateway

生产入口：

```powershell
python production_gateway.py --data canteen_test_data_300.json
```

文本链路：

```text
query
 -> normalize
 -> domain correction
 -> date-range extraction
 -> ONNX INT8 binary classifier
 -> in_domain / out_domain
```

当前文本 ONNX 结果：

```text
domain_accuracy:     100.00% (300/300)
time_accuracy:       100.00% (300/300)
correction_accuracy: 100.00% (300/300)
p95_ms:              about 6-8ms
under_50ms:          100.00%
```

## Date Range Extraction

时间提取依赖在线当天日期，即 `date.today()`。输出只保留 `YYYY-MM-DD`，不输出时分秒。

示例，假设当天是 `2026-04-27`：

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

返回结构：

```json
{
  "raw": "上个月",
  "granularity": "month",
  "start": "2026-03-01",
  "end": "2026-03-31"
}
```

测试：

```powershell
python test_time_ranges.py
```

## Multimodal Data

默认标签映射：

```text
train/chart -> in_domain
train/food  -> in_domain
train/fruit -> out_domain
```

构建 manifest：

```powershell
python multimodal_dataset.py
```

当前统计：

```text
images: 94
texts: 404
image_positive: 86
image_negative: 8
text_positive: 284
text_negative: 120
```

## Image Experiment 1: MobileNetV3-Small

约束：

- 不改训练数据
- 不改 MobileNetV3-Small 模型
- 不引入 OCR
- 不引入 VL

原始 PyTorch/PIL 基准：

```powershell
python run_image_mobilenetv3_experiment.py --epochs 6 --image-size 160 --latency-repeats 100
```

结果：

```text
accuracy: 100.00%
model_only_preprocessed_tensor p95_ms: 7.97
end_to_end_image_open_preprocess_model p95_ms: 95.51
```

结论：模型本体足够快，瓶颈在图片打开、解码、resize、normalize。

## Image Experiment 1B: MobileNetV3-Small ONNX + OpenCV

本次优化只做工程侧：

- PIL -> OpenCV 解码
- path read -> bytes 输入
- resize / normalize 合并
- ONNX Runtime 线程控制
- 拆分统计 read / preprocess / model / total

运行 bytes 输入模式：

```powershell
python run_mobilenetv3_onnx_opencv_benchmark.py ^
  --input-mode bytes ^
  --intra-threads 1 ^
  --inter-threads 1 ^
  --cv-threads 1 ^
  --decode-reduction 2 ^
  --repeats 300
```

结果文件：

```text
artifacts/mobilenetv3_onnx_opencv_160_results.json
```

bytes 输入结果：

```text
accuracy: 100.00%

read:
p95_ms: 0.0023
max_ms: 0.0063

preprocess:
p50_ms: 1.4437
p95_ms: 41.0074
max_ms: 47.0709

model:
p50_ms: 1.7824
p95_ms: 2.6511
max_ms: 4.8596

total:
p50_ms: 3.7166
p95_ms: 43.1110
p99_ms: 44.6502
max_ms: 48.8557
under_50ms: 100.00%
```

运行 path 输入模式：

```powershell
python run_mobilenetv3_onnx_opencv_benchmark.py ^
  --input-mode path ^
  --intra-threads 1 ^
  --inter-threads 1 ^
  --cv-threads 1 ^
  --decode-reduction 2 ^
  --repeats 300 ^
  --result artifacts/mobilenetv3_onnx_opencv_160_path_results.json
```

path 输入结果：

```text
accuracy: 100.00%
total p95_ms: 46.0568
total p99_ms: 48.3117
total max_ms: 61.4676
under_50ms: 99.33%
```

结论：

- 线上服务应按 bytes 请求体作为推理入口统计，不建议把本地磁盘 path 读取纳入主链路。
- `bytes -> OpenCV preprocess -> ONNX model` 已满足 50ms 硬要求。
- path 模式 P95/P99 达标，但存在一次磁盘/系统尾延迟导致 max 超过 50ms。
- `decode-reduction=4` 会降低时延，但准确率从 100% 降到 96%，因此不作为默认。

## Image/Text Contrast: MobileCLIP2-S0

```powershell
python run_mobileclip_linear_probe.py --latency-repeats 40
```

结果：

```text
accuracy: 100.00%
model_only_preprocessed_tensor p95_ms: 107.94
end_to_end_image_open_preprocess_model p95_ms: 167.25
under_50ms: 0.00%
```

结论：MobileCLIP2-S0 可以做泛化/准确率对照，但 CPU 上不满足 50ms 主链路。

## VL Contrast: FastVLM-0.5B

```powershell
python run_fastvlm_contrast.py --sample-size 4 --trust-remote-code
```

结果：

```text
p50_ms: about 6000ms
p95_ms: about 6155ms
under_50ms: 0.00%
```

结论：FastVLM-0.5B 只作为 VL 上限对照，不进入 50ms 线上候选。

## Summary

```powershell
python summarize_multimodal_results.py
```

输出：

```text
artifacts/multimodal_experiment_summary.json
```

当前线上建议：

```text
文本主路径：ONNX INT8 text classifier
图像主路径：MobileNetV3-Small ONNX + OpenCV bytes pipeline
图文对照：MobileCLIP2-S0
VL 对照：FastVLM-0.5B
```

## Install

```powershell
python -m pip install -r requirements-prod.txt
```

Codex 桌面环境可使用：

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pip install -r requirements-prod.txt
```

## Push / Change Log

每次推送都需要在 README 记录修改点。

### 2026-04-29

- 新增 `run_mobilenetv3_onnx_opencv_benchmark.py`
- 导出 `artifacts/vision_mobilenetv3_small_160.onnx`
- 新增 OpenCV bytes 输入推理基准
- 将图片链路拆分统计为 `read / preprocess / model / total`
- 增加 ORT 线程控制参数：`intra_threads / inter_threads / execution_mode`
- 增加 OpenCV 线程控制参数：`cv_threads`
- 增加 JPEG reduced decode 参数：`decode_reduction`
- 证明 bytes 输入下 MobileNetV3-Small 端到端 P95 进入 50ms
- 新增结果：
  - `artifacts/mobilenetv3_onnx_opencv_160_results.json`
  - `artifacts/mobilenetv3_onnx_opencv_160_path_results.json`
  - `artifacts/mobilenetv3_onnx_opencv_160_path_reduced4_results.json`

### 2026-04-28

- 新增图像/图文实验数据构建脚本
- 新增 MobileNetV3-Small 图像分类实验
- 新增 MobileCLIP2-S0 frozen image encoder + linear probe 实验
- 新增 FastVLM-0.5B VL 对照实验
- 上传 `train/` 图片数据集和 `ai_agent_records_202604280912.json`

### 2026-04-27

- 建立生产级文本前置网关
- 新增 query 纠错、时间区间提取、文本 ONNX 二分类
- 支持日期级时间范围：日、周、月、季度、年
- 上传文本模型 ONNX / ONNX INT8 产物
