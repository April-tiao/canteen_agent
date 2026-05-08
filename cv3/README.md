# 智慧食堂图像三分类

本项目使用 MobileNetV3-Small 做图像三分类，用于判断图片属于：

| 标签 | 类别 |
|---:|---|
| 0 | 其他图片 |
| 1 | 食物图片 |
| 2 | 图表图片 |

## 数据结构

默认数据目录名为 `image_dataset_3class`：

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

脚本会自动查找：

```text
./data/image_dataset_3class
../data/image_dataset_3class
./image_dataset_3class
../image_dataset_3class
/XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/image_dataset_3class
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`，跳过 `.gif`。

## 安装

```bash
conda create -n canteen-cv python=3.10 -y
conda activate canteen-cv
pip install -r requirements.txt
```

如需指定 CUDA 版本，请先安装服务器匹配的 `torch` / `torchvision`，再安装其余依赖。

## 使用

检查数据：

```bash
python canteen_3class_experiment.py scan
```

训练：

```bash
python canteen_3class_experiment.py train --epochs 5 --batch-size 64 --image-size 160
```

评估：

```bash
python canteen_3class_experiment.py evaluate --checkpoint outputs_3class/best_model.pt
```

测速：

```bash
python canteen_3class_experiment.py latency --checkpoint outputs_3class/best_model.pt
```

显式指定数据路径：

```bash
python canteen_3class_experiment.py scan --data-dir /path/to/image_dataset_3class
```

## 本地实验结果

数据集：3501 张，三类均衡，每类 1167 张。  
训练集：2799 张；测试集：702 张。

MobileNetV3-Small，ImageNet 预训练，输入 `160x160`，训练 5 epochs。

| 指标 | 结果 |
|---|---:|
| Accuracy | 98.15% |
| Macro Precision | 98.16% |
| Macro Recall | 98.15% |
| Macro F1 | 98.15% |

CPU 单张推理时延：

| 指标 | 时延 |
|---|---:|
| Mean | 8.92 ms |
| P50 | 8.82 ms |
| P95 | 11.23 ms |
| P99 | 13.90 ms |

## 文本 + 图像意图识别融合

总目标：一次请求同时拿到文本和图像，判断这组图文是否属于智慧食堂业务。

融合口径：

| 模态 | 原始类别 | 统一意图 |
|---|---|---|
| 文本 | `label=0` | `out_domain` |
| 文本 | `label=1` | `in_domain` |
| 图像 | `0_other` | `out_domain` |
| 图像 | `1_food` | `in_domain` |
| 图像 | `2_chart` | `in_domain` |

图文同时输入时的最终决策：

```text
文本模型结果 = text_intent
图像模型结果 = image_intent

final_intent = in_domain  if text_intent == in_domain or image_intent == in_domain
final_intent = out_domain if text_intent == out_domain and image_intent == out_domain
```

也就是：文本或图片任一模态出现智慧食堂业务证据，整条图文请求就判断为 `in_domain`。

融合脚本：

```powershell
python text_image_intent_fusion.py `
  --text-project ../canteen_agent-main/txt2 `
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

当前没有真实人工标注的图文成对 manifest，因此本次使用 `cycle` 方式构造 702 个图文对：图像 test 702 张全部参与，文本 test 200 条循环配对。成对标签按上述 OR 规则生成。

图文同时输入融合测试结果：

| 测试对象 | 样本 | 准确率 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 文本单模态 | 200 | 100.00% | 120 | 80 | 0 | 0 |
| 图像单模态 | 702 | 98.86% | 466 | 228 | 6 | 2 |
| 图文成对融合 | 702 | 99.72% | 610 | 90 | 2 | 0 |

时延结果：

| 项目 | avg | p50 | p95 | p99 | max | <=50ms |
|---|---:|---:|---:|---:|---:|---:|
| 文本 pipeline | 6.26ms | 6.20ms | 7.67ms | 8.32ms | 9.40ms | 100.00% |
| 图像端到端 | 60.28ms | 16.31ms | 191.89ms | 526.40ms | 1457.61ms | 73.65% |
| 图像模型推理 | 9.02ms | 8.95ms | 10.46ms | 13.35ms | 22.90ms | 100.00% |
| 图文串行端到端 | 70.77ms | 27.18ms | 209.04ms | 548.59ms | 1384.81ms | 70.09% |
| 图文并行估算 | 63.43ms | 19.95ms | 201.54ms | 540.91ms | 1377.04ms | 72.65% |

说明：图像端到端时延包含图片文件读取、PIL 解码、resize、normalize 和模型推理。测试集中存在超大图片，PIL 触发 `DecompressionBombWarning`，因此端到端 max/p99 有明显长尾；图像模型本体推理稳定在约 9.02ms。图文串行端到端表示先跑文本再跑图片；图文并行估算表示文本和图片同时推理，整条请求耗时近似取两者最大值。

## 图文实时链路优化

新增优化参数：

| 参数 | 作用 |
|---|---|
| `--image-backend opencv` | 用 OpenCV bytes 解码替代 PIL |
| `--decode-reduction 4` | JPEG 降采样解码，减少超大图解码成本 |
| `--cv-threads 1` | 控制 OpenCV 线程数，降低调度抖动 |
| `--max-image-bytes` | 限制图片文件大小 |
| `--max-image-pixels` | 限制图片总像素 |

OpenCV reduced4 结果：

| 测试对象 | 样本 | 准确率 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 图像单模态 | 702 | 98.15% | 465 | 224 | 10 | 3 |
| 图文成对融合 | 702 | 99.29% | 610 | 87 | 5 | 0 |

OpenCV reduced4 时延：

| 项目 | avg | p50 | p95 | p99 | max | <=50ms |
|---|---:|---:|---:|---:|---:|---:|
| 文本 pipeline | 6.25ms | 6.22ms | 7.29ms | 8.62ms | 9.26ms | 100.00% |
| 图像端到端 | 28.68ms | 11.08ms | 88.63ms | 306.43ms | 528.38ms | 84.05% |
| 图像模型推理 | 8.33ms | 8.39ms | 9.47ms | 10.51ms | 12.71ms | 100.00% |
| 图文真实并行 | 39.05ms | 18.18ms | 115.33ms | 305.55ms | 529.53ms | 75.36% |

结论：OpenCV reduced4 已将图文真实并行平均时延降到 39.05ms，但 p95/p99 仍受超大图片解码影响。线上若要求完整端到端稳定 50ms，需要在入口限制图片大小/像素数，或要求客户端上传前压缩图片。
