# 智慧食堂图像分类实验报告

## 任务

将图片分为三类：

| 标签 | 类别 |
|---:|---|
| 0 | 其他图片 |
| 1 | 食物图片 |
| 2 | 图表图片 |

## 数据集

数据目录：`D:\workspace\canteen\image_dataset_3class`

| Split | 其他(0) | 食物(1) | 图表(2) | 合计 |
|---|---:|---:|---:|---:|
| Train | 933 | 933 | 933 | 2799 |
| Test | 234 | 234 | 234 | 702 |
| Total | 1167 | 1167 | 1167 | 3501 |

数据来源：

- 食物类：`D:\workspace\canteen\images` 非图表图片，不足部分由 `E:\食品图片` 补充
- 图表类：`D:\workspace\canteen\images` 中图表相关目录
- 其他类：公开负样本数据集 `negative_public_images`

## 模型与训练

| 项目 | 配置 |
|---|---|
| 模型 | MobileNetV3-Small |
| 初始化 | ImageNet 预训练权重 |
| 输入尺寸 | 160x160 |
| 训练轮数 | 5 epochs |
| 最佳轮次 | epoch 4 |
| 设备 | CPU |

## 测试集结果

| 指标 | 结果 |
|---|---:|
| Accuracy | 98.15% |
| Macro Precision | 98.16% |
| Macro Recall | 98.15% |
| Macro F1 | 98.15% |
| Loss | 0.0565 |

### 分类别指标

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 其他(0) | 99.13% | 97.44% | 98.28% | 234 |
| 食物(1) | 97.46% | 98.29% | 97.87% | 234 |
| 图表(2) | 97.88% | 98.72% | 98.30% | 234 |

### 混淆矩阵

行是真实类别，列是预测类别。

|  | Pred 0 | Pred 1 | Pred 2 |
|---|---:|---:|---:|
| True 0 | 228 | 4 | 2 |
| True 1 | 1 | 230 | 3 |
| True 2 | 1 | 2 | 231 |

## 推理时延

CPU 单张推理，120 张样本。

| 指标 | 时延 |
|---|---:|
| Mean | 8.92 ms |
| P50 | 8.82 ms |
| P90 | 10.29 ms |
| P95 | 11.23 ms |
| P99 | 13.90 ms |

## 产物

- 模型权重：`D:\workspace\canteen\cv-2\outputs_mnv3_3class_160\best_model.pt`
- 评测报告：`D:\workspace\canteen\cv-2\outputs_mnv3_3class_160\final_3class_report.json`
- 训练脚本：`D:\workspace\canteen\cv-2\canteen_3class_experiment.py`
- 数据集清单：`D:\workspace\canteen\image_dataset_3class_manifest.json`

