# Canteen Agent

This repository contains the current text + image intent recognition project for deciding whether a request belongs to the smart canteen business domain.

## Project Layout

```text
txt2/
  Text pipeline:
  - query correction
  - time range extraction
  - binary text intent classification
  - exported ONNX/int8 artifacts and evaluation reports

cv3/
  Image and paired text+image pipeline:
  - MobileNetV3-Small image 3-class classifier
  - image class to binary business intent mapping
  - paired text+image fusion evaluation
```

## Final Intent Labels

| label | intent | meaning |
|---:|---|---|
| 0 | `out_domain` | Not smart-canteen business |
| 1 | `in_domain` | Smart-canteen business |

## Text + Image Fusion

Each request contains both text and image.

```text
text -> txt2 text pipeline -> text_intent
image -> cv3 image classifier -> image_intent

final_intent = in_domain  if text_intent == in_domain or image_intent == in_domain
final_intent = out_domain if text_intent == out_domain and image_intent == out_domain
```

Image mapping:

| image class | binary intent |
|---|---|
| `0_other` | `out_domain` |
| `1_food` | `in_domain` |
| `2_chart` | `in_domain` |

## Current Evaluation

Text full pipeline on `txt2`:

| split | samples | intent accuracy | time accuracy | correction accuracy |
|---|---:|---:|---:|---:|
| test | 200 | 100.00% | 100.00% | 85.50% |
| all | 1000 | 100.00% | 99.90% | 83.20% |

Paired text+image fusion on `cv3`:

| target | samples | accuracy | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| text only | 200 | 100.00% | 120 | 80 | 0 | 0 |
| image only | 702 | 98.86% | 466 | 228 | 6 | 2 |
| paired fusion | 702 | 99.72% | 610 | 90 | 2 | 0 |

See `txt2/README.md` and `cv3/README.md` for commands and detailed latency metrics.

## Data Note

The repository includes the text dataset and the int8 ONNX model needed for current inference/evaluation. The larger fp32 ONNX export and HF model weight file are omitted because GitHub API upload rejects near-100MB blobs; run `txt2/src/train_intent_classifier.py` to regenerate them when needed.

The raw image dataset directory is not included because it is several GB; `cv3/image_dataset_3class_manifest.json` and the image preparation script document the dataset structure.
