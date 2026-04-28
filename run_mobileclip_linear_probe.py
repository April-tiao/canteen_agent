from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from canteen_core import percentile


os.environ.setdefault("TORCH_HOME", str(Path("artifacts/model_cache/torch").resolve()))
os.environ.setdefault("HF_HOME", str(Path("artifacts/model_cache/huggingface").resolve()))


def encode_images(
    model: torch.nn.Module,
    preprocess: Any,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> np.ndarray:
    features = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            image = preprocess(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)
            feature = model.encode_image(image)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            features.append(feature.squeeze(0).detach().cpu().numpy())
    return np.asarray(features)


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    preprocess: Any,
    probe: LogisticRegression,
    rows: list[dict[str, Any]],
    device: torch.device,
    repeats: int,
) -> list[float]:
    model.eval()
    latencies: list[float] = []
    sample_rows = (rows * ((repeats // len(rows)) + 1))[:repeats]

    for row in sample_rows[:10]:
        image = preprocess(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)
        feature = model.encode_image(image)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        probe.predict(feature.detach().cpu().numpy())

    for row in sample_rows:
        started = time.perf_counter_ns()
        image = preprocess(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)
        feature = model.encode_image(image)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        probe.predict(feature.detach().cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    return latencies


@torch.no_grad()
def measure_model_only_latency(
    model: torch.nn.Module,
    preprocess: Any,
    probe: LogisticRegression,
    rows: list[dict[str, Any]],
    device: torch.device,
    repeats: int,
) -> list[float]:
    model.eval()
    tensors = [preprocess(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device) for row in rows]
    sample_tensors = (tensors * ((repeats // len(tensors)) + 1))[:repeats]
    for tensor in sample_tensors[:10]:
        feature = model.encode_image(tensor)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        probe.predict(feature.detach().cpu().numpy())

    latencies: list[float] = []
    for tensor in sample_tensors:
        started = time.perf_counter_ns()
        feature = model.encode_image(tensor)
        feature = feature / feature.norm(dim=-1, keepdim=True)
        probe.predict(feature.detach().cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    return latencies


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "avg": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "under_50ms": sum(1 for value in values if value <= 50) / len(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/multimodal_manifest.json")
    parser.add_argument("--model-name", default="MobileCLIP2-S0")
    parser.add_argument("--pretrained", default="dfndr2b")
    parser.add_argument("--result", default="artifacts/mobileclip2_s0_linear_probe_results.json")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--latency-repeats", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError("Install open_clip_torch before running MobileCLIP2 experiments.") from exc

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["images"]
    labels = [row["label"] for row in rows]
    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    for param in model.parameters():
        param.requires_grad = False

    train_x = encode_images(model, preprocess, train_rows, device)
    test_x = encode_images(model, preprocess, test_rows, device)
    train_y = np.asarray([row["label"] for row in train_rows])
    test_y = np.asarray([row["label"] for row in test_rows])

    probe = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    probe.fit(train_x, train_y)
    preds = probe.predict(test_x)
    end_to_end_latencies = measure_latency(model, preprocess, probe, test_rows, device, args.latency_repeats)
    model_only_latencies = measure_model_only_latency(model, preprocess, probe, test_rows, device, args.latency_repeats)

    result = {
        "scheme": "MobileCLIP2-S0 frozen image encoder + linear probe",
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "device": str(device),
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "accuracy": accuracy_score(test_y, preds),
        "classification_report": classification_report(test_y, preds, target_names=["out_domain", "in_domain"], output_dict=True),
        "latency_ms": {
            "end_to_end_image_open_preprocess_model": latency_summary(end_to_end_latencies),
            "model_only_preprocessed_tensor": latency_summary(model_only_latencies),
        },
    }
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
