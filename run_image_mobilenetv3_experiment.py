from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from canteen_core import percentile


os.environ.setdefault("TORCH_HOME", str(Path("artifacts/model_cache/torch").resolve()))


class ImageManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: transforms.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), torch.tensor(row["label"], dtype=torch.long)


def build_model(num_classes: int = 2, freeze_features: bool = True) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} train_loss={total_loss / max(len(train_loader), 1):.4f}")


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    preds: list[int] = []
    labels_out: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device)).detach().cpu()
        preds.extend(torch.argmax(logits, dim=1).tolist())
        labels_out.extend(labels.tolist())
    return preds, labels_out


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    rows: list[dict[str, Any]],
    transform: transforms.Compose,
    device: torch.device,
    repeats: int,
) -> list[float]:
    model.eval()
    latencies: list[float] = []
    sample_rows = (rows * ((repeats // len(rows)) + 1))[:repeats]

    for row in sample_rows[:20]:
        image = transform(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)
        model(image)

    for row in sample_rows:
        started = time.perf_counter_ns()
        image = transform(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)
        model(image)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    return latencies


@torch.no_grad()
def measure_model_only_latency(
    model: nn.Module,
    rows: list[dict[str, Any]],
    transform: transforms.Compose,
    device: torch.device,
    repeats: int,
) -> list[float]:
    model.eval()
    tensors = [
        transform(Image.open(row["path"]).convert("RGB")).unsqueeze(0).to(device)
        for row in rows
    ]
    sample_tensors = (tensors * ((repeats // len(tensors)) + 1))[:repeats]
    for tensor in sample_tensors[:20]:
        model(tensor)
    latencies: list[float] = []
    for tensor in sample_tensors:
        started = time.perf_counter_ns()
        model(tensor)
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
    parser.add_argument("--output", default="artifacts/vision_mobilenetv3_small.pt")
    parser.add_argument("--result", default="artifacts/mobilenetv3_small_results.json")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--fine-tune", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["images"]
    labels = [row["label"] for row in rows]
    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_loader = DataLoader(
        ImageManifestDataset(train_rows, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        ImageManifestDataset(test_rows, eval_transform),
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(freeze_features=not args.fine_tune)
    train_model(model, train_loader, device, args.epochs, args.learning_rate)
    preds, golds = evaluate_model(model, test_loader, device)
    end_to_end_latencies = measure_latency(model, test_rows, eval_transform, device, args.latency_repeats)
    model_only_latencies = measure_model_only_latency(model, test_rows, eval_transform, device, args.latency_repeats)

    result = {
        "scheme": "MobileNetV3-Small",
        "device": str(device),
        "image_size": args.image_size,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "accuracy": accuracy_score(golds, preds),
        "classification_report": classification_report(golds, preds, target_names=["out_domain", "in_domain"], output_dict=True),
        "latency_ms": {
            "end_to_end_image_open_preprocess_model": latency_summary(end_to_end_latencies),
            "model_only_preprocessed_tensor": latency_summary(model_only_latencies),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output_path)
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")
    print(f"wrote {args.result}")


if __name__ == "__main__":
    main()
