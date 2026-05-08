from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from canteen_3class_experiment import IMAGE_EXTS, INDEX_TO_LABEL, LABEL_TO_INDEX, build_model, build_transforms


LABEL_NAMES = ["out_domain", "in_domain"]
IMAGE_INTENT_MAP = {
    0: 0,  # other -> out_domain
    1: 1,  # food -> in_domain
    2: 1,  # chart -> in_domain
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paired text+image canteen-domain intent recognition.")
    parser.add_argument("--text-project", default="../canteen_agent-main/txt2")
    parser.add_argument("--text-data", default=None)
    parser.add_argument("--text-split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--text-tokenizer", default=None)
    parser.add_argument("--correction-lexicon", default=None)
    parser.add_argument("--image-data-dir", default="image_dataset_3class")
    parser.add_argument("--image-checkpoint", default="outputs_mnv3_3class_160/best_model.pt")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--pair-mode", choices=["cycle", "manifest"], default="cycle")
    parser.add_argument("--pair-manifest", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output", default="outputs_mnv3_3class_160/text_image_intent_fusion_eval.json")
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "avg_ms": statistics.mean(values) if values else 0.0,
        "p50_ms": statistics.median(values) if values else 0.0,
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else 0.0,
        "under_50ms": sum(1 for value in values if value <= 50) / len(values) if values else 0.0,
    }


def update_confusion(confusion: dict[str, int], gold: int, pred: int) -> None:
    if gold == 1 and pred == 1:
        confusion["tp"] += 1
    elif gold == 0 and pred == 0:
        confusion["tn"] += 1
    elif gold == 0 and pred == 1:
        confusion["fp"] += 1
    else:
        confusion["fn"] += 1


def row_text(row: dict[str, Any]) -> str:
    return str(row.get("query", row.get("q", "")))


def row_label(row: dict[str, Any]) -> int:
    if "label" in row:
        return int(row["label"])
    return 1 if str(row["domain"]) == "in_domain" else 0


def load_text_rows(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return list(payload)
    if not isinstance(payload, dict):
        raise TypeError("Text dataset must be a JSON list or an object with train/test splits.")
    if split == "all":
        return list(payload.get("train", [])) + list(payload.get("test", []))
    return list(payload.get(split, []))


class TextIntentRuntime:
    def __init__(self, project_dir: Path, model: Path, tokenizer: Path, correction_lexicon: Path, max_length: int) -> None:
        src_dir = project_dir / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

        import onnxruntime as ort
        from text_processing import correct_query, load_correction_table, normalize_text
        from transformers import AutoTokenizer

        self.np = np
        self.correct_query = correct_query
        self.normalize_text = normalize_text
        self.correction_table = load_correction_table(correction_lexicon)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def predict(self, text: str, threshold: float) -> dict[str, Any]:
        normalized = self.normalize_text(text)
        corrected, corrections = self.correct_query(normalized, self.correction_table)
        encoded = self.tokenizer(
            corrected,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        inputs = {
            name: self.np.asarray([values], dtype=self.np.int64)
            for name, values in encoded.items()
            if name in self.input_names
        }
        logits = self.session.run(None, inputs)[0][0]
        shifted = logits - self.np.max(logits)
        probs = self.np.exp(shifted) / self.np.sum(self.np.exp(shifted))
        score = float(probs[1])
        label = 1 if score >= threshold else 0
        return {
            "label": label,
            "intent": LABEL_NAMES[label],
            "score": score,
            "corrected_query": corrected,
            "corrections": corrections,
        }


class ImageIntentRuntime:
    def __init__(self, checkpoint_path: Path, image_size: int | None, device_name: str) -> None:
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = build_model(pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.image_size = image_size or int(checkpoint.get("image_size", 160))
        _, self.transform = build_transforms(self.image_size)

    @torch.no_grad()
    def predict(self, path: Path) -> dict[str, Any]:
        with Image.open(path) as img:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
        return self.predict_tensor(tensor)

    @torch.no_grad()
    def predict_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        logits = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        image_label = int(probs.argmax())
        intent_label = IMAGE_INTENT_MAP[image_label]
        return {
            "label": intent_label,
            "intent": LABEL_NAMES[intent_label],
            "image_label": image_label,
            "image_class": INDEX_TO_LABEL[image_label],
            "image_probs": {INDEX_TO_LABEL[index]: float(value) for index, value in enumerate(probs)},
        }


def evaluate_text(runtime: TextIntentRuntime, rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    latencies: list[float] = []
    details: list[dict[str, Any]] = []
    correct = 0

    for row in rows:
        text = row_text(row)
        gold = row_label(row)
        started = time.perf_counter_ns()
        pred = runtime.predict(text, threshold)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        latencies.append(latency_ms)
        correct += int(pred["label"] == gold)
        update_confusion(confusion, gold, pred["label"])
        details.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "input_type": "text",
                "query": text,
                "expected_intent": LABEL_NAMES[gold],
                "predicted_intent": pred["intent"],
                "score": round(pred["score"], 6),
                "latency_ms": round(latency_ms, 4),
                "correct": pred["label"] == gold,
            }
        )

    return {
        "samples": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "confusion": confusion,
        "latency": latency_summary(latencies),
        "details": details,
    }


def image_samples(data_dir: Path) -> list[tuple[Path, int, int]]:
    samples: list[tuple[Path, int, int]] = []
    for class_name, image_label in LABEL_TO_INDEX.items():
        class_dir = data_dir / "test" / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                samples.append((path, image_label, IMAGE_INTENT_MAP[image_label]))
    return samples


def paired_samples_cycle(
    text_rows: list[dict[str, Any]],
    images: list[tuple[Path, int, int]],
) -> list[dict[str, Any]]:
    if not text_rows or not images:
        return []
    pairs = []
    for index, (image_path, image_label, image_intent) in enumerate(images):
        text_row = text_rows[index % len(text_rows)]
        text_gold = row_label(text_row)
        # The business decision is positive if either modality contains canteen-domain evidence.
        fused_gold = 1 if text_gold == 1 or image_intent == 1 else 0
        pairs.append(
            {
                "id": f"pair_{index:04d}",
                "text_row": text_row,
                "image_path": image_path,
                "image_label": image_label,
                "image_intent": image_intent,
                "label": fused_gold,
            }
        )
    return pairs


def paired_samples_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("Pair manifest must be a JSON list.")
    pairs = []
    for index, row in enumerate(payload):
        text_label = int(row["text_label"]) if "text_label" in row else None
        image_intent = int(row["image_label"]) if "image_label" in row else None
        fused_label = int(row["label"]) if "label" in row else 1 if text_label == 1 or image_intent == 1 else 0
        pairs.append(
            {
                "id": row.get("id", f"pair_{index:04d}"),
                "text_row": {"id": row.get("text_id"), "query": row["query"], "label": text_label if text_label is not None else fused_label},
                "image_path": Path(row["image_path"]),
                "image_label": row.get("image_class_label"),
                "image_intent": image_intent,
                "label": fused_label,
            }
        )
    return pairs


def evaluate_image(runtime: ImageIntentRuntime, samples: list[tuple[Path, int, int]]) -> dict[str, Any]:
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    latencies: list[float] = []
    details: list[dict[str, Any]] = []
    correct = 0

    for path, image_gold, intent_gold in samples:
        started = time.perf_counter_ns()
        pred = runtime.predict(path)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        latencies.append(latency_ms)
        correct += int(pred["label"] == intent_gold)
        update_confusion(confusion, intent_gold, pred["label"])
        details.append(
            {
                "input_type": "image",
                "path": str(path),
                "expected_image_class": INDEX_TO_LABEL[image_gold],
                "predicted_image_class": pred["image_class"],
                "expected_intent": LABEL_NAMES[intent_gold],
                "predicted_intent": pred["intent"],
                "latency_ms": round(latency_ms, 4),
                "correct": pred["label"] == intent_gold,
            }
        )

    return {
        "samples": len(samples),
        "accuracy": correct / len(samples) if samples else 0.0,
        "confusion": confusion,
        "latency_end_to_end": latency_summary(latencies),
        "details": details,
    }


def evaluate_image_inference_latency(runtime: ImageIntentRuntime, samples: list[tuple[Path, int, int]]) -> dict[str, float]:
    tensors = []
    for path, _, _ in samples:
        with Image.open(path) as img:
            tensors.append(runtime.transform(img).unsqueeze(0).to(runtime.device))

    for tensor in tensors[:20]:
        runtime.predict_tensor(tensor)

    latencies = []
    for tensor in tensors:
        started = time.perf_counter_ns()
        runtime.predict_tensor(tensor)
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    return latency_summary(latencies)


def merge_summary(text_result: dict[str, Any], image_result: dict[str, Any]) -> dict[str, Any]:
    total = text_result["samples"] + image_result["samples"]
    correct = round(text_result["accuracy"] * text_result["samples"]) + round(image_result["accuracy"] * image_result["samples"])
    confusion = {
        key: text_result["confusion"][key] + image_result["confusion"][key]
        for key in ("tp", "tn", "fp", "fn")
    }


def evaluate_pairs(
    text_runtime: TextIntentRuntime,
    image_runtime: ImageIntentRuntime,
    pairs: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    text_latencies: list[float] = []
    image_latencies: list[float] = []
    sequential_latencies: list[float] = []
    parallel_estimated_latencies: list[float] = []
    details: list[dict[str, Any]] = []
    correct = 0

    for pair in pairs:
        query = row_text(pair["text_row"])
        gold = int(pair["label"])

        text_started = time.perf_counter_ns()
        text_pred = text_runtime.predict(query, threshold)
        text_latency_ms = (time.perf_counter_ns() - text_started) / 1_000_000

        image_started = time.perf_counter_ns()
        image_pred = image_runtime.predict(pair["image_path"])
        image_latency_ms = (time.perf_counter_ns() - image_started) / 1_000_000

        # Simultaneous text+image decision: a request belongs to canteen business if either
        # the text or the image provides in-domain evidence.
        pred = 1 if text_pred["label"] == 1 or image_pred["label"] == 1 else 0
        is_correct = pred == gold
        correct += int(is_correct)
        update_confusion(confusion, gold, pred)

        text_latencies.append(text_latency_ms)
        image_latencies.append(image_latency_ms)
        sequential_latencies.append(text_latency_ms + image_latency_ms)
        parallel_estimated_latencies.append(max(text_latency_ms, image_latency_ms))

        details.append(
            {
                "id": pair["id"],
                "query": query,
                "image_path": str(pair["image_path"]),
                "expected_intent": LABEL_NAMES[gold],
                "text_intent": text_pred["intent"],
                "text_score": round(text_pred["score"], 6),
                "image_class": image_pred["image_class"],
                "image_intent": image_pred["intent"],
                "predicted_intent": LABEL_NAMES[pred],
                "latency_text_ms": round(text_latency_ms, 4),
                "latency_image_end_to_end_ms": round(image_latency_ms, 4),
                "latency_sequential_ms": round(text_latency_ms + image_latency_ms, 4),
                "latency_parallel_estimated_ms": round(max(text_latency_ms, image_latency_ms), 4),
                "correct": is_correct,
            }
        )

    return {
        "samples": len(pairs),
        "accuracy": correct / len(pairs) if pairs else 0.0,
        "confusion": confusion,
        "latency_text": latency_summary(text_latencies),
        "latency_image_end_to_end": latency_summary(image_latencies),
        "latency_sequential": latency_summary(sequential_latencies),
        "latency_parallel_estimated": latency_summary(parallel_estimated_latencies),
        "details": details,
    }
    latencies = [item["latency_ms"] for item in text_result["details"]] + [item["latency_ms"] for item in image_result["details"]]
    return {
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "confusion": confusion,
        "latency_end_to_end": latency_summary(latencies),
    }


def main() -> None:
    args = parse_args()
    text_project = Path(args.text_project).resolve()
    text_data = Path(args.text_data).resolve() if args.text_data else text_project / "data" / "canteen_query_dataset.json"
    text_model = Path(args.text_model).resolve() if args.text_model else text_project / "artifacts" / "domain_classifier_int8.onnx"
    text_tokenizer = Path(args.text_tokenizer).resolve() if args.text_tokenizer else text_project / "artifacts" / "domain_classifier_hf"
    correction_lexicon = (
        Path(args.correction_lexicon).resolve()
        if args.correction_lexicon
        else text_project / "artifacts" / "correction_lexicon.json"
    )

    text_rows = load_text_rows(text_data, args.text_split)
    image_data_dir = Path(args.image_data_dir).resolve()
    image_rows = image_samples(image_data_dir)

    text_runtime = TextIntentRuntime(text_project, text_model, text_tokenizer, correction_lexicon, args.max_length)
    image_runtime = ImageIntentRuntime(Path(args.image_checkpoint), args.image_size, args.device)

    for _ in range(20):
        text_runtime.predict("今天食堂消费统计", args.threshold)

    text_result = evaluate_text(text_runtime, text_rows, args.threshold)
    image_result = evaluate_image(image_runtime, image_rows)
    image_result["latency_inference_only"] = evaluate_image_inference_latency(image_runtime, image_rows)
    if args.pair_mode == "manifest":
        if not args.pair_manifest:
            raise ValueError("--pair-manifest is required when --pair-mode manifest")
        pairs = paired_samples_manifest(Path(args.pair_manifest))
    else:
        pairs = paired_samples_cycle(text_rows, image_rows)
    paired_result = evaluate_pairs(text_runtime, image_runtime, pairs, args.threshold)

    output = {
        "task": "paired_text_image_canteen_domain_intent_fusion",
        "label_mapping": {
            "text": {"0": "out_domain", "1": "in_domain"},
            "image": {"0_other": "out_domain", "1_food": "in_domain", "2_chart": "in_domain"},
        },
        "fusion_policy": "paired request; run text and image models for the same request, then predict in_domain if either modality is in_domain",
        "text": {key: value for key, value in text_result.items() if key != "details"},
        "image": {key: value for key, value in image_result.items() if key != "details"},
        "paired": {key: value for key, value in paired_result.items() if key != "details"},
        "details": {
            "text": text_result["details"],
            "image": image_result["details"],
            "paired": paired_result["details"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Paired Text + Image Intent Fusion Evaluation ===")
    print(f"text_samples: {text_result['samples']} accuracy={text_result['accuracy']:.2%}")
    print(f"image_samples: {image_result['samples']} accuracy={image_result['accuracy']:.2%}")
    print(f"paired_samples: {paired_result['samples']} accuracy={paired_result['accuracy']:.2%}")
    print(f"paired_confusion: {json.dumps(paired_result['confusion'], ensure_ascii=False)}")
    print(f"text_latency: {json.dumps(text_result['latency'], ensure_ascii=False)}")
    print(f"image_latency_end_to_end: {json.dumps(image_result['latency_end_to_end'], ensure_ascii=False)}")
    print(f"image_latency_inference_only: {json.dumps(image_result['latency_inference_only'], ensure_ascii=False)}")
    print(f"paired_latency_sequential: {json.dumps(paired_result['latency_sequential'], ensure_ascii=False)}")
    print(f"paired_latency_parallel_estimated: {json.dumps(paired_result['latency_parallel_estimated'], ensure_ascii=False)}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
