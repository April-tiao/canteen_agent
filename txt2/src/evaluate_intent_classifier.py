from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


LABEL_NAMES = ["out_domain", "in_domain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an ONNX binary text intent classifier.")
    parser.add_argument("--data", default="data/canteen_query_dataset.json")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--model", default="artifacts/domain_classifier_int8.onnx")
    parser.add_argument("--tokenizer", default="artifacts/domain_classifier_hf")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--output", default="artifacts/intent_eval_test.json")
    return parser.parse_args()


def row_text(row: dict[str, Any]) -> str:
    return str(row.get("corrected_query", row.get("corrected", row.get("query", row.get("q", "")))))


def row_label(row: dict[str, Any]) -> int:
    if "label" in row:
        return int(row["label"])
    return 1 if str(row["domain"]) == "in_domain" else 0


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return list(payload)
    if not isinstance(payload, dict):
        raise TypeError("Dataset must be a JSON list or an object with train/test splits.")
    if split == "all":
        return list(payload.get("train", [])) + list(payload.get("test", []))
    return list(payload.get(split, []))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


class OnnxIntentClassifier:
    def __init__(self, model_path: Path, tokenizer_path: Path, max_length: int) -> None:
        import numpy as np
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.np = np
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def predict_score(self, text: str) -> float:
        encoded = self.tokenizer(
            text,
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
        return float(probs[1])


def main() -> None:
    args = parse_args()

    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    except ImportError as exc:
        raise RuntimeError("Missing scikit-learn. Install packages from requirements.txt first.") from exc

    rows = load_rows(Path(args.data), args.split)
    classifier = OnnxIntentClassifier(Path(args.model), Path(args.tokenizer), args.max_length)

    for _ in range(20):
        classifier.predict_score("今天食堂消费统计")

    golds: list[int] = []
    preds: list[int] = []
    details: list[dict[str, Any]] = []
    latencies: list[float] = []

    for row in rows:
        text = row_text(row)
        gold = row_label(row)

        started = time.perf_counter_ns()
        score = classifier.predict_score(text)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000

        pred = 1 if score >= args.threshold else 0
        golds.append(gold)
        preds.append(pred)
        latencies.append(latency_ms)
        details.append(
            {
                "id": row.get("id"),
                "source": row.get("source"),
                "text": text,
                "expected_label": gold,
                "expected_intent": LABEL_NAMES[gold],
                "predicted_label": pred,
                "predicted_intent": LABEL_NAMES[pred],
                "in_domain_score": round(score, 6),
                "latency_ms": round(latency_ms, 4),
                "correct": pred == gold,
            }
        )

    total = len(rows)
    accuracy = accuracy_score(golds, preds) if total else 0.0
    matrix = confusion_matrix(golds, preds, labels=[1, 0])
    tp, fn = int(matrix[0][0]), int(matrix[0][1])
    fp, tn = int(matrix[1][0]), int(matrix[1][1])
    report = classification_report(golds, preds, labels=[0, 1], target_names=LABEL_NAMES, output_dict=True, zero_division=0)

    summary = {
        "task": "binary_text_intent_classification",
        "labels": {"0": LABEL_NAMES[0], "1": LABEL_NAMES[1]},
        "data": str(Path(args.data)),
        "split": args.split,
        "test_size": total,
        "threshold": args.threshold,
        "accuracy": accuracy,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "classification_report": report,
        "latency": {
            "avg_ms": statistics.mean(latencies) if latencies else 0.0,
            "p50_ms": statistics.median(latencies) if latencies else 0.0,
            "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99),
            "max_ms": max(latencies) if latencies else 0.0,
            "under_50ms": sum(1 for value in latencies if value <= 50) / total if total else 0.0,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Intent Classification Evaluation ===")
    print(f"data: {Path(args.data)}")
    print(f"split: {args.split}")
    print(f"test_size: {total}")
    print(f"accuracy: {accuracy:.2%}")
    print(f"confusion: {json.dumps(summary['confusion'], ensure_ascii=False)}")
    print(f"macro_f1: {report['macro avg']['f1-score']:.4f}")
    print(f"weighted_f1: {report['weighted avg']['f1-score']:.4f}")
    print(f"avg_latency_ms: {summary['latency']['avg_ms']:.4f}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
