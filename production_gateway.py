from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from canteen_core import (
    correct_query,
    extract_time,
    normalize_text,
    percentile,
    rule_domain_score,
)


DEFAULT_MODEL_PATH = Path("artifacts/domain_classifier_int8.onnx")
DEFAULT_VOCAB_PATH = Path("artifacts/domain_classifier_hf")
DEFAULT_DATA_PATH = Path("canteen_test_data_300.json")


class BinaryClassifier(Protocol):
    def predict_score(self, text: str, time_info: dict[str, Any] | None) -> float:
        """Return in-domain probability in [0, 1]."""


class RuleFallbackClassifier:
    """Local smoke-test fallback. Do not use this as the production classifier."""

    def predict_score(self, text: str, time_info: dict[str, Any] | None) -> float:
        return rule_domain_score(text, time_info)


class OnnxBertBinaryClassifier:
    def __init__(self, model_path: Path, tokenizer_path: Path, max_length: int = 64) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Production model requires numpy, onnxruntime and transformers. "
                "Install dependencies from requirements-prod.txt."
            ) from exc

        if not model_path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Missing tokenizer directory: {tokenizer_path}")

        self.np = np
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def predict_score(self, text: str, time_info: dict[str, Any] | None) -> float:
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )
        inputs = {}
        for name, values in encoded.items():
            if name in self.input_names:
                inputs[name] = self.np.asarray([values], dtype=self.np.int64)

        logits = self.session.run(None, inputs)[0][0]
        if len(logits) == 1:
            return float(1.0 / (1.0 + self.np.exp(-logits[0])))

        shifted = logits - self.np.max(logits)
        probs = self.np.exp(shifted) / self.np.sum(self.np.exp(shifted))
        return float(probs[1])


@dataclass(frozen=True)
class GatewayOutput:
    domain: str
    domain_score: float
    original_query: str
    corrected_query: str
    correction_applied: bool
    corrections: list[dict[str, Any]]
    time: dict[str, Any] | None
    latency_ms: float


class ProductionCanteenGateway:
    def __init__(self, classifier: BinaryClassifier, threshold: float = 0.5) -> None:
        self.classifier = classifier
        self.threshold = threshold

    def handle(self, query: str) -> GatewayOutput:
        started = time.perf_counter_ns()

        normalized = normalize_text(query)
        corrected_query, corrections = correct_query(normalized)
        time_info = extract_time(corrected_query)
        score = self.classifier.predict_score(corrected_query, time_info)
        domain = "in_domain" if score >= self.threshold else "out_domain"

        latency_ms = (time.perf_counter_ns() - started) / 1_000_000
        return GatewayOutput(
            domain=domain,
            domain_score=round(score, 4),
            original_query=query,
            corrected_query=corrected_query,
            correction_applied=bool(corrections),
            corrections=corrections,
            time=time_info,
            latency_ms=latency_ms,
        )


def build_classifier(args: argparse.Namespace) -> BinaryClassifier:
    model_path = Path(args.model)
    tokenizer_path = Path(args.vocab)
    if model_path.exists() and tokenizer_path.exists():
        return OnnxBertBinaryClassifier(model_path, tokenizer_path, max_length=args.max_length)
    if args.allow_rule_fallback:
        return RuleFallbackClassifier()
    raise FileNotFoundError(
        f"Missing production model artifacts: {model_path} and/or {tokenizer_path}. "
        "Train/export the model first, or pass --allow-rule-fallback only for local smoke tests."
    )


def evaluate(gateway: ProductionCanteenGateway, data_path: Path) -> None:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    latencies: list[float] = []
    domain_correct = 0
    time_correct = 0
    correction_correct = 0

    for _ in range(300):
        gateway.handle("今天中午二食堂有宫爆鸡丁吗")

    for item in data:
        result = gateway.handle(item["q"])
        latencies.append(result.latency_ms)
        domain_correct += int(result.domain == item["domain"])
        time_correct += int(bool(result.time) == item["time"])
        correction_correct += int(result.corrected_query == item["corrected"])

    total = len(data)
    print("=== Production Gateway Evaluation ===")
    print(f"test_size: {total}")
    print(f"domain_accuracy:     {domain_correct / total:.2%} ({domain_correct}/{total})")
    print(f"time_accuracy:       {time_correct / total:.2%} ({time_correct}/{total})")
    print(f"correction_accuracy: {correction_correct / total:.2%} ({correction_correct}/{total})")
    print()
    print("=== Per-query Latency ===")
    print(f"avg_ms: {statistics.mean(latencies):.4f}")
    print(f"p50_ms: {statistics.median(latencies):.4f}")
    print(f"p95_ms: {percentile(latencies, 0.95):.4f}")
    print(f"p99_ms: {percentile(latencies, 0.99):.4f}")
    print(f"max_ms: {max(latencies):.4f}")
    print(f"under_50ms: {sum(1 for latency in latencies if latency <= 50) / total:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--vocab", default=str(DEFAULT_VOCAB_PATH), help="Tokenizer directory saved by training.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument(
        "--allow-rule-fallback",
        action="store_true",
        help="Only for local smoke tests when ONNX artifacts are unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier = build_classifier(args)
    gateway = ProductionCanteenGateway(classifier, threshold=args.threshold)
    evaluate(gateway, Path(args.data))


if __name__ == "__main__":
    main()
