from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Any

from evaluate_intent_classifier import LABEL_NAMES, OnnxIntentClassifier, percentile
from text_processing import correct_query, extract_time, load_correction_table, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the full text pipeline: correction, time extraction, intent.")
    parser.add_argument("--data", default="data/canteen_query_dataset.json")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--model", default="artifacts/domain_classifier_int8.onnx")
    parser.add_argument("--tokenizer", default="artifacts/domain_classifier_hf")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--correction-lexicon", default="artifacts/correction_lexicon.json")
    parser.add_argument("--output", default="artifacts/text_pipeline_eval_test.json")
    return parser.parse_args()


def load_rows(path: Path, split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return list(payload), {}
    if not isinstance(payload, dict):
        raise TypeError("Dataset must be a JSON list or an object with train/test splits.")

    meta = payload.get("meta", {})
    if split == "all":
        return list(payload.get("train", [])) + list(payload.get("test", [])), meta
    return list(payload.get(split, [])), meta


def expected_label(row: dict[str, Any]) -> int:
    if "label" in row:
        return int(row["label"])
    return 1 if str(row["domain"]) == "in_domain" else 0


def expected_corrected_query(row: dict[str, Any]) -> str:
    return str(row.get("corrected_query", row.get("corrected", row.get("query", row.get("q", "")))))


def expected_time_range(row: dict[str, Any]) -> tuple[str | None, str | None]:
    time_range = row.get("time_range")
    if not isinstance(time_range, dict):
        return None, None
    return time_range.get("start_date"), time_range.get("end_date")


def reference_date(row: dict[str, Any], meta: dict[str, Any]) -> date:
    value = None
    time_range = row.get("time_range")
    if isinstance(time_range, dict):
        value = time_range.get("reference_date")
    value = value or meta.get("reference_date")
    if value:
        return date.fromisoformat(value)
    return date.today()


def main() -> None:
    args = parse_args()
    rows, meta = load_rows(Path(args.data), args.split)
    classifier = OnnxIntentClassifier(Path(args.model), Path(args.tokenizer), args.max_length)
    correction_table = load_correction_table(args.correction_lexicon)

    for _ in range(20):
        classifier.predict_score("今天食堂消费统计")

    details: list[dict[str, Any]] = []
    latencies: list[float] = []
    intent_correct = 0
    time_correct = 0
    correction_correct = 0
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    by_source: dict[str, dict[str, int]] = {}

    for row in rows:
        query = str(row.get("query", row.get("q", "")))
        gold_label = expected_label(row)
        gold_intent = LABEL_NAMES[gold_label]
        gold_corrected = expected_corrected_query(row)
        gold_start, gold_end = expected_time_range(row)
        today = reference_date(row, meta)

        started = time.perf_counter_ns()
        normalized = normalize_text(query)
        predicted_corrected, corrections = correct_query(normalized, correction_table)
        time_info = extract_time(predicted_corrected, today=today)
        score = classifier.predict_score(predicted_corrected)
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000

        pred_label = 1 if score >= args.threshold else 0
        pred_intent = LABEL_NAMES[pred_label]
        pred_start = time_info["start"] if time_info else None
        pred_end = time_info["end"] if time_info else None

        is_intent_correct = pred_label == gold_label
        is_time_correct = (pred_start, pred_end) == (gold_start, gold_end)
        is_correction_correct = predicted_corrected == gold_corrected

        intent_correct += int(is_intent_correct)
        time_correct += int(is_time_correct)
        correction_correct += int(is_correction_correct)
        latencies.append(latency_ms)

        if gold_label == 1 and pred_label == 1:
            confusion["tp"] += 1
        elif gold_label == 0 and pred_label == 0:
            confusion["tn"] += 1
        elif gold_label == 0 and pred_label == 1:
            confusion["fp"] += 1
        else:
            confusion["fn"] += 1

        source = str(row.get("source", "unknown"))
        bucket = by_source.setdefault(
            source,
            {"total": 0, "intent_correct": 0, "time_correct": 0, "correction_correct": 0},
        )
        bucket["total"] += 1
        bucket["intent_correct"] += int(is_intent_correct)
        bucket["time_correct"] += int(is_time_correct)
        bucket["correction_correct"] += int(is_correction_correct)

        details.append(
            {
                "id": row.get("id"),
                "source": source,
                "query": query,
                "normalized_query": normalized,
                "expected_corrected_query": gold_corrected,
                "predicted_corrected_query": predicted_corrected,
                "corrections": corrections,
                "expected_intent": gold_intent,
                "predicted_intent": pred_intent,
                "in_domain_score": round(score, 6),
                "expected_time": {"start": gold_start, "end": gold_end},
                "predicted_time": time_info,
                "latency_ms": round(latency_ms, 4),
                "intent_correct": is_intent_correct,
                "time_correct": is_time_correct,
                "correction_correct": is_correction_correct,
            }
        )

    total = len(rows)
    summary = {
        "task": "full_text_pipeline",
        "labels": {"0": LABEL_NAMES[0], "1": LABEL_NAMES[1]},
        "data": str(Path(args.data)),
        "split": args.split,
        "test_size": total,
        "threshold": args.threshold,
        "intent_accuracy": intent_correct / total if total else 0.0,
        "time_accuracy": time_correct / total if total else 0.0,
        "correction_accuracy": correction_correct / total if total else 0.0,
        "confusion": confusion,
        "latency": {
            "avg_ms": statistics.mean(latencies) if latencies else 0.0,
            "p50_ms": statistics.median(latencies) if latencies else 0.0,
            "p95_ms": percentile(latencies, 0.95),
            "p99_ms": percentile(latencies, 0.99),
            "max_ms": max(latencies) if latencies else 0.0,
            "under_50ms": sum(1 for value in latencies if value <= 50) / total if total else 0.0,
        },
        "by_source": by_source,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "details": details}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Full Text Pipeline Evaluation ===")
    print(f"data: {Path(args.data)}")
    print(f"split: {args.split}")
    print(f"test_size: {total}")
    print(f"intent_accuracy:     {summary['intent_accuracy']:.2%} ({intent_correct}/{total})")
    print(f"time_accuracy:       {summary['time_accuracy']:.2%} ({time_correct}/{total})")
    print(f"correction_accuracy: {summary['correction_accuracy']:.2%} ({correction_correct}/{total})")
    print(f"confusion: {json.dumps(confusion, ensure_ascii=False)}")
    print(f"avg_latency_ms: {summary['latency']['avg_ms']:.4f}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
