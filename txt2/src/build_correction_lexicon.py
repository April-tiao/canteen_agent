from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a query correction lexicon from dataset annotations.")
    parser.add_argument("--data", default="data/canteen_query_dataset.json")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--output", default="artifacts/correction_lexicon.json")
    parser.add_argument("--min-count", type=int, default=1)
    return parser.parse_args()


def load_rows(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return list(payload)
    if not isinstance(payload, dict):
        raise TypeError("Dataset must be a JSON list or an object with train/test splits.")
    if split == "all":
        return list(payload.get("train", [])) + list(payload.get("test", []))
    return list(payload.get(split, []))


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.data), args.split)
    counts: Counter[tuple[str, str]] = Counter()

    for row in rows:
        for correction in row.get("corrections", []):
            wrong = correction.get("wrong")
            correct = correction.get("correct")
            if wrong and correct and wrong != correct:
                counts[(str(wrong), str(correct))] += 1

    corrections = [
        {"wrong": wrong, "correct": correct, "count": count}
        for (wrong, correct), count in counts.most_common()
        if count >= args.min_count
    ]

    output = {
        "source_data": str(Path(args.data)),
        "source_split": args.split,
        "min_count": args.min_count,
        "total_pairs": len(corrections),
        "corrections": corrections,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(rows)} corrections={len(corrections)} wrote={output_path}")


if __name__ == "__main__":
    main()
