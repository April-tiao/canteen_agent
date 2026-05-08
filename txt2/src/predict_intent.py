from __future__ import annotations

import argparse
import json
from pathlib import Path

from text_processing import correct_query, extract_time, load_correction_table, normalize_text


LABEL_NAMES = ["out_domain", "in_domain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one text intent classification prediction.")
    parser.add_argument("text")
    parser.add_argument("--model", default="artifacts/domain_classifier_int8.onnx")
    parser.add_argument("--tokenizer", default="artifacts/domain_classifier_hf")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--correction-lexicon", default="artifacts/correction_lexicon.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(args.tokenizer))
    session = ort.InferenceSession(str(Path(args.model)), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    normalized = normalize_text(args.text)
    correction_table = load_correction_table(args.correction_lexicon)
    corrected_query, corrections = correct_query(normalized, correction_table)
    time_info = extract_time(corrected_query)

    encoded = tokenizer(
        corrected_query,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,
    )
    inputs = {
        name: np.asarray([values], dtype=np.int64)
        for name, values in encoded.items()
        if name in input_names
    }
    logits = session.run(None, inputs)[0][0]
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    score = float(probs[1])
    label = 1 if score >= args.threshold else 0

    print(
        json.dumps(
            {
                "original_query": args.text,
                "normalized_query": normalized,
                "corrected_query": corrected_query,
                "corrections": corrections,
                "time": time_info,
                "label": label,
                "intent": LABEL_NAMES[label],
                "in_domain_score": round(score, 6),
                "threshold": args.threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
