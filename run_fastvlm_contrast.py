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

from canteen_core import percentile


os.environ.setdefault("HF_HOME", str(Path("artifacts/model_cache/huggingface").resolve()))


IMAGE_TOKEN_INDEX = -200
PROMPT = "判断这条图文输入是否属于智慧食堂业务范围。只回答 in_domain 或 out_domain。用户文本：{text}"


def pick_texts(manifest: dict[str, Any]) -> tuple[list[str], list[str]]:
    positives = [row["text"] for row in manifest["texts"] if row["label"] == 1]
    negatives = [row["text"] for row in manifest["texts"] if row["label"] == 0]
    return positives or ["这张图里的食堂菜品是什么"], negatives or ["请帮我写一首诗"]


def build_fastvlm_inputs(tokenizer: Any, model: Any, image_path: str, text: str) -> dict[str, Any]:
    prompt = "<image>\n" + PROMPT.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    pre, post = rendered.split("<image>", 1)
    pre_ids = tokenizer(pre, return_tensors="pt", add_special_tokens=False).input_ids
    post_ids = tokenizer(post, return_tensors="pt", add_special_tokens=False).input_ids
    image_token = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, image_token, post_ids], dim=1).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    image = Image.open(image_path).convert("RGB")
    pixel_values = model.get_vision_tower().image_processor(images=image, return_tensors="pt")["pixel_values"]
    pixel_values = pixel_values.to(model.device, dtype=model.dtype)
    return {
        "inputs": input_ids,
        "attention_mask": attention_mask,
        "images": pixel_values,
    }


def predict_label(tokenizer: Any, model: Any, image_path: str, text: str) -> int:
    inputs = build_fastvlm_inputs(tokenizer, model, image_path, text)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=8)
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True).lower()
    return 1 if "in_domain" in decoded and "out_domain" not in decoded else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/multimodal_manifest.json")
    parser.add_argument("--model-id", default="apple/FastVLM-0.5B")
    parser.add_argument("--result", default="artifacts/fastvlm_0_5b_contrast_results.json")
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["images"]
    positives, negatives = pick_texts(manifest)
    rows, _ = train_test_split(
        rows,
        test_size=max(0, len(rows) - args.sample_size),
        random_state=7,
        stratify=[row["label"] for row in rows],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    golds: list[int] = []
    preds: list[int] = []
    latencies: list[float] = []
    for index, row in enumerate(rows):
        text_pool = positives if row["label"] == 1 else negatives
        text = text_pool[index % len(text_pool)]
        started = time.perf_counter_ns()
        pred = predict_label(tokenizer, model, row["path"], text)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        golds.append(row["label"])
        preds.append(pred)

    result = {
        "scheme": "FastVLM-0.5B contrast group",
        "model_id": args.model_id,
        "device": str(device),
        "sample_size": len(rows),
        "accuracy": accuracy_score(golds, preds),
        "classification_report": classification_report(golds, preds, target_names=["out_domain", "in_domain"], output_dict=True),
        "latency_ms": {
            "avg": statistics.mean(latencies),
            "p50": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies),
            "under_50ms": sum(1 for value in latencies if value <= 50) / len(latencies),
        },
    }
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
