from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "uer/chinese_roberta_L-4_H-512"
DEFAULT_DATA_FILE = "data/canteen_query_dataset.json"
DEFAULT_OUTPUT_DIR = "artifacts"
LABEL_NAMES = ["out_domain", "in_domain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a binary text intent classifier.")
    parser.add_argument("--data", default=DEFAULT_DATA_FILE)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def row_text(row: dict[str, Any]) -> str:
    return str(row.get("corrected_query", row.get("corrected", row.get("query", row.get("q", "")))))


def row_label(row: dict[str, Any]) -> int:
    if "label" in row:
        return int(row["label"])
    domain = str(row["domain"])
    return 1 if domain == "in_domain" else 0


def load_dataset(path: Path, train_split: str, eval_split: str) -> tuple[list[str], list[int], list[str], list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and train_split in payload and eval_split in payload:
        train_rows = list(payload[train_split])
        eval_rows = list(payload[eval_split])
        return (
            [row_text(row) for row in train_rows],
            [row_label(row) for row in train_rows],
            [row_text(row) for row in eval_rows],
            [row_label(row) for row in eval_rows],
        )

    if not isinstance(payload, list):
        raise TypeError("Dataset must be a JSON list or an object with train/test splits.")

    rows = list(payload)
    return [row_text(row) for row in rows], [row_label(row) for row in rows], [], []


def main() -> None:
    args = parse_args()

    try:
        import numpy as np
        import torch
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing dependencies. Install packages from requirements.txt first.") from exc

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_texts, train_labels, eval_texts, eval_labels = load_dataset(
        data_path,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    if not eval_texts:
        train_texts, eval_texts, train_labels, eval_labels = train_test_split(
            train_texts,
            train_labels,
            test_size=0.2,
            random_state=args.seed,
            stratify=train_labels,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)
    print(f"base_model={args.base_model}")
    print(f"train_size={len(train_texts)} eval_size={len(eval_texts)}")

    class TextDataset(Dataset):
        def __init__(self, split_texts: list[str], split_labels: list[int]) -> None:
            self.texts = split_texts
            self.labels = split_labels

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            encoded = tokenizer(
                self.texts[index],
                max_length=args.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    train_loader = DataLoader(TextDataset(train_texts, train_labels), batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(TextDataset(eval_texts, eval_labels), batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())
        print(f"epoch={epoch + 1} train_loss={total_loss / max(len(train_loader), 1):.4f}")

    model.eval()
    preds: list[int] = []
    golds: list[int] = []
    with torch.no_grad():
        for batch in eval_loader:
            labels_tensor = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits.detach().cpu().numpy()
            preds.extend(np.argmax(logits, axis=1).tolist())
            golds.extend(labels_tensor.numpy().tolist())

    eval_accuracy = accuracy_score(golds, preds)
    report_text = classification_report(golds, preds, target_names=LABEL_NAMES)
    report_dict = classification_report(golds, preds, target_names=LABEL_NAMES, output_dict=True)
    print(f"eval_accuracy={eval_accuracy:.4f}")
    print(report_text)

    model_dir = output_dir / "domain_classifier_hf"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    vocab_src = model_dir / "vocab.txt"
    if vocab_src.exists():
        shutil.copyfile(vocab_src, output_dir / "vocab.txt")

    metrics_path = output_dir / "domain_classifier_eval_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "task": "binary_text_intent_classification",
                "labels": {"0": LABEL_NAMES[0], "1": LABEL_NAMES[1]},
                "data": str(data_path),
                "base_model": args.base_model,
                "train_size": len(train_texts),
                "eval_size": len(eval_texts),
                "eval_accuracy": eval_accuracy,
                "classification_report": report_dict,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dummy = tokenizer(
        "今天食堂消费统计",
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    dummy = {key: value.to(device) for key, value in dummy.items()}
    onnx_path = output_dir / "domain_classifier.onnx"
    int8_path = output_dir / "domain_classifier_int8.onnx"

    input_names = ["input_ids", "attention_mask"]
    if "token_type_ids" in dummy:
        input_names.append("token_type_ids")
    dynamic_axes = {name: {0: "batch"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    torch.onnx.export(
        model,
        tuple(dummy[name] for name in input_names),
        onnx_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        dynamo=False,
    )

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    print(f"wrote {model_dir}")
    print(f"wrote {onnx_path}")
    print(f"wrote {int8_path}")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
