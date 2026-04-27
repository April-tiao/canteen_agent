from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_BASE_MODEL = "uer/chinese_roberta_L-4_H-512"
DEFAULT_DATA_FILE = "canteen_test_data_300.json"
DEFAULT_OUTPUT_DIR = "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_FILE)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    return parser.parse_args()


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
        raise RuntimeError(
            "Missing training dependencies. Install requirements-prod.txt first."
        ) from exc

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(data_path.read_text(encoding="utf-8"))
    texts = [row["corrected"] for row in rows]
    labels = [1 if row["domain"] == "in_domain" else 0 for row in rows]

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

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

    print(f"eval_accuracy={accuracy_score(golds, preds):.4f}")
    print(classification_report(golds, preds, target_names=["out_domain", "in_domain"]))

    model_dir = output_dir / "domain_classifier_hf"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    vocab_src = model_dir / "vocab.txt"
    if vocab_src.exists():
        shutil.copyfile(vocab_src, output_dir / "vocab.txt")

    dummy = tokenizer(
        "今天中午二食堂有宫保鸡丁吗",
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
    torch.onnx.export(
        model,
        tuple(dummy[name] for name in input_names),
        onnx_path,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={name: {0: "batch"} for name in input_names} | {"logits": {0: "batch"}},
        opset_version=14,
        dynamo=False,
    )

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )

    print(f"wrote {onnx_path}")
    print(f"wrote {int8_path}")
    print(f"wrote {output_dir / 'vocab.txt'}")


if __name__ == "__main__":
    main()
