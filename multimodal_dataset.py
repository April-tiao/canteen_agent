from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_IMAGE_LABELS = {
    "chart": 1,
    "food": 1,
    "fruit": 0,
}


@dataclass(frozen=True)
class ImageRecord:
    path: str
    label: int
    source_class: str


@dataclass(frozen=True)
class TextRecord:
    text: str
    label: int
    source: str


def extract_user_query(raw_input: str) -> str:
    text = raw_input.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            text = text[1:-1]

    current_match = re.search(r"【当前问题】\s*(.+)", text, flags=re.S)
    if current_match:
        return current_match.group(1).strip()

    user_match = re.search(r"用户输入:\s*(.+)", text, flags=re.S)
    if user_match:
        value = user_match.group(1).strip()
        value = value.split("【历史对话上下文】", 1)[0].strip()
        return value

    return text


def load_dialog_records(path: Path) -> list[TextRecord]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("ai_agent_records", data if isinstance(data, list) else [])
    records: list[TextRecord] = []
    for row in rows:
        raw_input = str(row.get("input") or "")
        query = extract_user_query(raw_input)
        if query:
            records.append(TextRecord(text=query, label=1, source="ai_agent_records"))
    return records


def load_text_negatives(path: Path) -> list[TextRecord]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for row in data:
        if row.get("domain") == "out_domain":
            records.append(TextRecord(text=row["q"], label=0, source="canteen_test_data_300"))
    return records


def load_image_records(root: Path, label_map: dict[str, int]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if class_dir.name not in label_map:
            continue
        label = label_map[class_dir.name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(
                    ImageRecord(
                        path=str(image_path),
                        label=label,
                        source_class=class_dir.name,
                    )
                )
    return records


def build_manifest(
    image_root: Path,
    dialog_path: Path,
    text_negative_path: Path,
    output_path: Path,
    label_map: dict[str, int],
) -> dict[str, Any]:
    image_records = load_image_records(image_root, label_map)
    text_records = load_dialog_records(dialog_path) + load_text_negatives(text_negative_path)

    manifest = {
        "label_schema": {
            "0": "out_domain",
            "1": "in_domain",
        },
        "image_label_map": label_map,
        "images": [record.__dict__ for record in image_records],
        "texts": [record.__dict__ for record in text_records],
        "stats": {
            "images": len(image_records),
            "texts": len(text_records),
            "image_positive": sum(record.label for record in image_records),
            "image_negative": sum(1 - record.label for record in image_records),
            "text_positive": sum(record.label for record in text_records),
            "text_negative": sum(1 - record.label for record in text_records),
        },
    }
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default="train")
    parser.add_argument("--dialog-data", default="ai_agent_records_202604280912.json")
    parser.add_argument("--text-negatives", default="canteen_test_data_300.json")
    parser.add_argument("--output", default="artifacts/multimodal_manifest.json")
    parser.add_argument(
        "--image-labels",
        default="chart=1,food=1,fruit=0",
        help="Comma separated class labels, for example chart=1,food=1,fruit=0.",
    )
    return parser.parse_args()


def parse_label_map(value: str) -> dict[str, int]:
    result = {}
    for item in value.split(","):
        key, label = item.split("=", 1)
        result[key.strip()] = int(label)
    return result


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        image_root=Path(args.image_root),
        dialog_path=Path(args.dialog_data),
        text_negative_path=Path(args.text_negatives),
        output_path=output_path,
        label_map=parse_label_map(args.image_labels),
    )
    print(json.dumps(manifest["stats"], ensure_ascii=False, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
