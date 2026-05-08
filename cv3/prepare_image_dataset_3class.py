import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


RANDOM_SEED = 20260507
TRAIN_RATIO = 0.8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

WORKSPACE = Path(__file__).resolve().parent
CHART_ROOT = WORKSPACE / "images"
FOOD_ROOT = WORKSPACE / "images"
FOOD_FALLBACK_ROOT = Path("E:/\u98df\u54c1\u56fe\u7247")
OTHER_ROOT = WORKSPACE / "negative_public_images"
OUTPUT_ROOT = WORKSPACE / "image_dataset_3class"
MANIFEST_PATH = WORKSPACE / "image_dataset_3class_manifest.json"

LABELS = {
    "0_other": 0,
    "1_food": 1,
    "2_chart": 2,
}

CHART_KEYWORDS = [
    "\u56fe\u8868",
    "\u6570\u636e\u53ef\u89c6\u5316",
    "\u67f1\u72b6\u56fe",
    "\u6761\u5f62\u56fe",
    "\u6298\u7ebf\u56fe",
    "\u7edf\u8ba1\u56fe\u8868",
]


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def list_images(root: Path, limit=None):
    if not root.exists():
        return []
    result = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stat().st_size > 1024:
            result.append(p)
            if limit and len(result) >= limit:
                break
    return result


def collect_chart_images():
    return [
        p for p in list_images(CHART_ROOT)
        if any(keyword in str(p.parent) for keyword in CHART_KEYWORDS)
    ]


def collect_food_images():
    chart_set = set(collect_chart_images())
    return [p for p in list_images(FOOD_ROOT) if p not in chart_set]


def collect_food_fallback_images(limit):
    priority_dirs = [
        FOOD_FALLBACK_ROOT / "\u86cb\u7cd5\u56fe\u7247",
        FOOD_FALLBACK_ROOT / "\u96f6\u98df",
        FOOD_FALLBACK_ROOT / "B170\u83dc\u54c1\u56fe\u5e93",
        FOOD_FALLBACK_ROOT / "\u7f8e\u56e2\u56fe\u7247_\u7ca4\u6e2f\u83dc",
        FOOD_FALLBACK_ROOT / "\u7f8e\u56e2\u56fe\u72471129",
    ]
    result = []
    seen = set()
    for root in priority_dirs:
        for p in list_images(root, limit=max(limit * 3, 1000)):
            if p not in seen:
                result.append(p)
                seen.add(p)
                if len(result) >= limit:
                    return result
    for p in list_images(FOOD_FALLBACK_ROOT, limit=max(limit * 5, 5000)):
        if p not in seen:
            result.append(p)
            seen.add(p)
            if len(result) >= limit:
                break
    return result


def collect_unique_readable(paths, target=None, forbidden_hashes=None):
    forbidden_hashes = forbidden_hashes or set()
    result = []
    seen_hashes = set()
    for p in paths:
        if not is_readable_image(p):
            continue
        h = file_sha(p)
        if h in seen_hashes or h in forbidden_hashes:
            continue
        result.append((p, h))
        seen_hashes.add(h)
        if target and len(result) >= target:
            break
    return result


def split_items(items):
    train_size = int(len(items) * TRAIN_RATIO)
    return items[:train_size], items[train_size:]


def reset_output():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    for split in ("train", "test"):
        for label_dir in LABELS:
            (OUTPUT_ROOT / split / label_dir).mkdir(parents=True, exist_ok=True)


def copy_items(items, split, label_dir):
    target_dir = OUTPUT_ROOT / split / label_dir
    label = LABELS[label_dir]
    copied = []
    for index, (source, h) in enumerate(items, 1):
        suffix = ".jpg" if source.suffix.lower() == ".jpeg" else source.suffix.lower()
        dest = target_dir / f"{index:05d}_{h[:12]}{suffix}"
        shutil.copy2(source, dest)
        copied.append({
            "path": str(dest),
            "source_path": str(source),
            "split": split,
            "label": label,
            "label_name": label_dir,
            "sha256": h,
        })
    return copied


def summarize(items):
    return {
        "total": len(items),
        "by_split": dict(Counter(item["split"] for item in items)),
        "by_label": dict(Counter(item["label"] for item in items)),
        "by_split_label": {
            f"{split}_{label}": sum(
                1 for item in items
                if item["split"] == split and item["label"] == label
            )
            for split in ("train", "test")
            for label in (0, 1, 2)
        },
    }


def main():
    random.seed(RANDOM_SEED)

    chart_candidates = collect_chart_images()
    random.shuffle(chart_candidates)
    chart = collect_unique_readable(chart_candidates)

    other_candidates = list_images(OTHER_ROOT)
    random.shuffle(other_candidates)
    other = collect_unique_readable(other_candidates)

    food_candidates_primary = collect_food_images()
    food_candidates = list(food_candidates_primary)
    if len(food_candidates) < min(len(other), len(chart)):
        need = min(len(other), len(chart)) - len(food_candidates)
        food_candidates.extend(collect_food_fallback_images(limit=max(need * 3, need + 500)))
    random.shuffle(food_candidates)
    used_non_food_hashes = {h for _, h in chart} | {h for _, h in other}
    food = collect_unique_readable(food_candidates, forbidden_hashes=used_non_food_hashes)

    class_size = min(len(other), len(food), len(chart))
    if class_size == 0:
        raise RuntimeError("至少一个类别没有可用图片，无法构造三分类数据集")

    other = other[:class_size]
    food = food[:class_size]
    chart = chart[:class_size]

    reset_output()
    manifest = []
    for label_dir, items in [
        ("0_other", other),
        ("1_food", food),
        ("2_chart", chart),
    ]:
        random.shuffle(items)
        train_items, test_items = split_items(items)
        manifest.extend(copy_items(train_items, "train", label_dir))
        manifest.extend(copy_items(test_items, "test", label_dir))

    payload = {
        "meta": {
            "label_definition": {
                "0": "\u5176\u4ed6\u7c7b\uff1a\u4e0d\u5c5e\u4e8e\u667a\u6167\u98df\u5802\u4e1a\u52a1\u7684\u56fe\u7247",
                "1": "\u98df\u7269\u7c7b",
                "2": "\u56fe\u8868\u7c7b",
            },
            "class_dirs": LABELS,
            "train_ratio": TRAIN_RATIO,
            "image_exts": sorted(IMAGE_EXTS),
            "gif_policy": "skip",
            "sources": {
                "0_other": str(OTHER_ROOT),
                "1_food": f"{FOOD_ROOT} non-chart images, then fallback {FOOD_FALLBACK_ROOT}",
                "2_chart": str(CHART_ROOT),
            },
            "available_before_balance": {
                "0_other": len(other_candidates),
                "1_food_candidates_primary": len(food_candidates_primary),
                "1_food_candidates_with_fallback": len(food_candidates),
                "2_chart_candidates": len(chart_candidates),
            },
            "balanced_class_size": class_size,
            "output_root": str(OUTPUT_ROOT),
        },
        "summary": summarize(manifest),
        "items": manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
