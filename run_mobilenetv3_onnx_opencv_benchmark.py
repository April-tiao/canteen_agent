from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from canteen_core import percentile
from run_image_mobilenetv3_experiment import build_model


os.environ.setdefault("TORCH_HOME", str(Path("artifacts/model_cache/torch").resolve()))

RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
RGB_MEAN_CHW = RGB_MEAN.reshape(1, 3, 1, 1)
RGB_STD_CHW = RGB_STD.reshape(1, 3, 1, 1)


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "avg": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "under_50ms": sum(1 for value in values if value <= 50) / len(values),
    }


def export_onnx(pt_path: Path, onnx_path: Path, image_size: int) -> None:
    if onnx_path.exists():
        return
    model = build_model(freeze_features=True)
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=14,
        dynamo=False,
    )


def build_session(onnx_path: Path, intra_threads: int, inter_threads: int, sequential: bool) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = inter_threads
    if sequential:
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])


def decode_flag(reduction: int) -> int:
    if reduction == 2:
        return cv2.IMREAD_REDUCED_COLOR_2
    if reduction == 4:
        return cv2.IMREAD_REDUCED_COLOR_4
    if reduction == 8:
        return cv2.IMREAD_REDUCED_COLOR_8
    return cv2.IMREAD_COLOR


def preprocess_image_bytes(image_bytes: bytes, image_size: int, reduction: int) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, decode_flag(reduction))
    if image is None:
        raise ValueError("OpenCV failed to decode image bytes")
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 255.0,
        size=(image_size, image_size),
        mean=(0.0, 0.0, 0.0),
        swapRB=True,
        crop=False,
        ddepth=cv2.CV_32F,
    )
    blob -= RGB_MEAN_CHW
    blob /= RGB_STD_CHW
    return np.ascontiguousarray(blob, dtype=np.float32)


def predict(session: ort.InferenceSession, tensor: np.ndarray) -> int:
    logits = session.run(None, {"input": tensor})[0][0]
    return int(np.argmax(logits))


def benchmark(
    session: ort.InferenceSession,
    rows: list[dict[str, Any]],
    image_size: int,
    repeats: int,
    input_mode: str,
    reduction: int,
) -> tuple[list[int], list[int], dict[str, list[float]]]:
    sample_rows = (rows * ((repeats // len(rows)) + 1))[:repeats]
    cached_bytes = {row["path"]: Path(row["path"]).read_bytes() for row in rows}

    # Warm up both preprocess and ORT kernels.
    for row in sample_rows[:20]:
        image_bytes = cached_bytes[row["path"]]
        tensor = preprocess_image_bytes(image_bytes, image_size, reduction)
        predict(session, tensor)

    preds: list[int] = []
    golds: list[int] = []
    timings: dict[str, list[float]] = {
        "read": [],
        "preprocess": [],
        "model": [],
        "total": [],
    }

    for row in sample_rows:
        total_start = time.perf_counter_ns()

        read_start = time.perf_counter_ns()
        if input_mode == "path":
            image_bytes = Path(row["path"]).read_bytes()
        else:
            image_bytes = cached_bytes[row["path"]]
        read_ms = (time.perf_counter_ns() - read_start) / 1_000_000

        preprocess_start = time.perf_counter_ns()
        tensor = preprocess_image_bytes(image_bytes, image_size, reduction)
        preprocess_ms = (time.perf_counter_ns() - preprocess_start) / 1_000_000

        model_start = time.perf_counter_ns()
        pred = predict(session, tensor)
        model_ms = (time.perf_counter_ns() - model_start) / 1_000_000

        total_ms = (time.perf_counter_ns() - total_start) / 1_000_000

        timings["read"].append(read_ms)
        timings["preprocess"].append(preprocess_ms)
        timings["model"].append(model_ms)
        timings["total"].append(total_ms)
        preds.append(pred)
        golds.append(row["label"])

    return preds, golds, timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/multimodal_manifest.json")
    parser.add_argument("--pt", default="artifacts/vision_mobilenetv3_small_160.pt")
    parser.add_argument("--onnx", default="artifacts/vision_mobilenetv3_small_160.onnx")
    parser.add_argument("--result", default="artifacts/mobilenetv3_onnx_opencv_160_results.json")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--input-mode", choices=["bytes", "path"], default="bytes")
    parser.add_argument("--decode-reduction", type=int, choices=[1, 2, 4, 8], default=2)
    parser.add_argument("--cv-threads", type=int, default=1)
    parser.add_argument("--intra-threads", type=int, default=1)
    parser.add_argument("--inter-threads", type=int, default=1)
    parser.add_argument("--parallel", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(args.cv_threads)
    cv2.setUseOptimized(True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = manifest["images"]
    labels = [row["label"] for row in rows]
    _, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=42,
        stratify=labels,
    )

    export_onnx(Path(args.pt), Path(args.onnx), args.image_size)
    session = build_session(
        Path(args.onnx),
        intra_threads=args.intra_threads,
        inter_threads=args.inter_threads,
        sequential=not args.parallel,
    )
    preds, golds, timings = benchmark(
        session,
        test_rows,
        args.image_size,
        args.repeats,
        args.input_mode,
        args.decode_reduction,
    )

    result = {
        "scheme": "MobileNetV3-Small ONNX Runtime + OpenCV bytes pipeline",
        "input_mode": args.input_mode,
        "image_size": args.image_size,
        "onnx": args.onnx,
        "test_size": len(test_rows),
        "repeats": args.repeats,
        "ort": {
            "intra_threads": args.intra_threads,
            "inter_threads": args.inter_threads,
            "execution_mode": "parallel" if args.parallel else "sequential",
        },
        "opencv": {
            "cv_threads": args.cv_threads,
            "decode_reduction": args.decode_reduction,
            "preprocess": "cv2.imdecode + cv2.dnn.blobFromImage + per-channel normalize",
        },
        "accuracy": accuracy_score(golds, preds),
        "classification_report": classification_report(golds, preds, target_names=["out_domain", "in_domain"], output_dict=True),
        "latency_ms": {name: latency_summary(values) for name, values in timings.items()},
    }
    result_path = Path(args.result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
