from __future__ import annotations

import base64
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from canteen_core import correct_query, extract_time, normalize_text
from production_gateway import OnnxBertBinaryClassifier, ProductionCanteenGateway


TEXT_MODEL_PATH = Path(os.getenv("TEXT_MODEL_PATH", "artifacts/domain_classifier_int8.onnx"))
TEXT_TOKENIZER_PATH = Path(os.getenv("TEXT_TOKENIZER_PATH", "artifacts/domain_classifier_hf"))
VISION_MODEL_PATH = Path(os.getenv("VISION_MODEL_PATH", "artifacts/vision_mobilenetv3_small_160.onnx"))
TEXT_THRESHOLD = float(os.getenv("TEXT_THRESHOLD", "0.5"))
VISION_THRESHOLD = float(os.getenv("VISION_THRESHOLD", "0.5"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "160"))
ORT_INTRA_THREADS = int(os.getenv("ORT_INTRA_THREADS", "1"))
ORT_INTER_THREADS = int(os.getenv("ORT_INTER_THREADS", "1"))
CV_THREADS = int(os.getenv("CV_THREADS", "1"))
DECODE_REDUCTION = int(os.getenv("DECODE_REDUCTION", "2"))

RGB_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
RGB_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class TextRequest(BaseModel):
    query: str


class ImageBase64Request(BaseModel):
    image_base64: str
    query: str | None = None


class ImageClassifier:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing vision model: {model_path}")
        cv2.setNumThreads(CV_THREADS)
        cv2.setUseOptimized(True)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = ORT_INTRA_THREADS
        options.inter_op_num_threads = ORT_INTER_THREADS
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])

    @staticmethod
    def _decode_flag() -> int:
        if DECODE_REDUCTION == 2:
            return cv2.IMREAD_REDUCED_COLOR_2
        if DECODE_REDUCTION == 4:
            return cv2.IMREAD_REDUCED_COLOR_4
        if DECODE_REDUCTION == 8:
            return cv2.IMREAD_REDUCED_COLOR_8
        return cv2.IMREAD_COLOR

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, self._decode_flag())
        if image is None:
            raise ValueError("OpenCV failed to decode image bytes")
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0 / 255.0,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
            ddepth=cv2.CV_32F,
        )
        blob -= RGB_MEAN
        blob /= RGB_STD
        return np.ascontiguousarray(blob, dtype=np.float32)

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        started = time.perf_counter_ns()

        preprocess_started = time.perf_counter_ns()
        tensor = self.preprocess(image_bytes)
        preprocess_ms = (time.perf_counter_ns() - preprocess_started) / 1_000_000

        model_started = time.perf_counter_ns()
        logits = self.session.run(None, {"input": tensor})[0][0]
        model_ms = (time.perf_counter_ns() - model_started) / 1_000_000

        shifted = logits - np.max(logits)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        score = float(probs[1])
        domain = "in_domain" if score >= VISION_THRESHOLD else "out_domain"
        total_ms = (time.perf_counter_ns() - started) / 1_000_000
        return {
            "domain": domain,
            "domain_score": round(score, 4),
            "latency_ms": {
                "preprocess": round(preprocess_ms, 4),
                "model": round(model_ms, 4),
                "total": round(total_ms, 4),
            },
        }


app = FastAPI(title="Canteen Agent Gateway", version="1.0.0")
text_gateway = ProductionCanteenGateway(
    OnnxBertBinaryClassifier(TEXT_MODEL_PATH, TEXT_TOKENIZER_PATH),
    threshold=TEXT_THRESHOLD,
)
image_classifier = ImageClassifier(VISION_MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "text_model": str(TEXT_MODEL_PATH),
        "vision_model": str(VISION_MODEL_PATH),
        "image_size": IMAGE_SIZE,
        "decode_reduction": DECODE_REDUCTION,
        "ort_threads": {
            "intra": ORT_INTRA_THREADS,
            "inter": ORT_INTER_THREADS,
        },
        "cv_threads": CV_THREADS,
    }


@app.post("/intent/text")
def classify_text(request: TextRequest) -> dict[str, Any]:
    return asdict(text_gateway.handle(request.query))


@app.post("/intent/image-base64")
def classify_image_base64(request: ImageBase64Request) -> dict[str, Any]:
    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc
    image_result = image_classifier.predict(image_bytes)
    if not request.query:
        return {"image": image_result}
    text_result = asdict(text_gateway.handle(request.query))
    return {
        "text": text_result,
        "image": image_result,
        "domain": "in_domain" if text_result["domain"] == "in_domain" or image_result["domain"] == "in_domain" else "out_domain",
    }


@app.post("/intent/image-upload")
async def classify_image_upload(file: UploadFile = File(...), query: str | None = None) -> dict[str, Any]:
    image_bytes = await file.read()
    image_result = image_classifier.predict(image_bytes)
    if not query:
        return {"image": image_result}
    text_result = asdict(text_gateway.handle(query))
    return {
        "text": text_result,
        "image": image_result,
        "domain": "in_domain" if text_result["domain"] == "in_domain" or image_result["domain"] == "in_domain" else "out_domain",
    }


@app.post("/time/extract")
def extract_time_endpoint(request: TextRequest) -> dict[str, Any]:
    normalized = normalize_text(request.query)
    corrected, corrections = correct_query(normalized)
    return {
        "original_query": request.query,
        "corrected_query": corrected,
        "corrections": corrections,
        "time": extract_time(corrected),
    }
