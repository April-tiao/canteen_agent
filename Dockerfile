FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    ORT_INTRA_THREADS=1 \
    ORT_INTER_THREADS=1 \
    CV_THREADS=1 \
    IMAGE_SIZE=160 \
    DECODE_REDUCTION=2

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-runtime.txt

COPY canteen_core.py production_gateway.py serve_api.py ./
COPY artifacts/domain_classifier_int8.onnx artifacts/domain_classifier_int8.onnx
COPY artifacts/vision_mobilenetv3_small_160.onnx artifacts/vision_mobilenetv3_small_160.onnx
COPY artifacts/domain_classifier_hf/tokenizer.json artifacts/domain_classifier_hf/tokenizer.json
COPY artifacts/domain_classifier_hf/tokenizer_config.json artifacts/domain_classifier_hf/tokenizer_config.json
COPY artifacts/domain_classifier_hf/special_tokens_map.json artifacts/domain_classifier_hf/special_tokens_map.json
COPY artifacts/domain_classifier_hf/vocab.txt artifacts/domain_classifier_hf/vocab.txt
COPY artifacts/domain_classifier_hf/config.json artifacts/domain_classifier_hf/config.json

EXPOSE 8000

CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
