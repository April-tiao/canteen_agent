from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESULT_FILES = [
    Path("artifacts/mobilenetv3_small_160_results.json"),
    Path("artifacts/mobileclip2_s0_linear_probe_results.json"),
    Path("artifacts/fastvlm_0_5b_contrast_results.json"),
]


def compact_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    latency = data.get("latency_ms", {})
    return {
        "scheme": data.get("scheme"),
        "device": data.get("device"),
        "accuracy": data.get("accuracy"),
        "latency_ms": latency,
        "source": str(path),
    }


def main() -> None:
    results = [item for path in RESULT_FILES if (item := compact_result(path))]
    output = {
        "note": "Image experiments use train/chart, train/food as in_domain and train/fruit as out_domain by default. FastVLM is a contrast group, not a 50ms candidate.",
        "results": results,
    }
    path = Path("artifacts/multimodal_experiment_summary.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
