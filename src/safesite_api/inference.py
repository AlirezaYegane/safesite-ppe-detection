from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from safesite_api.schemas import BBox, Detection, PredictResponse


@dataclass(frozen=True)
class InferenceConfig:
    weights: str = "yolov8n.pt"
    conf: float = 0.25


_MODEL: Optional[Any] = None


def _require_ml_deps() -> None:
    try:
        import numpy  # noqa: F401
        import cv2  # noqa: F401
        from ultralytics import YOLO  # noqa: F401
    except Exception as e:
        raise RuntimeError('ML dependencies missing. Install: pip install -e ".[dev,ml]"') from e


def get_model(weights: str) -> Any:
    global _MODEL
    _require_ml_deps()
    if _MODEL is None:
        from ultralytics import YOLO  # local import (CI-safe)

        _MODEL = YOLO(weights)
    return _MODEL


def run_inference(image_bgr: Any, cfg: InferenceConfig = InferenceConfig()) -> PredictResponse:
    model = get_model(cfg.weights)

    results = model.predict(source=image_bgr, conf=cfg.conf, verbose=False)
    r0 = results[0]

    detections: list[Detection] = []
    if r0.boxes is not None:
        for box in r0.boxes:
            cls_id = int(box.cls[0].item())
            label = model.names.get(cls_id, str(cls_id))
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            detections.append(
                Detection(
                    label=label,
                    confidence=confidence,
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

    return PredictResponse(detections=detections, model_version=cfg.weights)
