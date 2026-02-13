from __future__ import annotations
from typing import Any
from fastapi import FastAPI, File, HTTPException, UploadFile
from safesite_api.inference import InferenceConfig, run_inference
from safesite_api.schemas import PredictResponse

app = FastAPI(title="SafeSite API", version="0.1.0")

def _decode_image(content: bytes) -> Any:
    """
    Decode bytes into an OpenCV BGR image.
    Kept behind a function so tests can monkeypatch it in CI.
    """
    try:
        import numpy as np
        import cv2
    except Exception as e:
        raise RuntimeError('ML dependencies missing. Install: pip install -e ".[dev,ml]"') from e

    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Only PNG/JPEG images are supported.")

    content = await file.read()

    try:
        img = _decode_image(content)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    cfg = InferenceConfig(weights="yolov8n.pt", conf=0.25)
    return run_inference(img, cfg=cfg)
