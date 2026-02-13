from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox


class PredictResponse(BaseModel):
    detections: list[Detection]
    model_version: str
