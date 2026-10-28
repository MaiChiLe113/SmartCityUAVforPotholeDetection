"""Server-side YOLO pre-check, FP32 PyTorch (§3.1 Table 3.4). Owner: Hai (SCRUM-29)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import INFER, InferConfig


@dataclass
class DetectionResult:
    cls: str
    confidence: float
    box_xyxy: list[float]
    frame_id: str
    model_version: str


class Detector:
    """Loads the weights once and runs predictions with per-class thresholds from config.INFER."""

    def __init__(self, weights: Path | None = None, cfg: InferConfig = INFER) -> None:
        self.cfg = cfg
        self.weights = weights or (Path(cfg.server_weights) if cfg.server_weights else None)
        self.model: Any = None
        self.model_version: str | None = None

    def load(self) -> None:
        """Load `ultralytics.YOLO(self.weights)`; if no weights are set, fetch the MLflow champion."""
        raise NotImplementedError("TODO(SCRUM-29, Hai): load server weights")

    def predict(self, image_path: Path, frame_id: str) -> list[DetectionResult]:
        """Run detection at cfg.imgsz and keep boxes above the per-class threshold."""
        raise NotImplementedError("TODO(SCRUM-29, Hai): run prediction")

    def annotate(self, image_path: Path, results: list[DetectionResult], out_path: Path) -> Path:
        """Save the image with boxes drawn; returns out_path."""
        raise NotImplementedError("TODO(SCRUM-29, Hai): save annotated image")
