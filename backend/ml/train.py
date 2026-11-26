"""Training wrapper (§3.1d). Metrics and weights are kept in the Ultralytics run folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainResult:
    model_name: str
    best_weights: Path
    metrics: dict[str, float]  # mAP50-95, mAP50, per-class AP on val


def train_one(model: Any, train_args: dict) -> TrainResult:
    """Train one Ultralytics YOLO model with `model.train(**train_args)`."""
    results = model.train(**train_args)
    metrics = {"mAP50-95": float(results.box.map), "mAP50": float(results.box.map50)}
    for class_index, ap in zip(results.ap_class_index, results.box.ap, strict=True):
        metrics[f"AP50-95/{results.names[int(class_index)]}"] = float(ap)
    return TrainResult(
        model_name=Path(str(model.ckpt_path)).name,
        best_weights=Path(model.trainer.best),
        metrics=metrics,
    )


def compare(results: list[TrainResult], out_csv: Path) -> Path:
    """Write the three-model comparison table (val mAP50-95, per-class AP, speed) to CSV."""
    raise NotImplementedError("TODO(SCRUM-25, Hai): write comparison table")
