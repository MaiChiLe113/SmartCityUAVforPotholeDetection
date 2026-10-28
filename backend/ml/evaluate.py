"""Evaluation on the frozen test set and field set (§3.1, §3.3). Owner: Hai (SCRUM-43)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalReport:
    weights: Path
    split: str  # "test" | "field" | "val"
    map50_95: float
    map50: float
    per_class_ap: dict[str, float]
    per_view_map: dict[str, float]  # {"ground": ..., "aerial": ...}
    precision: dict[str, float]
    recall: dict[str, float]


def evaluate(weights: Path, data_yaml: Path, split: str = "test") -> EvalReport:
    """Run `YOLO(weights).val(data=data_yaml, split=split)` and collect overall/per-class metrics."""
    raise NotImplementedError("TODO(SCRUM-43, Hai): evaluate on a split")


def evaluate_by_view(weights: Path, data_yaml: Path, split: str = "test") -> dict[str, float]:
    """mAP50-95 for the ground and aerial subsets separately."""
    raise NotImplementedError("TODO(SCRUM-43, Hai): ground vs aerial evaluation")


def error_analysis(weights: Path, data_yaml: Path, out_dir: Path) -> Path:
    """Save confusion matrix and worst false positives/negatives for the final report."""
    raise NotImplementedError("TODO(SCRUM-43, Hai): error analysis")
