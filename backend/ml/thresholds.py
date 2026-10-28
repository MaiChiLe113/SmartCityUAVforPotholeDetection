"""Per-class confidence thresholds chosen on validation data. Owner: Hai (SCRUM-26)."""

from __future__ import annotations

from pathlib import Path


def tune_thresholds(weights: Path, data_yaml: Path, metric: str = "f1") -> dict[str, float]:
    """Sweep confidence per class on the val split and return the best threshold per class.

    Copy the result into config.INFER.conf_thresholds (the values file stays the single source).
    """
    raise NotImplementedError("TODO(SCRUM-26, Hai): per-class threshold sweep on val")
