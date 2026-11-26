"""Weekly data drift report with Evidently (§3.3). Owner: Chi (SCRUM-36)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import MONITOR, MonitorConfig


@dataclass
class DriftResult:
    drifted_features: list[str]
    share_drifted: float
    alert: bool
    html_report: Path


def image_features(image: Path) -> dict[str, float | str]:
    """Simple features per image: brightness, contrast, blur (Laplacian variance), resolution, view."""
    raise NotImplementedError("TODO(SCRUM-36, Chi): compute image features")


class DriftMonitor:
    def __init__(self, cfg: MonitorConfig = MONITOR) -> None:
        self.cfg = cfg

    def build_reference(self, train_images: list[Path]) -> Path:
        """Compute features for the training set once and save them as the reference table."""
        raise NotImplementedError("TODO(SCRUM-36, Chi): reference feature table")

    def run(self, recent_images: list[Path], out_dir: Path) -> DriftResult:
        """Compare recent features with the reference (KS test per feature) and save the HTML report.
        Alert when share of drifted features > cfg.drift_feature_share."""
        raise NotImplementedError("TODO(SCRUM-36, Chi): Evidently drift report")
