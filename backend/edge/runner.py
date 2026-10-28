"""On-device detection during a flight (§3.4 "Running on the UAV"). Owner: Hai (SCRUM-34).

Keeps frames if detection fails mid-flight and processes them after landing.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sim.camera import Frame


class EdgeRunner:
    def __init__(self, package_path: Path, deployment: str, model_version: str) -> None:
        self.package_path = package_path
        self.deployment = deployment  # "jetson-int8" | "pi-int8"
        self.model_version = model_version

    def run(self, frames: Iterable[Frame], out_json: Path) -> Path:
        """Detect on each frame (batch 1, 640 px) and append box, class, confidence, frame id,
        model version and pose to out_json."""
        raise NotImplementedError("TODO(SCRUM-34, Hai): per-frame edge detection")

    def upload(self, out_json: Path, api_url: str, mission_id: int) -> None:
        """POST results to {api_url}/missions/{mission_id}/results after landing."""
        raise NotImplementedError("TODO(SCRUM-34, Hai): upload results")
