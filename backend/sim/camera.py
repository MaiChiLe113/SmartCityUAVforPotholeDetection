"""Virtual camera: one frame with pose, field of view and timestamp per capture. Owner: Hoan (SCRUM-22)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from config import UAV, UavConfig
from ml.geo import Pose


@dataclass
class Frame:
    frame_id: str
    image_path: Path
    pose: Pose
    fov_deg: float
    ts: datetime


class VirtualCamera:
    def __init__(self, cfg: UavConfig = UAV) -> None:
        self.cfg = cfg

    def capture(self, pose: Pose) -> Frame:
        """Return a frame for this pose: e.g. pick an aerial test image near the target,
        or render from the simulation environment chosen in SCRUM-22."""
        raise NotImplementedError("TODO(SCRUM-22, Hoan): capture frame from simulation")
