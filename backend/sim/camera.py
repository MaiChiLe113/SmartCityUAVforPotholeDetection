"""Replay development images with simulated pose metadata; no real capture is implied."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
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
    def __init__(
        self,
        cfg: UavConfig = UAV,
        *,
        image_dir: Path | None = None,
        image_paths: Sequence[Path] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if image_dir is not None and image_paths is not None:
            raise ValueError("provide either image_dir or image_paths, not both")
        if not math.isfinite(cfg.camera_fov_deg) or not 0 < cfg.camera_fov_deg < 180:
            raise ValueError("camera field of view must be between 0 and 180 degrees")
        if image_paths is None:
            folder = Path(image_dir if image_dir is not None else cfg.recorded_images_dir)
            if not folder.is_dir():
                raise FileNotFoundError(f"recorded image directory does not exist: {folder}")
            paths = sorted(
                (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                key=lambda p: (p.name.casefold(), p.name),
            )
        else:
            paths = [Path(p) for p in image_paths]
        if not paths:
            raise ValueError("no recorded images provided")
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"recorded image does not exist: {path}")
        self.cfg = cfg
        self._paths = paths
        self._next = 0
        self._clock = clock or (lambda: datetime.now(UTC))

    def capture(self, pose: Pose) -> Frame:
        """Associate the next recorded image with the supplied simulated pose and time."""
        if self._next >= len(self._paths):
            raise EOFError("recorded images exhausted")
        path = self._paths[self._next]
        if not path.is_file():
            raise FileNotFoundError(f"recorded image does not exist: {path}")
        ts = self._clock()
        self._next += 1
        return Frame(f"frame-{self._next:06d}", path, replace(pose), self.cfg.camera_fov_deg, ts)
