"""Turn a detection box on a UAV frame into a map location, and merge repeats (§2.1 step 4).
Owner: Hai (SCRUM-34)."""

from __future__ import annotations

from dataclasses import dataclass

from config import UAV, UavConfig


@dataclass
class Pose:
    lat: float
    lon: float
    altitude_m: float
    heading_deg: float
    pitch_deg: float = -90.0  # camera pointing straight down


@dataclass
class GeoDefect:
    lat: float
    lon: float
    size_m: float
    cls: str
    confidence: float
    frame_ids: list[str]


def box_to_latlon(
    box_xyxy: tuple[float, float, float, float],
    image_size: tuple[int, int],
    pose: Pose,
    fov_deg: float = UAV.camera_fov_deg,
) -> tuple[float, float, float]:
    """Project the box centre to ground (lat, lon) and estimate its size in metres
    from altitude and field of view (flat ground, nadir camera)."""
    raise NotImplementedError("TODO(SCRUM-34, Hai): pixel box -> lat/lon/size")


def merge_repeats(defects: list[GeoDefect], cfg: UavConfig = UAV) -> list[GeoDefect]:
    """Merge detections of the same class within cfg.merge_radius_m into one record."""
    raise NotImplementedError("TODO(SCRUM-34, Hai): merge repeat detections")
