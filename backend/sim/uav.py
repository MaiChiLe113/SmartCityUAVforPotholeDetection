"""Simulated UAV: takeoff -> fly route -> survey -> land (§2.1 step 3). Owner: Hoan (SCRUM-33)."""

from __future__ import annotations

from collections.abc import Iterator

from config import UAV, UavConfig
from sim.camera import Frame, VirtualCamera


class SimulatedUAV:
    def __init__(self, route: list[dict], api_url: str, mission_id: int, cfg: UavConfig = UAV) -> None:
        self.route = route
        self.api_url = api_url
        self.mission_id = mission_id
        self.cfg = cfg
        self.camera = VirtualCamera(cfg)

    def fly(self) -> Iterator[Frame]:
        """Move along the route at cfg.speed_mps, POST telemetry to
        {api_url}/missions/{mission_id}/telemetry every cfg.telemetry_interval_s,
        and yield a camera frame every cfg.frame_interval_s while surveying."""
        raise NotImplementedError("TODO(SCRUM-33, Hoan): fly the mission")
