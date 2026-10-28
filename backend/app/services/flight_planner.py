"""Propose a UAV flight plan for a confirmed report (§2.1 step 2). Owner: Hoan (SCRUM-27)."""

from __future__ import annotations

from dataclasses import dataclass

from config import UAV, UavConfig


@dataclass
class ProposedPlan:
    drone_id: str
    altitude_m: float
    survey_radius_m: float
    route: list[dict]  # [{"lat", "lon", "alt", "action"}], takeoff -> target -> survey -> land


class FlightPlanner:
    def __init__(self, cfg: UavConfig = UAV) -> None:
        self.cfg = cfg

    def propose(self, target_lat: float, target_lon: float, previous_reason: str | None = None) -> ProposedPlan:
        """Takeoff at home, fly to the pin, survey pattern of cfg.survey_radius_m, return and land.
        `previous_reason` is the admin's change request, if any."""
        raise NotImplementedError("TODO(SCRUM-27, Hoan): generate flight plan")
