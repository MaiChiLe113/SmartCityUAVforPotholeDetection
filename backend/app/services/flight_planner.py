"""Propose a short, deterministic UAV flight plan for a confirmed report."""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import UAV, UavConfig

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class ProposedPlan:
    drone_id: str
    altitude_m: float
    survey_radius_m: float
    route: list[dict]  # lat, lon, alt (metres), action


def _coordinates(lat: float, lon: float, name: str) -> None:
    if not math.isfinite(lat) or not -85 <= lat <= 85:
        raise ValueError(f"{name} latitude must be finite and between -85 and 85 degrees")
    if not math.isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError(f"{name} longitude must be finite and between -180 and 180 degrees")


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def _offset(lat: float, lon: float, distance_m: float, bearing_deg: float) -> tuple[float, float]:
    """Destination on a sphere; normalise longitude when crossing the antimeridian."""
    phi, lam, bearing = map(math.radians, (lat, lon, bearing_deg))
    angular = distance_m / EARTH_RADIUS_M
    phi2 = math.asin(math.sin(phi) * math.cos(angular) + math.cos(phi) * math.sin(angular) * math.cos(bearing))
    lam2 = lam + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(phi),
        math.cos(angular) - math.sin(phi) * math.sin(phi2),
    )
    return math.degrees(phi2), (math.degrees(lam2) + 180) % 360 - 180


class FlightPlanner:
    def __init__(self, cfg: UavConfig = UAV) -> None:
        self.cfg = cfg

    def propose(self, target_lat: float, target_lon: float, previous_reason: str | None = None) -> ProposedPlan:
        """Build a local route with three evenly spaced survey points around the target.

        The previous change request is accepted for API compatibility but does not alter
        this deterministic route. This planner neither approves nor dispatches a mission.
        """
        cfg = self.cfg
        _coordinates(cfg.home_lat, cfg.home_lon, "home")
        _coordinates(target_lat, target_lon, "target")
        for name, value in (
            ("altitude", cfg.default_altitude_m),
            ("survey radius", cfg.survey_radius_m),
            ("speed", cfg.speed_mps),
            ("maximum route distance", cfg.max_target_distance_m),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if cfg.max_target_distance_m > 10_000:
            raise ValueError("maximum target distance must not exceed 10 km")
        if cfg.survey_radius_m > cfg.max_target_distance_m:
            raise ValueError("survey radius must not exceed the maximum target distance")
        if _distance_m(cfg.home_lat, cfg.home_lon, target_lat, target_lon) > cfg.max_target_distance_m:
            raise ValueError("target exceeds the configured maximum distance from home")
        if not cfg.drone_ids or not cfg.drone_ids[0]:
            raise ValueError("at least one non-empty drone ID is required")

        def point(lat: float, lon: float, alt: float, action: str) -> dict:
            return {"lat": lat, "lon": lon, "alt": alt, "action": action}

        altitude = cfg.default_altitude_m
        route = [
            point(cfg.home_lat, cfg.home_lon, 0.0, "takeoff"),
            point(cfg.home_lat, cfg.home_lon, altitude, "takeoff"),
            point(target_lat, target_lon, altitude, "fly"),
        ]
        for bearing in (0, 120, 240):
            lat, lon = _offset(target_lat, target_lon, cfg.survey_radius_m, bearing)
            _coordinates(lat, lon, "survey waypoint")
            route.append(point(lat, lon, altitude, "survey"))
        route.extend(
            [point(cfg.home_lat, cfg.home_lon, altitude, "fly"), point(cfg.home_lat, cfg.home_lon, 0.0, "land")]
        )
        return ProposedPlan(cfg.drone_ids[0], altitude, cfg.survey_radius_m, route)
