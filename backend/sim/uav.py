"""Local mission-level waypoint simulation, without API dispatch or flight physics."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

from app.models import MissionState
from app.services.detector import DetectionResult
from app.services.flight_planner import _coordinates, _distance_m, _offset
from config import DATA, UAV, UavConfig
from ml.geo import Pose
from sim.camera import Frame, VirtualCamera
from sim.mock_detector import FrameProcessor, FrameResult


@dataclass(frozen=True)
class Telemetry:
    elapsed_s: float
    pose: Pose
    state: MissionState
    phase: str


class SimulatedUAV:
    """Single-use local engine. A route alone does not grant live dispatch approval."""

    def __init__(
        self,
        route: list[dict],
        api_url: str | None = None,
        mission_id: int | None = None,
        cfg: UavConfig = UAV,
        *,
        camera: VirtualCamera | None = None,
        processor: FrameProcessor | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        telemetry_sink: Callable[[Telemetry], None] | None = None,
    ) -> None:
        self.route = [dict(p) for p in route]
        self.api_url, self.mission_id, self.cfg = api_url, mission_id, cfg
        self.camera, self.processor = camera, processor
        self.clock, self.sleep, self.telemetry_sink = clock, sleep, telemetry_sink
        self.state = MissionState.pending
        self.phase = "pending"
        self.results: list[FrameResult] = []
        self.error: str | None = None
        self.telemetry: list[Telemetry] = []
        self._started = False
        self.pose: Pose | None = None
        self._last_clock: float | None = None

    def _now(self) -> float:
        value = self.clock()
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("clock must return a finite number")
        if self._last_clock is not None and value < self._last_clock:
            raise ValueError("clock must not move backwards")
        self._last_clock = value
        return value

    def _wait(self, seconds: float) -> None:
        before = self._now()
        self.sleep(seconds)
        if self._now() <= before:
            raise ValueError("clock must advance when sleeping")

    def _validate(self) -> None:
        cfg = self.cfg
        if not math.isfinite(cfg.speed_mps) or cfg.speed_mps <= 0:
            raise ValueError("speed must be finite and positive")
        if not math.isfinite(cfg.telemetry_interval_s) or cfg.telemetry_interval_s <= 0:
            raise ValueError("telemetry interval must be finite and positive")
        if not math.isfinite(cfg.processing_timeout_s) or cfg.processing_timeout_s <= 0:
            raise ValueError("processing timeout must be finite and positive")
        actions = ["takeoff", "takeoff", "fly", "survey", "survey", "survey", "fly", "land"]
        if len(self.route) != len(actions) or [p.get("action") for p in self.route] != actions:
            raise ValueError("route must have the local takeoff, target, three survey, return and landing sequence")
        for p in self.route:
            _coordinates(p["lat"], p["lon"], "waypoint")
            if not math.isfinite(p["alt"]) or p["alt"] < 0:
                raise ValueError("waypoint altitude must be finite and non-negative")
        home, takeoff, returning, land = self.route[0], self.route[1], self.route[-2], self.route[-1]
        if any((p["lat"], p["lon"]) != (home["lat"], home["lon"]) for p in (takeoff, returning, land)):
            raise ValueError("takeoff, return and landing must be at home")
        if home["alt"] != 0 or land["alt"] != 0 or takeoff["alt"] <= 0:
            raise ValueError("route must take off from and land on the ground")
        if any(p["alt"] != takeoff["alt"] for p in self.route[2:-1]):
            raise ValueError("route must maintain takeoff altitude until landing")
        if not math.isfinite(cfg.max_target_distance_m) or not 0 < cfg.max_target_distance_m <= 10_000:
            raise ValueError("maximum target distance must be positive and at most 10 km")
        if any(
            _distance_m(home["lat"], home["lon"], p["lat"], p["lon"]) > 2 * cfg.max_target_distance_m
            for p in self.route
        ):
            raise ValueError("waypoint exceeds the local route limit")
        if self.processor is None:
            raise ValueError("a frame processor is required")

    def _emit(self, start: float) -> None:
        assert self.pose is not None
        event = Telemetry(self._now() - start, replace(self.pose), self.state, self.phase)
        self.telemetry.append(event)
        if self.telemetry_sink:
            self.telemetry_sink(replace(event, pose=replace(event.pose)))

    def _move(self, point: dict, start: float) -> None:
        assert self.pose is not None
        origin = replace(self.pose)
        horizontal = _distance_m(origin.lat, origin.lon, point["lat"], point["lon"])
        distance = math.hypot(horizontal, point["alt"] - origin.altitude_m)
        duration = distance / self.cfg.speed_mps
        lon_delta = (point["lon"] - origin.lon + 180) % 360 - 180
        if horizontal:
            phi1, phi2 = math.radians(origin.lat), math.radians(point["lat"])
            dlambda = math.radians(lon_delta)
            heading = (
                math.degrees(
                    math.atan2(
                        math.sin(dlambda) * math.cos(phi2),
                        math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda),
                    )
                )
                % 360
            )
        else:
            heading = origin.heading_deg
        elapsed = 0.0
        while elapsed < duration:
            step = min(self.cfg.telemetry_interval_s, duration - elapsed)
            self._wait(step)
            elapsed += step
            fraction = min(1.0, elapsed / duration)
            lat, lon = (
                _offset(origin.lat, origin.lon, horizontal * fraction, heading)
                if horizontal
                else (origin.lat, origin.lon)
            )
            self.pose = Pose(lat, lon, origin.altitude_m + (point["alt"] - origin.altitude_m) * fraction, heading)
            if fraction == 1.0:
                self.pose.lat, self.pose.lon, self.pose.altitude_m = point["lat"], point["lon"], point["alt"]
            self._emit(start)

    @staticmethod
    def _validate_result(result: FrameResult, frame: Frame) -> FrameResult:
        """Reject malformed or mismatched outputs and detach saved result data."""
        if not isinstance(result, FrameResult) or result.frame != frame:
            raise ValueError("processor returned a result for a different or modified frame")
        if result.error is not None and not isinstance(result.error, str):
            raise ValueError("processor error must be a string or None")
        if not isinstance(result.detections, list):
            raise ValueError("processor detections must be a list")
        if not isinstance(result.model_version, str) or not result.model_version.strip():
            raise ValueError("processor model version must be non-empty")
        if result.source != "recorded":
            raise ValueError("processor source must identify recorded data")
        if result.error is not None and result.detections:
            raise ValueError("failed frame cannot contain detections")
        detections = []
        for d in result.detections:
            if not isinstance(d, DetectionResult) or d.frame_id != frame.frame_id:
                raise ValueError("detection must reference the captured frame")
            if d.cls not in DATA.classes:
                raise ValueError("unknown detection class")
            if (
                not isinstance(d.confidence, (int, float))
                or not math.isfinite(d.confidence)
                or not 0 <= d.confidence <= 1
            ):
                raise ValueError("detection confidence must be finite and between zero and one")
            if (
                not isinstance(d.box_xyxy, list)
                or len(d.box_xyxy) != 4
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in d.box_xyxy)
                or d.box_xyxy[0] >= d.box_xyxy[2]
                or d.box_xyxy[1] >= d.box_xyxy[3]
            ):
                raise ValueError("detection box must have four finite, non-negative ordered coordinates")
            if (
                not isinstance(d.model_version, str)
                or not d.model_version.strip()
                or d.model_version != result.model_version
            ):
                raise ValueError("detection model version must match frame result")
            detections.append(replace(d, box_xyxy=list(d.box_xyxy)))
        return FrameResult(
            replace(frame, pose=replace(frame.pose)), detections, result.error, result.source, result.model_version
        )

    def run(self) -> list[FrameResult]:
        """Run once, preserving successes and a failure result if processing fails."""
        if self._started:
            raise RuntimeError("simulator is single-use")
        self._started = True
        start = None
        try:
            start = self._now()
            self._validate()
            if self.camera is None:
                self.camera = VirtualCamera(self.cfg)
            home = self.route[0]
            self.pose = Pose(home["lat"], home["lon"], home["alt"], 0.0)
            self._emit(start)
            for index, point in enumerate(self.route[1:], 1):
                if index == 1:
                    self.state, self.phase = MissionState.takeoff, "takeoff"
                elif index == len(self.route) - 1:
                    self.state, self.phase = MissionState.landing, "landing"
                elif index == len(self.route) - 2:
                    self.state, self.phase = MissionState.surveying, "return"
                elif point["action"] == "survey":
                    self.state, self.phase = MissionState.surveying, "survey"
                else:
                    self.state, self.phase = MissionState.surveying, "outbound"
                self._emit(start)
                self._move(point, start)
                if point["action"] == "survey":
                    captured = self.camera.capture(self.pose)
                    frame = replace(captured, pose=replace(captured.pose))
                    processor_frame = replace(frame, pose=replace(frame.pose))
                    assert self.processor is not None
                    try:
                        job_start = self._now()
                        job = self.processor.start(processor_frame, job_start)
                        self.phase = "processing"
                        self._emit(start)
                        while True:
                            now = self._now()
                            if now > job_start + self.cfg.processing_timeout_s:
                                raise TimeoutError("frame processing timed out")
                            result = job.poll(now)
                            if result is not None:
                                result = self._validate_result(result, frame)
                                break
                            remaining = job_start + self.cfg.processing_timeout_s - self._now()
                            if remaining <= 0:
                                raise TimeoutError("frame processing timed out")
                            self._wait(min(self.cfg.telemetry_interval_s, remaining))
                            self._emit(start)
                    except Exception as exc:
                        self.results.append(FrameResult(frame, [], error=str(exc) or type(exc).__name__))
                        raise
                    self.results.append(result)
                    if result.error is not None:
                        raise RuntimeError(result.error or "processor returned an empty error")
            self.state, self.phase = MissionState.done, "done"
            self._emit(start)
        except Exception as exc:
            self.state, self.phase, self.error = MissionState.failed, "failed", str(exc) or type(exc).__name__
            if self.pose is not None and start is not None:
                try:
                    self._emit(start)
                except Exception:
                    pass  # The telemetry sink may itself be the cause of failure.
        return list(self.results)

    def fly(self) -> Iterator[Frame]:
        """Run eagerly, then return an iterator over the processed frames."""
        return iter(result.frame for result in self.run())
