"""Deterministic recorded-frame fixtures; this is not an AI model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from app.services.detector import DetectionResult
from sim.camera import Frame


@dataclass
class FrameResult:
    frame: Frame
    detections: list[DetectionResult]
    error: str | None = None
    source: str = "recorded"
    model_version: str = "mock-v1"


class ProcessingJob(Protocol):
    def poll(self, now: float) -> FrameResult | None:
        """Return None while processing, otherwise a result (including failures)."""
        ...


class FrameProcessor(Protocol):
    def start(self, frame: Frame, now: float) -> ProcessingJob: ...


@dataclass
class _MockJob:
    frame: Frame
    scenario: str
    ready_at: float

    def poll(self, now: float) -> FrameResult | None:
        if now < self.ready_at:
            return None
        if self.scenario == "error":
            return FrameResult(self.frame, [], error="mock processing error")
        detections = []
        if self.scenario == "positive":
            detections.append(DetectionResult("D40", 0.9, [10.0, 20.0, 80.0, 90.0], self.frame.frame_id, "mock-v1"))
        return FrameResult(self.frame, detections)


class MockDetector:
    """Map recorded image filenames to explicit positive, empty or error fixtures."""

    def __init__(self, scenarios: dict[str, str], delay_s: float = 0.0) -> None:
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError("mock delay must be finite and non-negative")
        if any(s not in {"positive", "empty", "error"} for s in scenarios.values()):
            raise ValueError("mock scenario must be positive, empty or error")
        self.scenarios = dict(scenarios)
        self.delay_s = delay_s

    def start(self, frame: Frame, now: float) -> ProcessingJob:
        try:
            scenario = self.scenarios[frame.image_path.name]
        except KeyError as exc:
            raise ValueError(f"no mock scenario for recorded image: {frame.image_path.name}") from exc
        return _MockJob(frame, scenario, now + self.delay_s)
