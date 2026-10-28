"""API request/response models. Keep in sync with frontend/lib/types.ts."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models import MissionState, PlanStatus, ReportStatus


class _Orm(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class DetectionOut(_Orm):
    id: int
    source: str
    deployment: str
    cls: str
    confidence: float
    box_xyxy: list[float]
    frame_id: str
    model_version: str
    lat: float | None = None
    lon: float | None = None
    size_m: float | None = None
    annotated_url: str | None = None
    flagged_wrong: bool


class ReportOut(_Orm):
    id: int
    lat: float
    lon: float
    photo_url: str
    status: ReportStatus
    review_note: str | None = None
    created_at: datetime
    updated_at: datetime
    detections: list[DetectionOut] = []


class ReportReview(BaseModel):
    """UAV admin decision on a report (SCRUM-29)."""

    confirmed: bool
    note: str | None = None


class FlagIn(BaseModel):
    """Mark a detection as wrong (SCRUM-37)."""

    note: str | None = None


class Waypoint(BaseModel):
    lat: float
    lon: float
    alt: float
    action: str  # "takeoff" | "fly" | "survey" | "land"


class FlightPlanOut(_Orm):
    id: int
    report_id: int
    drone_id: str
    altitude_m: float
    survey_radius_m: float
    route: list[Waypoint]
    status: PlanStatus
    decision_reason: str | None = None
    decided_at: datetime | None = None
    created_at: datetime


class PlanDecision(BaseModel):
    """Reject and changes-needed both require a reason (SCRUM-31)."""

    reason: str = Field(min_length=1)


class MissionOut(_Orm):
    id: int
    flight_plan_id: int
    state: MissionState
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TelemetryIn(BaseModel):
    lat: float
    lon: float
    altitude_m: float
    heading_deg: float
    state: MissionState


class TelemetryOut(_Orm, TelemetryIn):
    id: int
    mission_id: int
    ts: datetime


class EdgeDetectionIn(BaseModel):
    """One detection uploaded by the edge device after landing (§3.4 output check)."""

    frame_id: str
    cls: str
    confidence: float
    box_xyxy: list[float]
    model_version: str
    deployment: str
    pose: dict  # lat, lon, altitude_m, heading_deg
    image_size: tuple[int, int]


class MissionResultsIn(BaseModel):
    detections: list[EdgeDetectionIn]


class MetricsOut(BaseModel):
    """Monitoring signals (§3.3 Table 3.6)."""

    reports_total: int
    correction_rate: float | None
    uav_agreement_rate: float | None
    mean_confidence_by_class: dict[str, float]
    model_version: str | None
    retrain_triggers: list[str]
