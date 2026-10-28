"""Database tables. Owner: Chi (SCRUM-23). Statuses follow the activity diagram (§2)."""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _now() -> datetime:
    return datetime.now(UTC)


class ReportStatus(enum.StrEnum):
    submitted = "submitted"
    no_damage = "no_damage"  # closed by the AI pre-check
    under_review = "under_review"
    rejected = "rejected"  # rejected by the UAV admin
    uav_dispatched = "uav_dispatched"
    confirmed = "confirmed"
    not_confirmed = "not_confirmed"


class PlanStatus(enum.StrEnum):
    proposed = "proposed"
    approved = "approved"
    changes_needed = "changes_needed"
    rejected = "rejected"


class MissionState(enum.StrEnum):
    pending = "pending"
    takeoff = "takeoff"
    surveying = "surveying"
    landing = "landing"
    done = "done"
    failed = "failed"


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    photo_path: Mapped[str] = mapped_column(String)
    status: Mapped[ReportStatus] = mapped_column(Enum(ReportStatus), default=ReportStatus.submitted)
    review_note: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    detections: Mapped[list[Detection]] = relationship(back_populates="report")
    flight_plans: Mapped[list[FlightPlan]] = relationship(back_populates="report")


class Detection(Base):
    """One box from either the server pre-check or the UAV edge device."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int | None] = mapped_column(ForeignKey("reports.id"), nullable=True)
    mission_id: Mapped[int | None] = mapped_column(ForeignKey("missions.id"), nullable=True)
    source: Mapped[str] = mapped_column(String)  # "server" | "edge"
    deployment: Mapped[str] = mapped_column(String)  # "server-fp32" | "jetson-int8" | "pi-int8"
    cls: Mapped[str] = mapped_column(String)  # D00 | D10 | D20 | D40
    confidence: Mapped[float] = mapped_column(Float)
    box_xyxy: Mapped[list[float]] = mapped_column(JSON)
    frame_id: Mapped[str] = mapped_column(String)
    model_version: Mapped[str] = mapped_column(String)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)  # edge detections after geo-location
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    size_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    annotated_path: Mapped[str | None] = mapped_column(String, nullable=True)
    flagged_wrong: Mapped[bool] = mapped_column(Boolean, default=False)  # SCRUM-37
    flag_note: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    report: Mapped[Report | None] = relationship(back_populates="detections")


class FlightPlan(Base):
    __tablename__ = "flight_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("reports.id"))
    drone_id: Mapped[str] = mapped_column(String)
    altitude_m: Mapped[float] = mapped_column(Float)
    survey_radius_m: Mapped[float] = mapped_column(Float)
    route: Mapped[list[dict]] = mapped_column(JSON)  # [{"lat", "lon", "alt", "action"}]
    status: Mapped[PlanStatus] = mapped_column(Enum(PlanStatus), default=PlanStatus.proposed)
    decision_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    report: Mapped[Report] = relationship(back_populates="flight_plans")


class Mission(Base):
    __tablename__ = "missions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    flight_plan_id: Mapped[int] = mapped_column(ForeignKey("flight_plans.id"))
    state: Mapped[MissionState] = mapped_column(Enum(MissionState), default=MissionState.pending)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UavTelemetry(Base):
    __tablename__ = "uav_telemetry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mission_id: Mapped[int] = mapped_column(ForeignKey("missions.id"))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    altitude_m: Mapped[float] = mapped_column(Float)
    heading_deg: Mapped[float] = mapped_column(Float)
    state: Mapped[MissionState] = mapped_column(Enum(MissionState))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
