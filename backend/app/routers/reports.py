"""Resident reports and admin review. Owner: Chi (SCRUM-23, SCRUM-30, SCRUM-45)."""

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import todo
from app.schemas import ReportOut, ReportReview

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("", response_model=ReportOut, status_code=201)
async def create_report(
    lat: float = Form(...),
    lon: float = Form(...),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ReportOut:
    """Save photo + coordinates as one report, run the AI pre-check (services.detector),
    store detections, set status to under_review or no_damage.
    Validate type in config.API.allowed_types and size <= config.API.max_upload_mb."""
    raise todo("SCRUM-23/30 create report")


@router.get("", response_model=list[ReportOut])
def list_reports(status: str | None = None, db: Session = Depends(get_db)) -> list[ReportOut]:
    """Report queue for the monitor, newest first, optional status filter."""
    raise todo("SCRUM-23 list reports")


@router.get("/{report_id}", response_model=ReportOut)
def get_report(report_id: int, db: Session = Depends(get_db)) -> ReportOut:
    """One report with its detections (also used by the resident status page, SCRUM-45)."""
    raise todo("SCRUM-23 get report")


@router.patch("/{report_id}/review", response_model=ReportOut)
def review_report(report_id: int, body: ReportReview, db: Session = Depends(get_db)) -> ReportOut:
    """Admin confirms (-> propose a flight plan) or rejects the report."""
    raise todo("SCRUM-29 review report")
