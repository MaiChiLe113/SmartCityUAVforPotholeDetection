"""Monitoring signals and drift report. Owner: Chi (SCRUM-36)."""

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import todo
from app.schemas import MetricsOut

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/metrics", response_model=MetricsOut)
def metrics(db: Session = Depends(get_db)) -> MetricsOut:
    """Correction rate, UAV agreement, mean confidence per class, active triggers (§3.3)."""
    raise todo("SCRUM-36 metrics")


@router.get("/drift-report")
def drift_report() -> FileResponse:
    """Latest Evidently HTML report written by run/drift_report.py."""
    raise todo("SCRUM-36 drift report")
