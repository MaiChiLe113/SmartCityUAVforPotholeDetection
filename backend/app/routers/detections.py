"""Detection results and "wrong detection" flags. Owner: Hai / Chi (SCRUM-29, SCRUM-37)."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import todo
from app.schemas import DetectionOut, FlagIn

router = APIRouter(tags=["detections"])


@router.get("/reports/{report_id}/detections", response_model=list[DetectionOut])
def list_detections(report_id: int, db: Session = Depends(get_db)) -> list[DetectionOut]:
    raise todo("SCRUM-29 list detections")


@router.post("/detections/{detection_id}/flag", response_model=DetectionOut)
def flag_detection(detection_id: int, body: FlagIn, db: Session = Depends(get_db)) -> DetectionOut:
    """Mark as wrong; feeds the corrections-rate retraining trigger (§3.2)."""
    raise todo("SCRUM-37 flag wrong detection")
