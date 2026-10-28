"""Mission dispatch, live telemetry and edge results. Owner: Hoan / Hai (SCRUM-33, SCRUM-34, SCRUM-39)."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import todo
from app.schemas import MissionOut, MissionResultsIn, TelemetryIn, TelemetryOut

router = APIRouter(prefix="/missions", tags=["missions"])


@router.get("", response_model=list[MissionOut])
def list_missions(active: bool = False, db: Session = Depends(get_db)) -> list[MissionOut]:
    raise todo("SCRUM-39 list missions")


@router.post("/{mission_id}/dispatch", response_model=MissionOut)
def dispatch_mission(mission_id: int, db: Session = Depends(get_db)) -> MissionOut:
    """Start the simulated UAV (sim.uav) for an approved plan; report -> uav_dispatched."""
    raise todo("SCRUM-33 dispatch mission")


@router.post("/{mission_id}/telemetry", response_model=TelemetryOut, status_code=201)
def push_telemetry(mission_id: int, body: TelemetryIn, db: Session = Depends(get_db)) -> TelemetryOut:
    """Called by the simulator every config.UAV.telemetry_interval_s."""
    raise todo("SCRUM-33 store telemetry")


@router.get("/{mission_id}/telemetry/latest", response_model=TelemetryOut)
def latest_telemetry(mission_id: int, db: Session = Depends(get_db)) -> TelemetryOut:
    """Polled by the dashboard map (SCRUM-39)."""
    raise todo("SCRUM-39 latest telemetry")


@router.post("/{mission_id}/results", response_model=MissionOut)
def upload_results(mission_id: int, body: MissionResultsIn, db: Session = Depends(get_db)) -> MissionOut:
    """Edge detections after landing: geo-locate (ml.geo), merge repeats, store,
    mark the report confirmed / not_confirmed."""
    raise todo("SCRUM-34 upload edge results")
