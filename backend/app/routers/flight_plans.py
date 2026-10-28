"""Flight plan proposal and approval. Owner: Hoan (SCRUM-27, SCRUM-31). No flight without approval."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import todo
from app.schemas import FlightPlanOut, PlanDecision

router = APIRouter(tags=["flight-plans"])


@router.post("/reports/{report_id}/flight-plan", response_model=FlightPlanOut, status_code=201)
def propose_flight_plan(report_id: int, db: Session = Depends(get_db)) -> FlightPlanOut:
    """Generate a plan with services.flight_planner for a confirmed report."""
    raise todo("SCRUM-27 propose flight plan")


@router.get("/flight-plans/{plan_id}", response_model=FlightPlanOut)
def get_flight_plan(plan_id: int, db: Session = Depends(get_db)) -> FlightPlanOut:
    raise todo("SCRUM-31 get flight plan")


@router.post("/flight-plans/{plan_id}/approve", response_model=FlightPlanOut)
def approve_flight_plan(plan_id: int, db: Session = Depends(get_db)) -> FlightPlanOut:
    """Approve and create a pending Mission. Log the decision."""
    raise todo("SCRUM-31 approve flight plan")


@router.post("/flight-plans/{plan_id}/reject", response_model=FlightPlanOut)
def reject_flight_plan(plan_id: int, body: PlanDecision, db: Session = Depends(get_db)) -> FlightPlanOut:
    """Reject with a required reason. Log the decision."""
    raise todo("SCRUM-31 reject flight plan")


@router.post("/flight-plans/{plan_id}/request-changes", response_model=FlightPlanOut)
def request_changes(plan_id: int, body: PlanDecision, db: Session = Depends(get_db)) -> FlightPlanOut:
    """Send back for changes; the planner proposes a new version."""
    raise todo("SCRUM-31 request plan changes")
