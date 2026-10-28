"""Retraining triggers (§3.2 Table 3.5). Never trains automatically: it only opens a proposal
for a human to approve. Owner: Hai (SCRUM-46, backlog)."""

from __future__ import annotations

from sqlalchemy.orm import Session

from config import MONITOR, MonitorConfig


def active_triggers(db: Session, cfg: MonitorConfig = MONITOR) -> list[str]:
    """Return the names of triggers that fire: corrections, uav_disagreement, drift, scheduled."""
    raise NotImplementedError("TODO(SCRUM-46, Hai): evaluate retraining triggers")


def open_retrain_proposal(triggers: list[str]) -> str:
    """Create the Jira issue (or a local record) proposing retraining. Returns its id/link."""
    raise NotImplementedError("TODO(SCRUM-46, Hai): open retraining proposal")
