"""Retraining triggers (§3.2 Table 3.5).

Triggers explain why a run looked at retraining; retraining itself needs new data (or a manual force),
because retraining on unchanged data cannot produce a better model:
- new_data: batches on disk that the champion was not trained on
- drift_alert: the drift monitor flagged the new images
- enough_new_images: at least MonitorConfig.min_new_images pending images
- review_due: MonitorConfig.scheduled_review_days or more since the last retrain
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from config import MONITOR, MonitorConfig


@dataclass
class TriggerResult:
    triggers: list[str]
    should_retrain: bool
    reason: str


def active_triggers(
    pending_images: int,
    drift_alert: bool,
    last_retrain: datetime | None,
    now: datetime,
    force: bool = False,
    cfg: MonitorConfig = MONITOR,
) -> TriggerResult:
    triggers = []
    if pending_images > 0:
        triggers.append("new_data")
    if drift_alert:
        triggers.append("drift_alert")
    if pending_images >= cfg.min_new_images:
        triggers.append("enough_new_images")
    if last_retrain is None or (now - last_retrain).days >= cfg.scheduled_review_days:
        triggers.append("review_due")
    if force:
        triggers.append("manual_force")

    if force:
        return TriggerResult(triggers, True, "forced from the manual trigger")
    if pending_images > 0:
        return TriggerResult(triggers, True, f"{pending_images} new images the champion has not seen")
    return TriggerResult(triggers, False, "no new data: monitoring only")
