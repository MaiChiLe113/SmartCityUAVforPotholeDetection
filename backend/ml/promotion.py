"""Champion vs challenger promotion checks (§3.2). Owner: Hai (SCRUM-46)."""

from __future__ import annotations

from dataclasses import dataclass

from config import PROMOTION, PromotionConfig
from ml.evaluate import EvalReport


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str]  # one line per failed or passed check


def check_promotion(
    champion_test: EvalReport,
    challenger_test: EvalReport,
    champion_field: EvalReport,
    challenger_field: EvalReport,
    edge_checks_passed: bool,
    cfg: PromotionConfig = PROMOTION,
) -> PromotionDecision:
    """Promote only if: higher mAP50-95 overall and on aerial, no class AP drop > max_class_ap_drop,
    no worse on the field set, and the INT8 copy passed the edge checks."""
    raise NotImplementedError("TODO(SCRUM-46, Hai): promotion checks")
