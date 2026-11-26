"""Champion vs challenger promotion checks (§3.2), decided on validation only.

Both models are scored on the same validation list with ml.evaluate.evaluate_val_subsets. The frozen test
set is scored afterwards for the record and never changes the decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import PROMOTION, PromotionConfig


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str]  # one line per check: "PASS ...", "FAIL ..." or "N/A ..."


def check_promotion(champion: dict, challenger: dict, cfg: PromotionConfig = PROMOTION) -> PromotionDecision:
    """Promote only if every available check passes.

    champion / challenger: evaluate_val_subsets output, with at least the "all" subset and, for the aerial
    check, an "aerial" subset. Each subset has "mAP50-95" and per-class "AP50-95".
    """
    reasons: list[str] = []
    ok = True

    def check(passed: bool, text: str) -> None:
        nonlocal ok
        ok &= passed
        reasons.append(("PASS " if passed else "FAIL ") + text)

    old, new = champion["all"]["mAP50-95"], challenger["all"]["mAP50-95"]
    check(new - old > cfg.min_map_gain, f"val mAP50-95 {new:.4f} vs champion {old:.4f} (needs > +{cfg.min_map_gain})")

    if cfg.require_aerial_gain:
        if "aerial" in champion and "aerial" in challenger:
            a_old, a_new = champion["aerial"]["mAP50-95"], challenger["aerial"]["mAP50-95"]
            check(a_new >= a_old, f"aerial val mAP50-95 {a_new:.4f} vs champion {a_old:.4f} (must not fall)")
        else:
            reasons.append("N/A aerial check: no aerial images in validation")

    for cls, ap_old in sorted(champion["all"]["AP50-95"].items()):
        ap_new = challenger["all"]["AP50-95"].get(cls, 0.0)
        check(
            ap_old - ap_new <= cfg.max_class_ap_drop,
            f"{cls} AP50-95 {ap_new:.4f} vs {ap_old:.4f} (max drop {cfg.max_class_ap_drop})",
        )

    if cfg.require_field_no_worse:
        reasons.append("N/A field-set check: no field set collected yet")
    if cfg.require_edge_checks:
        reasons.append("N/A edge export check: edge export is not part of the training pipeline")
    return PromotionDecision(ok, reasons)
