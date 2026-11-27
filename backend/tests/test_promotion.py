from dataclasses import replace

from config import PROMOTION
from ml.promotion import check_promotion


def report(overall, aerial, per_class):
    return {
        "all": {"mAP50-95": overall, "AP50-95": per_class},
        "aerial": {"mAP50-95": aerial, "AP50-95": per_class},
    }


CHAMPION = report(0.300, 0.200, {"D00": 0.30, "D10": 0.25, "D20": 0.35, "D40": 0.30})


def test_promotes_when_every_check_passes():
    challenger = report(0.310, 0.205, {"D00": 0.31, "D10": 0.25, "D20": 0.345, "D40": 0.33})
    decision = check_promotion(CHAMPION, challenger)
    assert decision.promote
    assert any(r.startswith("N/A field") for r in decision.reasons)


def test_rejects_no_overall_gain():
    challenger = report(0.300, 0.210, CHAMPION["all"]["AP50-95"])
    assert not check_promotion(CHAMPION, challenger).promote


def test_rejects_aerial_drop():
    challenger = report(0.320, 0.190, {"D00": 0.33, "D10": 0.27, "D20": 0.36, "D40": 0.32})
    decision = check_promotion(CHAMPION, challenger)
    assert not decision.promote
    assert any(r.startswith("FAIL aerial") for r in decision.reasons)


def test_rejects_class_drop_beyond_limit():
    challenger = report(0.320, 0.210, {"D00": 0.33, "D10": 0.23, "D20": 0.36, "D40": 0.34})
    decision = check_promotion(CHAMPION, challenger)
    assert not decision.promote
    assert any(r.startswith("FAIL D10") for r in decision.reasons)


def test_aerial_check_can_be_disabled():
    challenger = report(0.320, 0.190, {"D00": 0.33, "D10": 0.27, "D20": 0.36, "D40": 0.32})
    assert check_promotion(CHAMPION, challenger, replace(PROMOTION, require_aerial_gain=False)).promote
