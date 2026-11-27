import random

import pytest

from ml import drift

pytest.importorskip("scipy")


def test_compare_numeric_flags_shift_and_not_same_distribution():
    rng = random.Random(0)
    ref = [rng.gauss(100, 10) for _ in range(500)]
    same = [rng.gauss(100, 10) for _ in range(300)]
    shifted = [rng.gauss(140, 10) for _ in range(300)]
    assert not drift.compare_numeric(ref, same, 0.01)["drifted"]
    assert drift.compare_numeric(ref, shifted, 0.01)["drifted"]


def test_compare_numeric_needs_values():
    assert drift.compare_numeric([1.0, 2.0], [], 0.05)["p_value"] is None


def test_compare_counts_flags_class_mix_change():
    ref = {"D00": 400, "D10": 300, "D20": 200, "D40": 100}
    assert not drift.compare_counts(ref, {"D00": 40, "D10": 30, "D20": 20, "D40": 10}, 0.05)["drifted"]
    assert drift.compare_counts(ref, {"D00": 5, "D10": 5, "D20": 5, "D40": 200}, 0.05)["drifted"]


def test_compare_counts_skips_single_category():
    result = drift.compare_counts({"ground": 10}, {"ground": 5}, 0.05)
    assert result["p_value"] is None and not result["drifted"]


def test_decide_uses_share_of_scored_tests():
    tests = {
        "a": {"p_value": 0.001, "drifted": True},
        "b": {"p_value": 0.5, "drifted": False},
        "c": {"p_value": 0.4, "drifted": False},
        "d": {"p_value": None, "drifted": False},
    }
    drifted, share, alert = drift.decide(tests, 1 / 3)
    assert drifted == ["a"] and share == pytest.approx(1 / 3) and not alert
    tests["b"] = {"p_value": 0.01, "drifted": True}
    assert drift.decide(tests, 1 / 3)[2]


def test_render_html_marks_drifted_rows():
    report = {
        "tests": {"brightness": {"test": "ks", "statistic": 0.5, "p_value": 0.001, "drifted": True}},
        "data_quality": {"images": 3},
        "alert": True,
        "images": 3,
        "reference_images": 10,
        "drifted_features": ["brightness"],
        "share_drifted": 1.0,
        "alert_share": 1 / 3,
        "alpha": 0.05,
        "created_utc": "2026-11-27T00:00:00+00:00",
    }
    page = drift.render_html(report)
    assert "ALERT" in page and "class='drift'" in page
