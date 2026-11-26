from datetime import UTC, datetime, timedelta

from ml.retrain_trigger import active_triggers

NOW = datetime(2026, 11, 27, tzinfo=UTC)


def test_new_data_retrains():
    result = active_triggers(120, False, NOW - timedelta(days=2), NOW)
    assert result.should_retrain and result.triggers == ["new_data"]


def test_no_new_data_only_monitors_even_when_review_is_due():
    result = active_triggers(0, False, NOW - timedelta(days=45), NOW)
    assert not result.should_retrain and result.triggers == ["review_due"]


def test_drift_and_volume_are_reported():
    result = active_triggers(800, True, None, NOW)
    assert result.should_retrain
    assert result.triggers == ["new_data", "drift_alert", "enough_new_images", "review_due"]


def test_force_retrains_without_new_data():
    result = active_triggers(0, False, NOW, NOW, force=True)
    assert result.should_retrain and "manual_force" in result.triggers
