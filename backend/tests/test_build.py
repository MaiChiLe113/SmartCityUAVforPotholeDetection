import pytest

from ml import build


def test_exporter_rejects_unknown_target():
    with pytest.raises(ValueError):
        build.build_exporter("toaster")


def test_builders_return_objects():
    assert build.build_flight_planner() is not None
    assert build.build_drift_monitor() is not None
    assert build.build_registry() is not None


@pytest.mark.xfail(raises=NotImplementedError, reason="SCRUM-19 (Hai)")
def test_compile_train_args_has_core_keys(tmp_path):
    args = build.compile_train_args("yolov8n.pt", tmp_path / "data.yaml")
    assert {"data", "imgsz", "epochs", "seed"} <= set(args)
