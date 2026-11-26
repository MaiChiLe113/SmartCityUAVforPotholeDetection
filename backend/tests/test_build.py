import pytest

from ml import build


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build.build_model("toaster.pt")


def test_compile_train_args_has_core_keys(tmp_path):
    args = build.compile_train_args("yolov8n.pt", tmp_path / "data.yaml")
    assert {"data", "imgsz", "epochs", "seed"} <= set(args)


def test_compile_train_args_pins_the_optimizer(tmp_path):
    args = build.compile_train_args("yolo26n.pt", tmp_path / "data.yaml", smoke=True)
    assert (args["optimizer"], args["lr0"], args["momentum"]) == ("MuSGD", 0.01, 0.9)
