from pathlib import Path

import pytest

from ml import build


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build.build_model("toaster.pt")


def test_init_guard_allows_coco_and_champion_only(tmp_path):
    from dataclasses import replace

    from config import RETRAIN

    retrain = replace(RETRAIN, champion=tmp_path / "champion.pt")
    assert build.assert_allowed_init(Path("yolo26n.pt"), retrain=retrain)
    assert build.assert_allowed_init(tmp_path / "champion.pt", retrain=retrain)
    for bad in (tmp_path / "runs" / "train" / "yolo26n" / "weights" / "best.pt", Path("TamAko783_best.pt")):
        with pytest.raises(ValueError):
            build.assert_allowed_init(bad, retrain=retrain)


def test_compile_train_args_epochs_override(tmp_path):
    assert build.compile_train_args("yolo26n.pt", tmp_path / "d.yaml", epochs=20)["epochs"] == 20


def test_compile_train_args_has_core_keys(tmp_path):
    args = build.compile_train_args("yolov8n.pt", tmp_path / "data.yaml")
    assert {"data", "imgsz", "epochs", "seed"} <= set(args)


def test_compile_train_args_pins_the_optimizer(tmp_path):
    args = build.compile_train_args("yolo26n.pt", tmp_path / "data.yaml", smoke=True)
    assert (args["optimizer"], args["lr0"], args["momentum"]) == ("MuSGD", 0.01, 0.9)
