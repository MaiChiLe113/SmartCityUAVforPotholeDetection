"""Build/compile layer: turns the values in config.py into objects that are ready to run.

Run scripts (run/*.py) call these functions and hold no values of their own.
Every function takes its config section as a default argument so tests can pass a different one.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import (
    AUGMENT,
    DATA,
    MONITOR,
    PATHS,
    TRAIN,
    AugmentConfig,
    DataConfig,
    MonitorConfig,
    TrainConfig,
)
from ml import data
from ml.augment import to_ultralytics_hyp

if TYPE_CHECKING:
    from ml.drift import DriftMonitor


# --- Data (SCRUM-18, Chi) ---------------------------------------------------
def build_raw_dataset(cfg: DataConfig = DATA) -> Path:
    """Download the dataset from Hugging Face (or reuse the local copy). Returns its root."""
    return data.download_dataset(cfg, PATHS.raw_dir)


def build_splits(raw_path: Path, cfg: DataConfig = DATA) -> data.SplitManifest:
    """Dedupe, group, split 70/15/15 by view and freeze the test set."""
    raise NotImplementedError("TODO(SCRUM-18, Chi): chain data.deduplicate/group_related/make_splits")


def build_data_yaml(manifest: data.SplitManifest) -> Path:
    """Write the Ultralytics data.yaml for a split manifest."""
    return data.write_data_yaml(manifest, PATHS.processed_dir)


def build_data_yaml_from_disk() -> Path:
    """Load the frozen split written by run/prepare_data.py and return its data.yaml."""
    return build_data_yaml(data.load_manifest(PATHS.splits_dir))


# --- Model (SCRUM-19, SCRUM-25, Hai) ----------------------------------------
def build_model(name: str = TRAIN.baseline, cfg: TrainConfig = TRAIN) -> Any:
    """Return an `ultralytics.YOLO` model for one of cfg.candidates (pretrained weights)."""
    if name not in cfg.candidates:
        raise ValueError(f"unknown model {name!r}, expected one of {cfg.candidates}")

    from ultralytics import YOLO

    return YOLO(name)


def compile_train_args(
    model_name: str,
    data_yaml: Path,
    cfg: TrainConfig = TRAIN,
    aug: AugmentConfig = AUGMENT,
    smoke: bool = False,
) -> dict:
    """Merge TRAIN and AUGMENT into the kwargs for `model.train(**args)`.

    Same args for every candidate so the comparison is fair. `smoke=True` uses
    cfg.smoke_fraction / cfg.smoke_epochs for a quick local check.
    """
    args = {
        "data": str(data_yaml),
        "imgsz": cfg.imgsz,
        "epochs": cfg.smoke_epochs if smoke else cfg.epochs,
        "patience": cfg.patience,
        "batch": cfg.batch,
        "device": cfg.device,
        "workers": cfg.workers,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "momentum": cfg.momentum,
        "seed": cfg.seed,
        "project": str(PATHS.runs_dir / "train"),
        "name": f"{Path(model_name).stem}{'-smoke' if smoke else ''}",
        **to_ultralytics_hyp(aug),
    }
    if smoke:
        args["fraction"] = cfg.smoke_fraction
    return args


# --- Monitoring (SCRUM-36) -----------------------------------------------------
def build_drift_monitor(cfg: MonitorConfig = MONITOR) -> DriftMonitor:
    from ml.drift import DriftMonitor

    return DriftMonitor(cfg)
