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
    EXPORT,
    INFER,
    MLOPS,
    MONITOR,
    PATHS,
    TRAIN,
    UAV,
    AugmentConfig,
    DataConfig,
    ExportConfig,
    InferConfig,
    MlopsConfig,
    MonitorConfig,
    TrainConfig,
    UavConfig,
)
from ml import data

if TYPE_CHECKING:
    from app.services.detector import Detector
    from app.services.flight_planner import FlightPlanner
    from ml.drift import DriftMonitor
    from ml.export import Exporter
    from ml.registry import ModelRegistry


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
    raise NotImplementedError("TODO(SCRUM-19, Hai): return ultralytics.YOLO(name)")


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
    raise NotImplementedError("TODO(SCRUM-19, Hai): build train kwargs (data, imgsz, epochs, ...)")


# --- Serving (SCRUM-29, Hai) ------------------------------------------------
def build_detector(weights: Path | None = None, cfg: InferConfig = INFER) -> Detector:
    """Return the server Detector. weights=None uses cfg.server_weights, else the MLflow champion."""
    from app.services.detector import Detector

    return Detector(weights, cfg)


# --- Edge (SCRUM-35, Hoan) --------------------------------------------------
def build_exporter(target: str, cfg: ExportConfig = EXPORT) -> Exporter:
    """Return the export job for "jetson" (TensorRT INT8) or "pi" (NCNN INT8)."""
    from ml.export import Exporter

    return Exporter(target, cfg)


# --- UAV (SCRUM-27, Hoan) ---------------------------------------------------
def build_flight_planner(cfg: UavConfig = UAV) -> FlightPlanner:
    from app.services.flight_planner import FlightPlanner

    return FlightPlanner(cfg)


# --- MLOps (SCRUM-28, SCRUM-36, Chi) ----------------------------------------
def build_drift_monitor(cfg: MonitorConfig = MONITOR) -> DriftMonitor:
    from ml.drift import DriftMonitor

    return DriftMonitor(cfg)


def build_registry(cfg: MlopsConfig = MLOPS) -> ModelRegistry:
    from ml.registry import ModelRegistry

    return ModelRegistry(cfg)
