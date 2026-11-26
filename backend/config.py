"""Every changeable value in the backend lives here.

Build functions (ml/build.py) read these settings, and run scripts (run/*.py) only call
build functions. To change an experiment, a threshold or a path, edit this file only.

Machine-specific values and secrets come from backend/.env (template: backend/.env.example).
Section references (§3.x) point to the Task 1 report.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value else default


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"  # Hugging Face download
    processed_dir: Path = ROOT / "data" / "processed"  # deduped, YOLO format
    splits_dir: Path = ROOT / "data" / "splits"  # train/val/test lists + frozen test hash
    field_dir: Path = ROOT / "data" / "field"  # checked resident/UAV images (field set)
    runs_dir: Path = ROOT / "runs"  # Ultralytics training output
    weights_dir: Path = ROOT / "weights"  # exported/served weights
    reports_dir: Path = ROOT / "runs" / "reports"  # comparison tables, drift HTML
    media_dir: Path = Path(_env("MEDIA_DIR", str(ROOT / "data" / "media")))


# ---------------------------------------------------------------------------
# Data (§3.1a, §3.1b) - SCRUM-18
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    hf_dataset_id: str = "TODO/unified-road-defect-dataset"  # TODO(SCRUM-18, Chi): real HF repo id
    hf_revision: str = "main"
    hf_repo_type: str = "dataset"
    hf_token: str | None = os.getenv("HF_TOKEN") or None
    classes: tuple[str, ...] = ("D00", "D10", "D20", "D40")
    class_names: tuple[str, ...] = (
        "longitudinal crack",
        "transverse crack",
        "alligator crack",
        "pothole",
    )
    split: tuple[float, float, float] = (0.70, 0.15, 0.15)  # train, val, test
    stratify_by: str = "view"  # "ground" vs "aerial" ratio kept in every subset
    views: tuple[str, ...] = ("ground", "aerial")
    dedupe_hash_threshold: int = 4  # max perceptual-hash distance counted as duplicate
    seed: int = 42
    oversample_classes: tuple[str, ...] = ("D20", "D40")  # only if their AP falls behind
    oversample_factor: int = 1  # 1 = off


# ---------------------------------------------------------------------------
# Grouped train/val split for prototype runs only - SCRUM-38 (the audit replaces it)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PrototypeSplitConfig:
    source_dir: Path = ROOT / "data" / "prototype"  # extracted dataset with the published split
    output_dir: Path = ROOT / "data" / "prototype" / "grouped"
    val_share: dict[str, float] = field(default_factory=lambda: {"ground": 0.15, "aerial": 0.20})
    seed: int = 42


# ---------------------------------------------------------------------------
# Dataset v1: grouped 70/15/15 train/val/test with a frozen test set (ADR-007)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetV1Config:
    unified_dir: Path = ROOT / "data" / "prototype"  # extracted unified dataset (rdd_, rdv_ and uav_ files)
    uavpdd_zip: Path = ROOT / "data" / "raw" / "uavpdd2023" / "UAV-PDD2023.zip"
    rdd_drone_zip: Path = ROOT / "data" / "raw" / "rdd2022_china_drone" / "RDD2022" / "China_Drone.zip"
    highrpd_zip: Path = ROOT / "data" / "raw" / "highrpd" / "HighRPD.zip"
    processed_dir: Path = ROOT / "data" / "processed"
    output_dir: Path = ROOT / "data" / "interim_v1"
    shares: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "val": {"ground": 0.15, "aerial": 0.15},
            "test": {"ground": 0.15, "aerial": 0.15},
        }
    )
    seed: int = 42
    # Near-duplicates: dHash finds candidates, pixels confirm them. Random aerial pairs collide on dHash
    # (11 of 20,000 within 4 bits, calibration.json), so a dHash match alone never links two images.
    dhash_max_distance: int = 6  # bits of 64
    pixel_mad_max: float = 5.0  # mean abs grey difference at 64x64, best of 8 poses: keeps a pair on one side
    pixel_mad_copy: float = 2.0  # at or below this the pair is the same photo, so val/test score it once
    uavpdd_classes: dict[str, int] = field(  # Oblique crack and Repair have no 4-class match and are dropped
        default_factory=lambda: {"Longitudinal crack": 0, "Transverse crack": 1, "Alligator crack": 2, "Pothole": 3}
    )
    # UAV-PDD ships each frame as lr_/tb_/r180_ flipped copies of 4 tiles: one frame id links all 12 files.
    uavpdd_frame_pattern: str = r"^uavpdd_(?:lr|tb|r180)_(?P<frame>\d+)_(?:top|bottom)_(?:left|right)$"
    rdd_classes: dict[str, int] = field(default_factory=lambda: {"D00": 0, "D10": 1, "D20": 2, "D40": 3})
    highrpd_pothole_id: int = 2  # HighRPD: 0 line crack, 1 block crack, 2 pothole -> our D40 only
    handoff_dir: Path = ROOT / "data" / "handoff" / "yolo26n-v1"
    val_sample: dict[str, int] = field(default_factory=lambda: {"aerial": 70, "ground": 30})


# ---------------------------------------------------------------------------
# Augmentation, training images only (§3.1 Table 3.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AugmentConfig:
    hsv_v: float = 0.4  # brightness
    hsv_s: float = 0.5  # saturation/contrast
    blur_p: float = 0.01
    noise_p: float = 0.01
    fliplr: float = 0.5
    degrees: float = 5.0  # small rotations
    scale: float = 0.5
    crop_fraction: float = 1.0
    mosaic: float = 1.0


# ---------------------------------------------------------------------------
# Training (§3.1d) - SCRUM-19, SCRUM-25
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    candidates: tuple[str, ...] = ("yolov8n.pt", "yolo11n.pt", "yolo26n.pt")
    baseline: str = "yolov8n.pt"
    imgsz: int = 640
    epochs: int = 100
    patience: int = 20  # early stopping
    batch: int = 32
    device: str = _env("DEVICE", "0")
    workers: int = 4  # 8 left only 1.3 GB of 31.8 GB RAM free on Windows, even in a smoke run
    optimizer: str = "MuSGD"  # pinned: "auto" ignores lr0 and picks AdamW for short smoke runs (ADR-003)
    lr0: float = 0.01
    momentum: float = 0.9
    seed: int = 42
    smoke_fraction: float = 0.05  # share of data for quick "smoke" runs on a local GPU
    smoke_epochs: int = 3


# ---------------------------------------------------------------------------
# Inference on the server (§3.1d Table 3.4) - SCRUM-26, SCRUM-29
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InferConfig:
    # Per-class F1-maximising confidences for yolo26n-v1, tuned on interim_v1 validation
    # (runs/reports/v1/thresholds.json, 2026-11-23). Never tune these on test.
    conf_thresholds: dict[str, float] = field(
        default_factory=lambda: {"D00": 0.232, "D10": 0.233, "D20": 0.287, "D40": 0.267}
    )
    # YOLO26 is end2end (NMS-free), so iou is unused; it applies only to end2end=False exports.
    iou: float = 0.7
    max_det: int = 100
    imgsz: int = 640
    server_precision: str = "fp32"
    server_weights: str = os.getenv("SERVER_WEIGHTS", "")  # empty = MLflow champion


# ---------------------------------------------------------------------------
# Edge export (§3.4) - SCRUM-35
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExportConfig:
    formats: dict[str, str] = field(default_factory=lambda: {"jetson": "engine", "pi": "ncnn"})
    int8: bool = True
    calib_split: str = "val"  # includes aerial views
    batch: int = 1
    imgsz: int = 640
    max_map_drop: float = 0.01  # vs FP32 on the frozen test set
    stability_minutes: int = 30
    warmup_runs: int = 10
    benchmark_runs: int = 100


# ---------------------------------------------------------------------------
# Champion vs challenger promotion (§3.2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromotionConfig:
    max_class_ap_drop: float = 0.01
    require_aerial_gain: bool = True
    require_field_no_worse: bool = True
    require_edge_checks: bool = True


# ---------------------------------------------------------------------------
# MLOps tooling (§3.5) - SCRUM-21, SCRUM-28
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MlopsConfig:
    mlflow_uri: str = _env("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = "pothole-yolo"
    registry_model_name: str = "pothole-detector"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"


# ---------------------------------------------------------------------------
# Monitoring, drift and retraining triggers (§3.2 Table 3.5, §3.3) - SCRUM-36, SCRUM-46
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MonitorConfig:
    drift_features: tuple[str, ...] = (
        "brightness",
        "contrast",
        "blur",
        "resolution",
        "view",
        "predicted_class",
    )
    ks_pvalue: float = 0.05
    drift_feature_share: float = 1 / 3  # alert if more than this share of features drift
    corrections_window: int = 200
    corrections_rate: float = 0.15
    uav_window: int = 50
    uav_disagree_rate: float = 0.20
    min_new_images: int = 500
    scheduled_review_days: int = 30


# ---------------------------------------------------------------------------
# UAV simulation and flight planning (§2.1) - SCRUM-22, SCRUM-27, SCRUM-33
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class UavConfig:
    home_lat: float = 10.7769  # TODO(SCRUM-22, Hoan): set the base location
    home_lon: float = 106.7009
    default_altitude_m: float = 30.0
    survey_radius_m: float = 25.0
    speed_mps: float = 5.0
    camera_fov_deg: float = 84.0
    camera_resolution: tuple[int, int] = (1920, 1080)
    frame_interval_s: float = 1.0
    telemetry_interval_s: float = 1.0
    drone_ids: tuple[str, ...] = ("sim-uav-1",)
    merge_radius_m: float = 2.0  # detections closer than this are one defect


# ---------------------------------------------------------------------------
# API server - SCRUM-23
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ApiConfig:
    database_url: str = _env("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'app.db'}")
    cors_origins: tuple[str, ...] = tuple(_env("CORS_ORIGINS", "http://localhost:3000").split(","))
    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_mb: int = 10
    allowed_types: tuple[str, ...] = ("image/jpeg", "image/png")


PATHS = PathsConfig()
DATA = DataConfig()
PROTOTYPE_SPLIT = PrototypeSplitConfig()
DATASET_V1 = DatasetV1Config()
AUGMENT = AugmentConfig()
TRAIN = TrainConfig()
INFER = InferConfig()
EXPORT = ExportConfig()
PROMOTION = PromotionConfig()
MLOPS = MlopsConfig()
MONITOR = MonitorConfig()
UAV = UavConfig()
API = ApiConfig()
