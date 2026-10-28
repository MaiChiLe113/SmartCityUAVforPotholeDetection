"""Edge export: FP32 -> ONNX -> TensorRT INT8 (Jetson) / NCNN INT8 (Pi) (§3.4). Owner: Hoan (SCRUM-35)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import EXPORT, ExportConfig


@dataclass
class ExportPackage:
    target: str  # "jetson" | "pi"
    path: Path
    model_version: str
    settings: dict  # export args + software versions (JetPack, TensorRT, NCNN)


class Exporter:
    def __init__(self, target: str, cfg: ExportConfig = EXPORT) -> None:
        if target not in cfg.formats:
            raise ValueError(f"unknown export target {target!r}, expected one of {list(cfg.formats)}")
        self.target = target
        self.cfg = cfg

    def __call__(self, weights: Path, data_yaml: Path, model_version: str) -> ExportPackage:
        """Export with `YOLO(weights).export(format=..., int8=..., data=data_yaml, imgsz=..., batch=1)`.

        The TensorRT engine must be built on the Jetson itself.
        """
        raise NotImplementedError("TODO(SCRUM-35, Hoan): export INT8 package")

    def check_accuracy(self, package: ExportPackage, fp32_map: float, data_yaml: Path) -> bool:
        """Pass if INT8 mAP50-95 on the frozen test set is within cfg.max_map_drop of FP32."""
        raise NotImplementedError("TODO(SCRUM-35, Hoan): accuracy-after-export check")
