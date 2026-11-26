"""Per-class confidence thresholds chosen on validation data. Owner: Hai (SCRUM-26).

YOLO26 runs end-to-end (no NMS): Ultralytics only applies a confidence filter to its outputs
(`utils/nms.py`, end2end branch), so there is no IoU threshold to tune for the exported model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from config import TRAIN, TrainConfig


def best_conf_per_class(
    px: Sequence[float],
    f1_curve: Sequence[Sequence[float]],
    p_curve: Sequence[Sequence[float]],
    r_curve: Sequence[Sequence[float]],
    class_index: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Pick, per class, the confidence where F1 peaks on its F1-confidence curve (curves are per present class)."""
    out = {}
    for row, cls in enumerate(class_index):
        f1 = list(f1_curve[row])
        i = max(range(len(f1)), key=f1.__getitem__)
        out[class_names[int(cls)]] = {
            "conf": round(float(px[i]), 3),
            "f1": float(f1[i]),
            "precision": float(p_curve[row][i]),
            "recall": float(r_curve[row][i]),
        }
    return out


def tune_thresholds(
    weights: Path,
    data_yaml: Path,
    class_names: Sequence[str],
    out_json: Path | None = None,
    cfg: TrainConfig = TRAIN,
) -> dict[str, dict[str, float]]:
    """Validate at conf 0.001 in FP32 and return the F1-best confidence per class.

    Copy the confidences into config.INFER.conf_thresholds (the values file stays the single source).
    """
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="val",
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        conf=0.001,
        half=False,
        plots=False,
    )
    box = metrics.box
    result = best_conf_per_class(box.px, box.f1_curve, box.p_curve, box.r_curve, metrics.ap_class_index, class_names)
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
    return result
