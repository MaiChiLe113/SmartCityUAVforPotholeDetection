"""Evaluation on the frozen test set and field set (§3.1, §3.3). Owner: Hai (SCRUM-43)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from config import TRAIN, TrainConfig
from ml.grouping import view_of


@dataclass
class EvalReport:
    weights: Path
    split: str  # "test" | "field" | "val"
    map50_95: float
    map50: float
    per_class_ap: dict[str, float]
    per_view_map: dict[str, float]  # {"ground": ..., "aerial": ...}
    precision: dict[str, float]
    recall: dict[str, float]


def evaluate(weights: Path, data_yaml: Path, split: str = "test") -> EvalReport:
    """Run `YOLO(weights).val(data=data_yaml, split=split)` and collect overall/per-class metrics."""
    raise NotImplementedError("TODO(SCRUM-43, Hai): evaluate on a split")


def evaluate_by_view(weights: Path, data_yaml: Path, split: str = "test") -> dict[str, float]:
    """mAP50-95 for the ground and aerial subsets separately."""
    raise NotImplementedError("TODO(SCRUM-43, Hai): ground vs aerial evaluation")


def error_analysis(weights: Path, data_yaml: Path, out_dir: Path) -> Path:
    """Save confusion matrix and worst false positives/negatives for the final report."""
    raise NotImplementedError("TODO(SCRUM-43, Hai): error analysis")


def partition_val_list(image_paths: list[str]) -> dict[str, list[str]]:
    """Split validation image paths into overlapping ground, aerial and per-source subsets."""
    subsets: dict[str, list[str]] = {"all": list(image_paths)}
    for path in image_paths:
        stem = Path(path).stem
        for name in (view_of(stem), f"source_{stem.split('_', 1)[0]}"):
            subsets.setdefault(name, []).append(path)
    return subsets


def at_confidence(px, curve_row, conf: float) -> float:
    """Value of a per-class Ultralytics curve (1000 confidence points) at the given confidence."""
    i = min(range(len(px)), key=lambda k: abs(float(px[k]) - conf))
    return float(curve_row[i])


def false_alarms(
    weights: Path,
    image_paths: list[Path],
    tags: dict[str, str],
    conf_thresholds: dict[str, float],
    class_names: tuple[str, ...],
    cfg: TrainConfig = TRAIN,
) -> dict[str, dict]:
    """Share of negative images (no labelled damage) with at least one detection at the tuned confidences."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    floor = min(conf_thresholds.values())
    counts: dict[str, list[int]] = {}
    for path in image_paths:
        result = next(
            iter(model.predict(str(path), conf=floor, imgsz=cfg.imgsz, device=cfg.device, half=False, verbose=False))
        )
        hit = any(
            float(c) >= conf_thresholds[class_names[int(k)]]
            for k, c in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True)
        )
        key = f"{tags.get(path.stem, 'negative')}_{view_of(path.stem)}"
        counts.setdefault(key, [0, 0])
        counts[key][0] += 1
        counts[key][1] += hit
    return {k: {"images": n, "with_detection": h, "false_alarm_rate": h / n} for k, (n, h) in sorted(counts.items())}


def evaluate_val_subsets(
    weights: Path,
    val_list: Path,
    out_dir: Path,
    class_names: tuple[str, ...],
    cfg: TrainConfig = TRAIN,
    negatives: set[str] | None = None,
    conf_thresholds: dict[str, float] | None = None,
    split_name: str = "val",
    partition: bool = True,
) -> dict[str, dict]:
    """Score one model in FP32 on each subset of an image list and save the metrics as JSON.

    partition=False scores the whole list only (for external sets whose names have no source prefix).

    With negatives, adds a "damaged_only" subset without them. With conf_thresholds, adds per-class
    precision, recall and F1 at those confidences (read from Ultralytics' confidence curves).
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names_yaml = "".join(f"  {i}: {name}\n" for i, name in enumerate(class_names))
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}
    if val_list.read_text().startswith("./"):
        from ml.dataset_v1 import list_paths

        all_paths = [str(p) for p in list_paths(val_list)]
    else:
        all_paths = val_list.read_text().splitlines()
    subsets = partition_val_list(all_paths) if partition else {"all": all_paths}
    if negatives:
        subsets["damaged_only"] = [p for p in all_paths if Path(p).stem not in negatives]
    for name, paths in subsets.items():
        list_path = out_dir / f"val_{name}.txt"
        list_path.write_text("\n".join(paths) + "\n")
        yaml_path = out_dir / f"val_{name}.yaml"
        yaml_path.write_text(
            f"path: {out_dir.resolve().as_posix()}\ntrain: {list_path.name}\nval: {list_path.name}\n"
            f"nc: {len(class_names)}\nnames:\n{names_yaml}"
        )
        metrics = model.val(
            data=str(yaml_path),
            split="val",
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=cfg.device,
            workers=cfg.workers,
            conf=0.001,
            half=False,
            plots=False,
            project=str(out_dir),
            name=name,
            exist_ok=True,
        )
        if conf_thresholds:
            box = metrics.box
            tuned = {}
            for row, c in enumerate(metrics.ap_class_index):
                cls = class_names[int(c)]
                t = conf_thresholds[cls]
                tuned[cls] = {
                    "conf": t,
                    "precision": at_confidence(box.px, box.p_curve[row], t),
                    "recall": at_confidence(box.px, box.r_curve[row], t),
                    "f1": at_confidence(box.px, box.f1_curve[row], t),
                }
        report[name] = {
            "split": split_name,
            "precision_fp32": "half=False",
            "images": len(paths),
            "negatives": sum(Path(p).stem in (negatives or set()) for p in paths),
            **({"at_tuned_conf": tuned} if conf_thresholds else {}),
            "mAP50-95": float(metrics.box.map),
            "mAP50": float(metrics.box.map50),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "AP50-95": {
                class_names[int(c)]: float(ap) for c, ap in zip(metrics.ap_class_index, metrics.box.ap, strict=True)
            },
            "AP50": {
                class_names[int(c)]: float(ap) for c, ap in zip(metrics.ap_class_index, metrics.box.ap50, strict=True)
            },
        }
    (out_dir / f"{split_name}_subset_metrics.json").write_text(json.dumps(report, indent=2))
    return report
