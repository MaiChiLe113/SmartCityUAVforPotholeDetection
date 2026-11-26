"""Assemble the self-contained handoff folder: weights, frozen test set, validation sample, external test, README."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

from config import DATA, DatasetV1Config
from ml import dataset_v1
from ml.grouping import view_of

YAML_NOTE = (
    "# Paths are relative to this file. "
    "Class ids: 0 D00 longitudinal, 1 D10 transverse, 2 D20 alligator, 3 D40 pothole.\n"
)


def _names() -> str:
    return "".join(f"  {i}: {n}\n" for i, n in enumerate(DATA.classes))


def _copy_set(images: list[Path], dest: Path) -> None:
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    for p in images:
        shutil.copyfile(p, dest / "images" / p.name)
        shutil.copyfile(dataset_v1.label_for(p), dest / "labels" / f"{p.stem}.txt")


def sample_val(val_images: list[Path], counts: dict[str, int], seed: int) -> list[Path]:
    """Seeded sample per view; aerial is spread over sources so UAV-PDD is always included."""
    rng = random.Random(seed)
    chosen: list[Path] = []
    ground = [p for p in val_images if view_of(p.stem) == "ground"]
    chosen += rng.sample(ground, min(counts["ground"], len(ground)))
    aerial = [p for p in val_images if view_of(p.stem) == "aerial"]
    by_source: dict[str, list[Path]] = {}
    for p in aerial:
        by_source.setdefault(p.stem.split("_")[0] + ("cd" if "China_Drone" in p.stem else ""), []).append(p)
    share = counts["aerial"] // max(1, len(by_source))
    for group in by_source.values():
        chosen += rng.sample(group, min(share, len(group)))
    rest = [p for p in aerial if p not in chosen]
    chosen += rng.sample(rest, max(0, counts["aerial"] + counts["ground"] - len(chosen)))
    return sorted(chosen)


def build_handoff(cfg: DatasetV1Config, weights: Path, run_dir: Path, report_dir: Path) -> Path:
    out = cfg.handoff_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)
    shutil.copyfile(weights, out / "weights" / "best.pt")

    digest = dataset_v1.verify_frozen_test(cfg)
    test_images = dataset_v1.list_paths(cfg.output_dir / "test.txt")
    _copy_set(test_images, out / "test")
    (out / "test" / "data_test.yaml").write_text(
        YAML_NOTE + f"# Frozen test set, lock sha256 {digest}\n"
        f"train: images\nval: images\ntest: images\nnc: {len(DATA.classes)}\nnames:\n{_names()}"
    )
    tags = dataset_v1.negative_tags(cfg)
    with (out / "test" / "image_info.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "view", "source", "negative"])
        for p in test_images:
            w.writerow([p.name, view_of(p.stem), p.stem.split("_")[0], tags.get(p.stem, "")])
    shutil.copyfile(cfg.output_dir / "test_lock.json", out / "test" / "test_lock.json")

    val_images = dataset_v1.list_paths(cfg.output_dir / "val.txt")
    sample = sample_val(val_images, cfg.val_sample, cfg.seed)
    _copy_set(sample, out / "val_sample")
    (out / "val_sample" / "data_val_sample.yaml").write_text(
        YAML_NOTE + f"train: images\nval: images\nnc: {len(DATA.classes)}\nnames:\n{_names()}"
    )

    hr_base = cfg.processed_dir / "highrpd"
    hr = [hr_base / line[2:] for line in (hr_base / "subset.txt").read_text().splitlines()]
    _copy_set(hr, out / "external_highrpd")
    (out / "external_highrpd" / "data.yaml").write_text(
        YAML_NOTE + "# HighRPD (CC BY 4.0): only potholes are labelled, so score D40 only.\n"
        f"train: images\nval: images\ntest: images\nnc: {len(DATA.classes)}\nnames:\n{_names()}"
    )

    for name in ("args.yaml", "results.csv"):
        shutil.copyfile(run_dir / name, out / name)
    for name in ("evaluation.json", "thresholds.json"):
        shutil.copyfile(report_dir / name, out / name)
    return out
