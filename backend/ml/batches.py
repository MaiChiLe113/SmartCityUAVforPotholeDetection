"""Turn RDD2022 country images that the unified dataset dropped into new-data batches for retraining.

The unified dataset kept exactly the RDD2022 images that contain damage. For Czech that is 1,072 of 2,829
labelled images; the other 1,757 are labelled road photos with no damage (empty YOLO label files). They are
real images the model has never seen, and adding clean images of this kind is the planned next data step for
reducing false alarms (PROJECT_CONTEXT §6, at most about 10% of training). Damaged images are kept too when
a country has any that the unified dataset dropped.
Leakage against v1 validation and test is still checked at ingest time (ml.ingest).
"""

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path

from config import DATASET_V1, RETRAIN, DatasetV1Config, RetrainConfig
from ml import convert
from ml.dataset_v1 import _pairs, safe_extract
from ml.grouping import IMAGE_SUFFIXES


def unified_numbers(country: str, cfg: DatasetV1Config = DATASET_V1) -> set[str]:
    """Image numbers of this RDD2022 country already in the unified dataset (any split)."""
    pattern = re.compile(rf"^rdd_{re.escape(country)}_{re.escape(country)}_(\d+)$")
    numbers = set()
    for side in ("train", "val"):
        folder = cfg.unified_dir / "images" / side
        if folder.exists():
            numbers |= {m[1] for p in folder.iterdir() if (m := pattern.match(p.stem))}
    return numbers


def split_into_batches(stems: list[str], n_batches: int, seed: int) -> list[list[str]]:
    """Seeded, near-equal batches; every stem lands in exactly one batch."""
    order = sorted(stems)
    random.Random(seed).shuffle(order)
    return [sorted(order[i::n_batches]) for i in range(n_batches)]


def prepare_country_batches(
    archive: Path,
    country: str,
    n_batches: int,
    first_index: int = 1,
    cfg: RetrainConfig = RETRAIN,
    v1: DatasetV1Config = DATASET_V1,
) -> dict:
    """Extract the RDD2022 zip, convert dropped labelled images to YOLO and write batch folders.

    Batches are written as data/new/batch_<NNN>/{images,labels}. Existing batch folders are never
    overwritten.
    """
    root = safe_extract(archive, archive.parent / "extracted")
    pairs, _ = _pairs(root)
    number = re.compile(rf"{re.escape(country)}_(\d+)$")
    kept = unified_numbers(country, v1)
    dropped = [(i, x) for i, x in pairs if (m := number.search(i.stem)) and m[1] not in kept]
    staging = cfg.new_data_dir / f".staging_{country.lower()}"
    if staging.exists():
        shutil.rmtree(staging)
    records = convert.convert_voc_dir(
        dropped, staging, v1.rdd_classes, lambda p: f"rdd_{country}_{country}_{number.search(p.stem)[1]}"
    )
    usable = [r["stem"] for r in records if r["tag"] in (convert.DAMAGED, convert.CLEAN)]
    batches = split_into_batches(usable, n_batches, cfg.seed)
    written = {}
    for k, stems in enumerate(batches, start=first_index):
        out = cfg.new_data_dir / f"batch_{k:03d}"
        if out.exists():
            raise FileExistsError(f"{out} already exists; choose another --first-index")
        (out / "images").mkdir(parents=True)
        (out / "labels").mkdir(parents=True)
        for stem in stems:
            image = next(p for p in (staging / "images").glob(f"{stem}.*") if p.suffix.lower() in IMAGE_SUFFIXES)
            shutil.move(image, out / "images" / image.name)
            shutil.move(staging / "labels" / f"{stem}.txt", out / "labels" / f"{stem}.txt")
        written[out.name] = len(stems)
    shutil.rmtree(staging)
    return {
        "country": country,
        "labelled_in_zip": len(pairs),
        "already_in_unified": len(kept),
        "dropped_converted": len(records),
        "dropped_damaged": sum(r["tag"] == convert.DAMAGED for r in records),
        "dropped_clean": sum(r["tag"] == convert.CLEAN for r in records),
        "dropped_unmapped_only": len(records) - len(usable),
        "batches": written,
    }
