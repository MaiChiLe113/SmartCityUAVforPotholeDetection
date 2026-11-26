"""Ingest new labelled batches into the retraining split without touching the frozen test set.

A batch is a folder data/new/batch_NNN with images/ and labels/ (YOLO format, one .txt per image).
Every run rebuilds data/retrain from the dataset v1 lists plus every batch on disk, so the result does not
depend on the order batches arrived in. artifacts/metadata/batches.json lists the batches the current
champion was trained on; any other batch counts as new data and triggers retraining.

Leakage rules (same thresholds as dataset v1):
- an exact copy (sha256) of a v1 validation or test image is rejected
- a near-duplicate (dHash candidate confirmed by pixel difference, any flip or rotation) of a validation
  or test image is rejected
- new images that are near-duplicates of each other share a group, and a group goes wholly to train or val
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from config import DATA, DATASET_V1, RETRAIN, DatasetV1Config, RetrainConfig
from ml.grouping import IMAGE_SUFFIXES, link_key, view_of


@dataclass
class NewImage:
    stem: str
    batch: str
    image: Path
    label: Path


@dataclass
class IngestReport:
    batches: dict[str, dict] = field(default_factory=dict)  # per batch: images, train, val, rejected
    rejected: list[dict] = field(default_factory=list)  # {"stem", "batch", "reason"}
    train: list[NewImage] = field(default_factory=list)
    val: list[NewImage] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "batches": self.batches,
            "accepted_train": len(self.train),
            "accepted_val": len(self.val),
            "rejected": len(self.rejected),
            "rejected_by_reason": dict(Counter(r["reason"].split(":")[0] for r in self.rejected)),
            "rejected_images": self.rejected,
        }


# --- batch discovery ---------------------------------------------------------------------------
def list_batches(new_dir: Path) -> list[Path]:
    """Batch folders (batch_*) that contain an images/ folder, in name order."""
    if not new_dir.exists():
        return []
    return sorted(
        p for p in new_dir.iterdir() if p.is_dir() and p.name.startswith("batch_") and (p / "images").is_dir()
    )


def trained_batches(metadata_dir: Path) -> dict[str, dict]:
    path = metadata_dir / "batches.json"
    return json.loads(path.read_text()) if path.exists() else {}


def pending_batches(cfg: RetrainConfig = RETRAIN) -> list[Path]:
    """Batches on disk that the current champion was not trained on."""
    done = trained_batches(cfg.metadata_dir)
    return [b for b in list_batches(cfg.new_data_dir) if b.name not in done]


def batch_images(batch: Path) -> list[NewImage]:
    return [
        NewImage(p.stem, batch.name, p, batch / "labels" / f"{p.stem}.txt")
        for p in sorted((batch / "images").iterdir())
        if p.suffix.lower() in IMAGE_SUFFIXES
    ]


# --- label validation --------------------------------------------------------------------------
def label_errors(text: str, n_classes: int) -> list[str]:
    """Problems in one YOLO label file. An empty file is valid: the image has no damage."""
    errors = []
    for n, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            errors.append(f"line {n}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"line {n}: not numeric")
            continue
        if not 0 <= cls < n_classes:
            errors.append(f"line {n}: class {cls} outside 0..{n_classes - 1}")
        if any(not 0.0 <= v <= 1.0 for v in coords):
            errors.append(f"line {n}: coordinates outside [0, 1]")
        elif coords[2] <= 0 or coords[3] <= 0:
            errors.append(f"line {n}: empty box")
    return errors


def safe_view(stem: str) -> str:
    """View from the file name; names outside the known sources count as ground photos."""
    try:
        return view_of(stem)
    except ValueError:
        return "ground"


# --- split assignment --------------------------------------------------------------------------
def goes_to_val(group: str, val_share: float, seed: int) -> bool:
    """Stable per-group coin: the same group always lands on the same side, whatever else is in the batch."""
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < val_share


class _UnionFind:
    def __init__(self, items) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def group_new_images(stems: list[str], near_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """Group id per new image: images that share a source link or are near-duplicates are one group."""
    uf = _UnionFind(stems)
    by_key: dict[str, str] = {}
    for s in stems:
        if (key := link_key(s)) is not None:
            if key in by_key:
                uf.union(s, by_key[key])
            else:
                by_key[key] = s
    for a, b in near_pairs:
        uf.union(a, b)
    return {s: uf.find(s) for s in stems}


# --- hashing -----------------------------------------------------------------------------------
def reference_hashes(v1: DatasetV1Config, sides: tuple[str, ...] = ("val", "test")) -> dict[str, dict]:
    """sha256 and 8-pose dHash of every v1 image on the given sides, from the dataset v1 cache.

    The cache is keyed by stem; images missing from it are hashed now.
    """
    from ml import dataset_v1, split_v1

    cache_path = v1.output_dir / "cache" / "hashes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    out = {}
    for side in sides:
        for path in dataset_v1.list_paths(v1.output_dir / f"{side}.txt"):
            hit = cache.get(path.stem)
            if hit is None:
                hit = {"sha256": split_v1._sha256(path), "dhash": split_v1.dhash_poses(path)}
            out[path.stem] = {"side": side, "path": path, "sha256": hit["sha256"], "dhash": hit["dhash"]}
    return out


def cross_candidates(
    new: dict[str, list[int]], ref: dict[str, list[int]], max_distance: int
) -> list[tuple[str, str, int]]:
    """(new stem, ref stem, bits) where a pose of the new image is within max_distance of the ref pose 0."""
    import numpy as np

    if not new or not ref:
        return []
    ref_names = sorted(ref)
    ref_base = np.array([ref[n][0] for n in ref_names], dtype=np.uint64)
    out = []
    for stem, poses in sorted(new.items()):
        q = np.array(poses, dtype=np.uint64)
        dist = np.bitwise_count(q[:, None] ^ ref_base[None, :]).min(axis=0)
        for j in np.nonzero(dist <= max_distance)[0]:
            out.append((stem, ref_names[int(j)], int(dist[j])))
    return out


# --- ingest ------------------------------------------------------------------------------------
def ingest(cfg: RetrainConfig = RETRAIN, v1: DatasetV1Config = DATASET_V1) -> IngestReport:
    """Validate every batch, drop leaks into val/test, and split accepted images into train and val."""
    from ml import split_v1

    report = IngestReport()
    candidates: list[NewImage] = []
    seen: dict[str, str] = {}
    for batch in list_batches(cfg.new_data_dir):
        images = batch_images(batch)
        report.batches[batch.name] = {"images": len(images), "train": 0, "val": 0, "rejected": 0}
        for item in images:
            reason = None
            if item.stem in seen:
                reason = f"duplicate_name:{seen[item.stem]}"
            elif not item.label.exists():
                reason = "missing_label"
            elif errors := label_errors(item.label.read_text(), len(DATA.classes)):
                reason = "bad_label:" + "; ".join(errors[:3])
            if reason:
                report.rejected.append({"stem": item.stem, "batch": item.batch, "reason": reason})
                report.batches[batch.name]["rejected"] += 1
            else:
                seen[item.stem] = item.batch
                candidates.append(item)

    hashes: dict[str, dict] = {}
    keep: list[NewImage] = []
    for item in candidates:
        try:
            hashes[item.stem] = {"sha256": split_v1._sha256(item.image), "dhash": split_v1.dhash_poses(item.image)}
            keep.append(item)
        except ValueError:
            report.rejected.append({"stem": item.stem, "batch": item.batch, "reason": "unreadable_image"})
            report.batches[item.batch]["rejected"] += 1

    ref = reference_hashes(v1)
    ref_by_sha = {v["sha256"]: s for s, v in ref.items()}
    leaks: dict[str, str] = {}
    for item in keep:
        if (match := ref_by_sha.get(hashes[item.stem]["sha256"])) is not None:
            leaks[item.stem] = f"copy_of_{ref[match]['side']}:{match}"
    pairs = cross_candidates(
        {s: hashes[s]["dhash"] for s in (i.stem for i in keep) if s not in leaks},
        {s: v["dhash"] for s, v in ref.items()},
        v1.dhash_max_distance,
    )
    thumbs: dict[Path, object] = {}

    def thumb(path: Path):
        if path not in thumbs:
            thumbs[path] = split_v1.thumbnail(path)
        return thumbs[path]

    by_stem = {i.stem: i for i in keep}
    for stem, ref_stem, _bits in pairs:
        if stem in leaks:
            continue
        mad = split_v1.best_pose_mad(thumb(by_stem[stem].image), thumb(ref[ref_stem]["path"]))
        if mad <= v1.pixel_mad_max:
            leaks[stem] = f"near_duplicate_of_{ref[ref_stem]['side']}:{ref_stem}"
    for stem, reason in sorted(leaks.items()):
        report.rejected.append({"stem": stem, "batch": by_stem[stem].batch, "reason": reason})
        report.batches[by_stem[stem].batch]["rejected"] += 1

    accepted = [i for i in keep if i.stem not in leaks]
    near = split_v1.near_duplicate_pairs({i.stem: hashes[i.stem]["dhash"] for i in accepted}, v1.dhash_max_distance)
    confirmed = [
        (a, b)
        for a, b, _ in near
        if split_v1.best_pose_mad(thumb(by_stem[a].image), thumb(by_stem[b].image)) <= v1.pixel_mad_max
    ]
    groups = group_new_images([i.stem for i in accepted], confirmed)
    for item in accepted:
        side = "val" if goes_to_val(groups[item.stem], cfg.val_share, cfg.seed) else "train"
        getattr(report, side).append(item)
        report.batches[item.batch][side] += 1
    return report


def _relative(path: Path, base: Path) -> str:
    return "./" + Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()


def write_retrain_split(report: IngestReport, cfg: RetrainConfig = RETRAIN, v1: DatasetV1Config = DATASET_V1) -> Path:
    """Write data/retrain/{train,val}.txt and data.yaml. Test stays the frozen v1 list, untouched."""
    from ml import dataset_v1

    out = cfg.retrain_dir
    out.mkdir(parents=True, exist_ok=True)
    for side, new in (("train", report.train), ("val", report.val)):
        old = dataset_v1.list_paths(v1.output_dir / f"{side}.txt")
        lines = [_relative(p, out) for p in old] + [_relative(i.image, out) for i in new]
        (out / f"{side}.txt").write_text("\n".join(lines) + "\n")
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(DATA.classes))
    test_rel = Path(os.path.relpath((v1.output_dir / "test.txt").resolve(), out.resolve())).as_posix()
    yaml = out / "data.yaml"
    yaml.write_text(
        "# Retraining split: dataset v1 plus accepted new batches. Test is the frozen v1 list (do not edit).\n"
        f"train: train.txt\nval: val.txt\ntest: {test_rel}\nnc: {len(DATA.classes)}\nnames:\n{names}"
    )
    return yaml


def write_ingest_outputs(report: IngestReport, cfg: RetrainConfig = RETRAIN) -> dict:
    """ingest_report.json for this run and data_version.txt naming the batches in the split."""
    summary = report.summary() | {"created_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    reports = cfg.monitoring_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "ingest_report.json").write_text(json.dumps(summary, indent=2))
    cfg.metadata_dir.mkdir(parents=True, exist_ok=True)
    version = "v1+" + "+".join(sorted(report.batches)) if report.batches else "v1"
    (cfg.metadata_dir / "data_version.txt").write_text(version + "\n")
    return summary


def new_train_images(retrain_dir: Path) -> list[Path]:
    """Images in the retrain train list that came from data/new (for the CPU demo)."""
    from ml import dataset_v1

    return [p for p in dataset_v1.list_paths(retrain_dir / "train.txt") if "new" in p.parts]


def view_counts(items: list[NewImage]) -> dict[str, int]:
    return dict(Counter(safe_view(i.stem) for i in items))
