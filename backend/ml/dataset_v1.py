"""Dataset v1: combine the unified dataset with the original UAV-PDD2023 tiles and RDD2022 China_Drone
negatives, link copies and frames, split 70/15/15 by group and freeze the test set (ADR-007).

Stages, run in order by run/build_dataset_v1.py:
  prepare   - extract the archives, convert VOC labels to YOLO, tag clean and hard-negative images
  calibrate - hash every image, report near-duplicate distances for known copies vs random pairs
  build     - link, split, write relative file lists, data.yaml, manifest, summary and the test lock
Heavy imports (numpy, cv2) stay inside functions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import DATA, DatasetV1Config
from ml import convert
from ml.grouping import IMAGE_SUFFIXES, link_key, read_class_counts, stratum_of, view_of
from ml.split_v1 import (
    assert_disjoint,
    best_pose_mad,
    build_groups,
    dhash_poses,
    near_duplicate_pairs,
    split_three_way,
    thumbnail,
    verify_test_lock,
    write_test_lock,
)

_UNIFIED_RDD_DRONE = re.compile(r"^rdd_China_Drone_China_Drone_(\d+)$")


# --- prepare ------------------------------------------------------------------------------
def safe_extract(archive: Path, out_dir: Path) -> Path:
    """Extract a zip once, refusing members that would land outside out_dir."""
    marker = out_dir / ".extracted"
    if marker.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (out_dir / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe member {member.filename!r} in {archive.name}")
        z.extractall(out_dir)
    marker.write_text(archive.name)
    return out_dir


def _pairs(root: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """Match every VOC xml under root to the image with the same stem. Returns pairs and images without xml."""
    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    xmls = {p.stem: p for p in root.rglob("*.xml")}
    pairs = [(images[s], xmls[s]) for s in sorted(xmls) if s in images]
    return pairs, [images[s] for s in sorted(images.keys() - xmls.keys())]


def prepare(cfg: DatasetV1Config) -> dict:
    report: dict = {}

    uav_root = safe_extract(cfg.uavpdd_zip, cfg.uavpdd_zip.parent / "extracted")
    pairs, unlabelled = _pairs(uav_root)
    records = convert.convert_voc_dir(
        pairs, cfg.processed_dir / "uavpdd", cfg.uavpdd_classes, lambda p: f"uavpdd_{p.stem}"
    )
    (cfg.processed_dir / "uavpdd" / "records.json").write_text(json.dumps(records))
    report["uavpdd"] = _record_summary(records) | {"images_without_xml": len(unlabelled)}

    drone_root = safe_extract(cfg.rdd_drone_zip, cfg.rdd_drone_zip.parent / "extracted")
    pairs, unlabelled = _pairs(drone_root)
    kept = {
        m[1]
        for side in ("train", "val")
        for p in (cfg.unified_dir / "images" / side).iterdir()
        if (m := _UNIFIED_RDD_DRONE.match(p.stem))
    }
    number = re.compile(r"China_Drone_(\d+)$")
    dropped = [(i, x) for i, x in pairs if (m := number.search(i.stem)) and m[1] not in kept]
    records = convert.convert_voc_dir(
        dropped,
        cfg.processed_dir / "rdd_china_drone_extra",
        cfg.rdd_classes,
        lambda p: f"rdd_China_Drone_China_Drone_{number.search(p.stem)[1]}",
    )
    (cfg.processed_dir / "rdd_china_drone_extra" / "records.json").write_text(json.dumps(records))
    report["rdd_china_drone_extra"] = _record_summary(records) | {
        "raw_images": len(pairs) + len(unlabelled),
        "already_in_unified": len(kept),
        "images_without_xml": len(unlabelled),
    }

    hr_root = safe_extract(cfg.highrpd_zip, cfg.highrpd_zip.parent / "extracted")
    report["highrpd"] = prepare_highrpd(hr_root, cfg.processed_dir / "highrpd", cfg.highrpd_pothole_id)

    out = cfg.processed_dir / "prepare_report.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def _record_summary(records: list[dict]) -> dict:
    raw, dropped = Counter(), Counter()
    for r in records:
        raw.update(r["raw_classes"])
        dropped.update(r["dropped"])
    (out := {})["images"] = len(records)
    out["tags"] = dict(Counter(r["tag"] for r in records))
    out["yolo_boxes"] = sum(r["boxes"] for r in records)
    out["raw_classes"] = dict(raw)
    out["dropped"] = dict(dropped)
    out["sizes"] = dict(Counter(f"{r['width']}x{r['height']}" for r in records).most_common(5))
    return out


def prepare_highrpd(root: Path, out_dir: Path, pothole_id: int) -> dict:
    """Keep only pothole boxes, renumbered to D40, for an external pothole-only test. Images are hard-linked."""
    d40 = DATA.classes.index("D40")
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    potholes = with_pothole = 0
    for image in images:
        label = next((p for p in image.parent.parent.joinpath("labels").glob(f"{image.stem}.txt")), None)
        rows = [r.split() for r in (label.read_text().splitlines() if label else []) if r.strip()]
        keep = [f"{d40} {' '.join(r[1:5])}" for r in rows if int(r[0]) == pothole_id]
        potholes += len(keep)
        with_pothole += bool(keep)
        target = out_dir / "images" / image.name
        if not target.exists():
            try:
                os.link(image, target)
            except OSError:
                shutil.copyfile(image, target)
        (out_dir / "labels" / f"{image.stem}.txt").write_text("".join(f"{k}\n" for k in keep))
    (out_dir / "test.txt").write_text("".join(f"./images/{p.name}\n" for p in images))
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(DATA.classes))
    (out_dir / "data.yaml").write_text(
        "# External aerial pothole test (HighRPD). Only D40 boxes are labelled; score D40 only.\n"
        f"train: test.txt\nval: test.txt\ntest: test.txt\nnc: {len(DATA.classes)}\nnames:\n{names}"
    )
    return {"images": len(images), "images_with_pothole": with_pothole, "pothole_boxes": potholes}


def negative_tags(cfg: DatasetV1Config) -> dict[str, str]:
    """stem -> 'clean' or 'hard_negative', from the records written by prepare."""
    tags = {}
    for name in ("uavpdd", "rdd_china_drone_extra"):
        for r in json.loads((cfg.processed_dir / name / "records.json").read_text()):
            if r["tag"] != convert.DAMAGED:
                tags[r["stem"]] = r["tag"]
    return tags


def list_paths(list_file: Path) -> list[Path]:
    """Resolve a relative Ultralytics list file (./ entries are relative to the list's folder)."""
    base = list_file.parent
    return [
        (base / line[2:]).resolve() if line.startswith("./") else Path(line)
        for line in list_file.read_text().splitlines()
        if line.strip()
    ]


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def verify_frozen_test(cfg: DatasetV1Config) -> str:
    """Raise unless test.txt still lists byte-identical images and labels to test_lock.json."""
    images = list_paths(cfg.output_dir / "test.txt")
    return verify_test_lock(images, label_for, cfg.output_dir / "test_lock.json")


def highrpd_subset(cfg: DatasetV1Config, seed: int = 42) -> Path:
    """All HighRPD tiles with a pothole plus as many seeded tiles without, so the shipped test stays small."""
    base = cfg.processed_dir / "highrpd"
    tiles = sorted((base / "images").iterdir())
    with_pothole = [p for p in tiles if (base / "labels" / f"{p.stem}.txt").read_text().strip()]
    without = sorted(set(tiles) - set(with_pothole))
    chosen = sorted(with_pothole + random.Random(seed).sample(without, min(len(without), len(with_pothole))))
    (base / "subset.txt").write_text("".join(f"./images/{p.name}\n" for p in chosen))
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(DATA.classes))
    yaml = base / "data_subset.yaml"
    yaml.write_text(
        "# HighRPD external pothole test: every tile with a pothole plus an equal seeded sample without.\n"
        f"train: subset.txt\nval: subset.txt\ntest: subset.txt\nnc: {len(DATA.classes)}\nnames:\n{names}"
    )
    return yaml


# --- inventory ------------------------------------------------------------------------------
def inventory(cfg: DatasetV1Config) -> tuple[dict[str, Path], dict[str, Path], set[str], dict[str, str]]:
    """Return image paths, label paths, negative stems and the origin of every image in dataset v1."""
    images, labels, origin = {}, {}, {}

    def add(stem: str, image: Path, label: Path, where: str) -> None:
        if stem in images:
            raise ValueError(f"duplicate image name {stem}: {images[stem]} and {image}")
        images[stem], labels[stem], origin[stem] = image, label, where

    for side in ("train", "val"):
        for p in sorted((cfg.unified_dir / "images" / side).iterdir()):
            if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.startswith("uav_"):
                add(p.stem, p, cfg.unified_dir / "labels" / side / f"{p.stem}.txt", "unified")
    negatives: set[str] = set()
    for name in ("uavpdd", "rdd_china_drone_extra"):
        base = cfg.processed_dir / name
        for p in sorted((base / "images").iterdir()):
            label = base / "labels" / f"{p.stem}.txt"
            add(p.stem, p, label, name)
            if not label.read_text().strip():
                negatives.add(p.stem)
    return images, labels, negatives, origin


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash_all(images: dict[str, Path], cache: Path) -> tuple[dict[str, str], dict[str, list[int]]]:
    """sha256 and 8-pose dHash for every image, cached by path, size and mtime."""
    old = json.loads(cache.read_text()) if cache.exists() else {}
    out: dict[str, dict] = {}

    def one(item):
        stem, path = item
        st = path.stat()
        key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
        if old.get(stem, {}).get("key") == key:
            return stem, old[stem]
        return stem, {"key": key, "sha256": _sha256(path), "dhash": dhash_poses(path)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        for stem, value in pool.map(one, sorted(images.items())):
            out[stem] = value
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return {s: v["sha256"] for s, v in out.items()}, {s: v["dhash"] for s, v in out.items()}


# --- calibrate ------------------------------------------------------------------------------
def calibrate(cfg: DatasetV1Config, seed: int = 0) -> dict:
    """Distances for known copies (RDD vs RDV China_Drone, unified UAV flips vs original UAV-PDD) and controls."""
    import numpy as np

    images, _, _, _ = inventory(cfg)
    _, hashes = hash_all(images, cfg.output_dir / "cache" / "hashes.json")

    def dist(a: str, b: str) -> int:
        q = np.array(hashes[a], dtype=np.uint64)
        return int(np.bitwise_count(q ^ np.uint64(hashes[b][0])).min())

    by_key: dict[str, list[str]] = defaultdict(list)
    for stem in images:
        if (key := link_key(stem)) and key.startswith("china_drone:"):
            by_key[key].append(stem)
    rdd_rdv = [
        dist(min(g), s) for g in by_key.values() for s in g if s.startswith("rdv_") and min(g).startswith("rdd_")
    ]
    # UAV-PDD: each lr_ tile has a flipped twin among the tb_ tiles of the same frame.
    rx = re.compile(cfg.uavpdd_frame_pattern)
    frames: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for stem in images:
        if m := rx.match(stem):
            frames[m["frame"]][stem.split("_")[1]].append(stem)
    uav_best = [
        min(dist(a, b) for b in by_prefix["tb"]) for by_prefix in frames.values() if by_prefix["tb"]
        for a in by_prefix["lr"]
    ]
    rng = random.Random(seed)
    names = sorted(images)
    controls = [dist(*rng.sample(names, 2)) for _ in range(20000)]
    aerial = [s for s in names if view_of(s) == "aerial"]
    aerial_controls = [dist(*rng.sample(aerial, 2)) for _ in range(20000)]

    def hist(values: list[int]) -> dict:
        c = Counter(values)
        return {"n": len(values), "le": {t: sum(v for d, v in c.items() if d <= t) for t in (0, 2, 4, 6, 8, 10, 12)}}

    report = {
        "rdd_vs_rdv_copies": hist(rdd_rdv),
        "uavpdd_lr_tile_vs_best_tb_tile_same_frame": hist(uav_best),
        "random_pairs": hist(controls),
        "random_aerial_pairs": hist(aerial_controls),
    }
    out = cfg.output_dir / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


# --- build ------------------------------------------------------------------------------
def _source_of(stem: str) -> str:
    """Source name for summaries; RDD China_Drone is counted apart from ground RDD."""
    stratum = stratum_of(stem)
    return "rdd_china_drone" if stratum.startswith("rdd:China_Drone") else stratum.split(":")[0]


def frame_links(stems: list[str], pattern: str | None) -> list[tuple[str, str]]:
    """Link tiles whose names share a frame id (regex group 'frame')."""
    if not pattern:
        return []
    rx = re.compile(pattern)
    by_frame: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        if m := rx.match(stem):
            by_frame[m["frame"]].append(stem)
    return [(g[0], s) for g in by_frame.values() for s in g[1:]]


def build(cfg: DatasetV1Config, frame_pattern: str | None, freeze: bool = False) -> dict:
    """Write the split. Without freeze it is a dry run: no lock is written, an existing lock is still checked."""
    images, labels, negatives, origin = inventory(cfg)
    digests, hashes = hash_all(images, cfg.output_dir / "cache" / "hashes.json")
    stems = sorted(images)

    copy_links: list[tuple[str, str]] = []
    by_key: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        if (key := link_key(stem)) and not key.startswith("uav:"):
            by_key[key].append(stem)
    for members in by_key.values():
        copy_links += [(members[0], s) for s in members[1:]]
    by_sha: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        by_sha[digests[stem]].append(stem)
    copy_links += [(g[0], s) for g in by_sha.values() for s in g[1:]]
    # dHash alone collides on plain asphalt (calibration.json), so candidates must also match in pixels.
    candidates = near_duplicate_pairs(hashes, cfg.dhash_max_distance)
    needed = sorted({s for a, b, _ in candidates for s in (a, b)})
    with ThreadPoolExecutor(max_workers=8) as pool:
        thumbs = dict(zip(needed, pool.map(lambda s: thumbnail(images[s]), needed), strict=True))
    scored = [(a, b, d, best_pose_mad(thumbs[a], thumbs[b])) for a, b, d in candidates]
    near = [(a, b, d, m) for a, b, d, m in scored if m <= cfg.pixel_mad_max]
    # Visual review: <= ~2 are the same photo; 2-5 mixes neighbouring frames with look-alike pavement.
    # Both keep images on one side (safe for leakage); only true copies collapse to one scored image.
    copy_links += [(a, b) for a, b, _, m in near if m <= cfg.pixel_mad_copy]
    frames = frame_links(stems, frame_pattern) + [(a, b) for a, b, _, m in near if m > cfg.pixel_mad_copy]

    groups, photo_of = build_groups(stems, copy_links, frames)

    # A negative is only a negative if nothing it is linked to carries damage labels.
    boxes = {s: read_class_counts(labels[s]) for s in stems}
    damaged_photo = {photo_of[s] for s in stems if boxes[s]}
    invalid_neg = {s for s in negatives if photo_of[s] in damaged_photo}
    negatives = negatives - invalid_neg

    split = split_three_way(groups, photo_of, boxes, negatives, cfg.shares, cfg.seed)
    excluded = dict(split.excluded)
    for side in ("val", "test"):  # labelled copies of a "negative" can't be scored as clean roads
        split.sides[side][:] = [s for s in split.sides[side] if s not in invalid_neg]
    excluded |= {s: "negative_with_labelled_copy" for s in invalid_neg}
    split.sides["train"][:] = [s for s in split.sides["train"] if s not in invalid_neg]
    assert_disjoint(split, digests)

    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for side, members in split.sides.items():
        rel = [os.path.relpath(images[s], out).replace(os.sep, "/") for s in members]
        (out / f"{side}.txt").write_text("".join(f"./{r}\n" for r in rel))
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(DATA.classes))
    (out / "data.yaml").write_text(
        "# Dataset v1 (ADR-007): grouped 70/15/15, frozen test set (see test_lock.json). Paths are relative.\n"
        f"train: train.txt\nval: val.txt\ntest: test.txt\nnc: {len(DATA.classes)}\nnames:\n{names}"
    )
    test_images = [images[s] for s in split.sides["test"]]
    if any(label_for(p) != labels[s] for s, p in zip(split.sides["test"], test_images, strict=True)):
        raise ValueError("a test label is not where Ultralytics will look for it")
    lock = out / "test_lock.json"
    if lock.exists():
        digest = verify_test_lock(test_images, label_for, lock)  # a rebuild must not change the test set
    elif freeze:
        digest = write_test_lock(test_images, label_for, lock)
    else:
        digest = "not frozen (dry run)"
    (out / "negatives_val.txt").write_text("".join(f"{s}\n" for s in split.sides["val"] if s in negatives))
    (out / "negatives_test.txt").write_text("".join(f"{s}\n" for s in split.sides["test"] if s in negatives))

    side_of = {s: side for side, members in split.sides.items() for s in members}
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "origin", "source", "view", "group_id", "photo_id", "negative", "split", "excluded"])
        for s in stems:
            w.writerow(
                [images[s].name, origin[s], stratum_of(s), view_of(s), split.group_of[s], photo_of[s],
                 s in negatives, side_of.get(s, ""), excluded.get(s, "")]
            )

    def summarise(members: list[str]) -> dict:
        cls: Counter = Counter()
        for s in members:
            cls.update(boxes[s])
        return {
            "images": len(members),
            "negatives": sum(s in negatives for s in members),
            "boxes": sum(cls.values()),
            "by_view": dict(sorted(Counter(view_of(s) for s in members).items())),
            "by_source": dict(sorted(Counter(_source_of(s) for s in members).items())),
            "by_class": {n: cls[i] for i, n in enumerate(DATA.classes)},
            "aerial_by_class": {
                n: sum(boxes[s][i] for s in members if view_of(s) == "aerial") for i, n in enumerate(DATA.classes)
            },
        }

    summary = {
        "seed": cfg.seed,
        "shares": cfg.shares,
        "dhash_max_distance": cfg.dhash_max_distance,
        "frame_pattern": frame_pattern,
        "images": len(stems),
        "groups": len(groups),
        "dhash_candidates": len(candidates),
        "near_duplicate_pairs": len(near),
        "near_duplicate_pairs_between_sources": sum(stratum_of(a) != stratum_of(b) for a, b, _, _ in near),
        "frame_links": len(frames),
        "invalid_negatives": len(invalid_neg),
        "excluded": dict(Counter(excluded.values())),
        "test_lock_sha256": digest,
        **{side: summarise(members) for side, members in split.sides.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "near_duplicates.csv").write_text(
        "a,b,dhash_bits,pixel_mad,linked,already_linked_by_name\n"
        + "".join(
            f"{a},{b},{d},{m:.2f},{m <= cfg.pixel_mad_max},{link_key(a) is not None and link_key(a) == link_key(b)}\n"
            for a, b, d, m in scored
        )
    )
    return summary
