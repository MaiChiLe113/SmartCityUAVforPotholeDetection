"""Group related images and split them whole, so no known copy crosses train and val (SCRUM-38).

The links come from filename patterns plus byte-identical files. This is a prototype split: the
SCRUM-18 audit adds perceptual matching and produces the real one. Nothing here touches
data/raw or data/splits, and the output is a set of file lists, so no image is copied.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

_RDD = re.compile(r"^rdd_(?P<region>.+?)_(?P=region)_(?P<number>\d+)$")
_RDV_DRONE = re.compile(r"^rdv_China_Drone_(\d+)_jpg\.rf\.[0-9a-f]+$")
_RDV = re.compile(r"^rdv_(.+?)_(?:jpg|jpeg|png)\.rf\.[0-9a-f]+$")
_UAV = re.compile(r"^uav_([a-z0-9]+)_(\d+)_(?:top|bottom)_(?:left|right)$")
_UAVPDD = re.compile(r"^uavpdd_(?P<name>.+)$")  # tiles converted from the original UAV-PDD2023 release
_UAV_PREFERENCE = ("lr", "tb", "r180")
AERIAL_STRATA = {"rdd:China_Drone", "rdv", "uav", "uavpdd"}


def stratum_of(stem: str) -> str:
    """Return the data source of an image. Unknown names raise, because they would escape grouping."""
    if m := _RDD.match(stem):
        return f"rdd:{m['region']}"
    if _RDV.match(stem):
        return "rdv"
    if _UAV.match(stem):
        return "uav"
    if _UAVPDD.match(stem):
        return "uavpdd"
    raise ValueError(f"unrecognised image name: {stem!r}")


def view_of(stem: str) -> str:
    return "aerial" if stratum_of(stem) in AERIAL_STRATA else "ground"


def link_key(stem: str) -> str | None:
    """Return the key shared by every copy of this image's source photo or frame, if it has one."""
    if (m := _RDD.match(stem)) and m["region"] == "China_Drone":
        return f"china_drone:{m['number']}"
    if m := _RDV_DRONE.match(stem):
        return f"china_drone:{m[1]}"
    if m := _RDV.match(stem):
        return f"rdv:{m[1]}"
    if m := _UAV.match(stem):
        return f"uav:{m[2]}"  # lr, tb and r180 are flips of one frame, so the prefix is left out
    return None


def group_images(stems: Iterable[str], digests: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    """Merge images linked by name or by identical bytes. Each group's id is its first stem."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    stems = list(stems)
    for stem in stems:
        find(stem)
        if key := link_key(stem):
            parent[find(stem)] = find(f"key:{key}")
        if digests and stem in digests:
            parent[find(stem)] = find(f"sha:{digests[stem]}")

    members: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        members[find(stem)].append(stem)
    return {min(group): sorted(group) for group in members.values()}


def val_representatives(members: list[str]) -> list[str]:
    """Pick what a validation group contributes, so one photo is never scored more than once."""
    uav = [s for s in members if s.startswith("uav_")]
    if uav:
        tiles = Counter(_UAV.match(s)[1] for s in uav)
        best = max(tiles, key=lambda p: (tiles[p], -_prefix_rank(p)))
        return [s for s in uav if _UAV.match(s)[1] == best]
    return [min(members)]  # sorts rdd_ before rdv_, so the original is preferred over a copy


def _prefix_rank(prefix: str) -> int:
    return _UAV_PREFERENCE.index(prefix) if prefix in _UAV_PREFERENCE else len(_UAV_PREFERENCE)


@dataclass(frozen=True)
class GroupedSplit:
    train: list[str]
    val: list[str]
    excluded: list[str]  # copies from validation groups: never scored, and unsafe to train on
    group_of: dict[str, str]


def split_groups(
    groups: Mapping[str, list[str]],
    boxes: Mapping[str, Counter],
    val_share: Mapping[str, float],
    seed: int,
) -> GroupedSplit:
    """Send whole groups to validation, spread evenly over each source and each group's rarest class."""
    totals: Counter = Counter()
    for members in groups.values():
        for stem in members:
            totals.update(boxes.get(stem, {}))

    def rarest_count(group_id: str) -> int:
        present = {c for stem in groups[group_id] for c in boxes.get(stem, {})}
        return min((totals[c] for c in present), default=0)

    strata: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(groups):
        strata[stratum_of(group_id)].append(group_id)

    rng = random.Random(seed)
    val_groups: set[str] = set()
    for stratum in sorted(strata):
        ids = strata[stratum]
        rng.shuffle(ids)
        ids.sort(key=rarest_count)  # stable, so the order stays random within one rarest class
        share = val_share[view_of(ids[0])]
        val_groups.update(g for i, g in enumerate(ids) if int((i + 1) * share) > int(i * share))

    train, val, excluded, group_of = [], [], [], {}
    for group_id, members in groups.items():
        group_of.update(dict.fromkeys(members, group_id))
        if group_id in val_groups:
            kept = set(val_representatives(members))
            val += [s for s in members if s in kept]
            excluded += [s for s in members if s not in kept]
        else:
            train += members
    return GroupedSplit(sorted(train), sorted(val), sorted(excluded), group_of)


def assert_no_leak(split: GroupedSplit, digests: Mapping[str, str] | None = None) -> None:
    """Raise if an image, a group or a file's bytes appear on both sides."""
    train, val = set(split.train), set(split.val)
    if overlap := train & val:
        raise ValueError(f"{len(overlap)} images are in both splits")
    if overlap := {split.group_of[s] for s in train} & {split.group_of[s] for s in val | set(split.excluded)}:
        raise ValueError(f"{len(overlap)} groups cross the split")
    if digests and (overlap := {digests[s] for s in train} & {digests[s] for s in val}):
        raise ValueError(f"{len(overlap)} identical files cross the split")


def read_class_counts(label_path: Path) -> Counter:
    return Counter(int(line.split()[0]) for line in label_path.read_text().splitlines() if line.strip())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summarise(names: list[str], boxes: Mapping[str, Counter], class_names: tuple[str, ...]) -> dict:
    classes: Counter = Counter()
    for stem in names:
        classes.update(boxes[stem])
    return {
        "images": len(names),
        "boxes": sum(classes.values()),
        "by_view": dict(sorted(Counter(view_of(s) for s in names).items())),
        "by_source": dict(sorted(Counter(s.split("_", 1)[0] for s in names).items())),
        "by_class": {name: classes[i] for i, name in enumerate(class_names)},
    }


def build_grouped_split(
    source_dir: Path,
    output_dir: Path,
    val_share: Mapping[str, float],
    seed: int,
    class_names: tuple[str, ...],
) -> dict:
    """Write train.txt, val.txt, data.yaml, manifest.csv and summary.json for the extracted dataset."""
    paths: dict[str, Path] = {}
    published: dict[str, str] = {}
    for side in ("train", "val"):
        for path in sorted((source_dir / "images" / side).iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.stem in paths:
                raise ValueError(f"image name appears twice: {path.stem}")
            paths[path.stem] = path
            published[path.stem] = side

    boxes = {
        stem: read_class_counts(source_dir / "labels" / published[stem] / f"{stem}.txt") for stem in paths
    }
    digests = {stem: sha256_file(path) for stem, path in paths.items()}
    groups = group_images(paths, digests)
    split = split_groups(groups, boxes, val_share, seed)
    assert_no_leak(split, digests)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, stems in (("train", split.train), ("val", split.val)):
        (output_dir / f"{name}.txt").write_text("".join(f"{paths[s]}\n" for s in stems))
    names_yaml = "".join(f"  {i}: {name}\n" for i, name in enumerate(class_names))
    (output_dir / "data.yaml").write_text(
        "# Prototype only: grouped by filename links and identical bytes, not by the SCRUM-18 audit.\n"
        f"path: {output_dir.resolve().as_posix()}\ntrain: train.txt\nval: val.txt\n"
        f"nc: {len(class_names)}\nnames:\n{names_yaml}"
    )

    new_split = {s: "train" for s in split.train} | {s: "val" for s in split.val}
    new_split |= {s: "excluded" for s in split.excluded}
    with (output_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "group_id", "source", "view", "published_split", "new_split"])
        for stem in sorted(paths):
            writer.writerow(
                [
                    paths[stem].name,
                    split.group_of[stem],
                    stratum_of(stem),
                    view_of(stem),
                    published[stem],
                    new_split[stem],
                ]
            )

    val_ids = {split.group_of[s] for s in split.val}
    summary = {
        "seed": seed,
        "val_share": dict(val_share),
        "published_images": len(paths),
        "groups": len(groups),
        "val_groups": len(val_ids),
        "train": _summarise(split.train, boxes, class_names),
        "val": _summarise(split.val, boxes, class_names),
        "excluded": _summarise(split.excluded, boxes, class_names),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
