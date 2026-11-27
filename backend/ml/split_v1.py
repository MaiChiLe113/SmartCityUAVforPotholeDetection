"""Grouped three-way split (train/val/test) for dataset v1, near-duplicate search and the frozen test lock.

Two kinds of link feed the groups:
  copy links  - the same photo under another name (RDV re-export, identical bytes, near-identical pixels)
  frame links - different crops of one camera frame (UAV-PDD tiles)
Both keep images on one side of the split. Only copy links decide what validation and test keep,
so every distinct photo is scored once. numpy and cv2 are imported inside functions (CI has neither).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ml.grouping import stratum_of, view_of

SIDES = ("train", "val", "test")


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, node: str) -> str:
        self.parent.setdefault(node, node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def build_groups(
    stems: Iterable[str],
    copy_links: Iterable[tuple[str, str]],
    frame_links: Iterable[tuple[str, str]] = (),
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Return {group_id: members} using both link kinds, and {stem: photo_id} using copy links only."""
    stems = sorted(set(stems))
    copy_links = list(copy_links)
    photos, groups = UnionFind(), UnionFind()
    for stem in stems:
        photos.find(stem)
        groups.find(stem)
    for a, b in copy_links:
        photos.union(a, b)
        groups.union(a, b)
    for a, b in frame_links:
        groups.union(a, b)

    def collect(uf: UnionFind) -> dict[str, str]:
        members: dict[str, list[str]] = defaultdict(list)
        for stem in stems:
            members[uf.find(stem)].append(stem)
        return {stem: min(group) for group in members.values() for stem in group}

    group_of = collect(groups)
    by_group: dict[str, list[str]] = defaultdict(list)
    for stem in stems:
        by_group[group_of[stem]].append(stem)
    return dict(by_group), collect(photos)


def representatives(members: list[str], photo_of: Mapping[str, str]) -> list[str]:
    """One image per distinct photo: the first name, which prefers an rdd_ original over an rdv_ copy."""
    by_photo: dict[str, list[str]] = defaultdict(list)
    for stem in members:
        by_photo[photo_of[stem]].append(stem)
    return sorted(min(stems) for stems in by_photo.values())


@dataclass(frozen=True)
class SplitV1:
    sides: dict[str, list[str]]
    excluded: dict[str, str]  # stem -> reason
    group_of: dict[str, str]
    group_side: dict[str, str]


def split_three_way(
    groups: Mapping[str, list[str]],
    photo_of: Mapping[str, str],
    boxes: Mapping[str, Counter],
    negatives: set[str],
    shares: Mapping[str, Mapping[str, float]],
    seed: int,
) -> SplitV1:
    """Assign whole groups to train/val/test per source stratum, spreading each group's rarest class.

    shares = {"val": {"ground": .15, "aerial": .15}, "test": {...}}. Groups made only of negatives
    (clean or hard-negative images) go half to val and half to test, because negatives stay out of train.
    """
    totals: Counter = Counter()
    for counts in boxes.values():
        totals.update(counts)

    def rarest(group_id: str) -> int:
        present = {c for stem in groups[group_id] for c in boxes.get(stem, {})}
        return min((totals[c] for c in present), default=0)

    strata: dict[str, list[str]] = defaultdict(list)
    for group_id in sorted(groups):
        negative_only = all(stem in negatives for stem in groups[group_id])
        strata[("neg:" if negative_only else "") + stratum_of(group_id)].append(group_id)

    rng = random.Random(seed)
    group_side: dict[str, str] = {}
    for name in sorted(strata):
        ids = strata[name]
        rng.shuffle(ids)
        if name.startswith("neg:"):
            for i, group_id in enumerate(ids):
                group_side[group_id] = "val" if i % 2 == 0 else "test"
            continue
        ids.sort(key=rarest)
        view = view_of(ids[0])
        v, t = shares["val"][view], shares["test"][view]
        held = v + t
        picked = 0
        for i, group_id in enumerate(ids):
            if int((i + 1) * held) > int(i * held):
                group_side[group_id] = "test" if int((picked + 1) * t / held) > int(picked * t / held) else "val"
                picked += 1
            else:
                group_side[group_id] = "train"

    sides: dict[str, list[str]] = {side: [] for side in SIDES}
    excluded: dict[str, str] = {}
    group_of: dict[str, str] = {}
    for group_id, members in groups.items():
        side = group_side[group_id]
        group_of.update(dict.fromkeys(members, group_id))
        if side == "train":
            for stem in members:
                if stem in negatives:
                    excluded[stem] = "negative_in_train_group"
                else:
                    sides["train"].append(stem)
        else:
            kept = set(representatives(members, photo_of))
            for stem in members:
                if stem in kept:
                    sides[side].append(stem)
                else:
                    excluded[stem] = f"copy_in_{side}_group"
    return SplitV1({s: sorted(v) for s, v in sides.items()}, excluded, group_of, group_side)


def assert_disjoint(split: SplitV1, digests: Mapping[str, str] | None = None) -> None:
    """Raise if an image, a group or identical bytes appear on two sides."""
    seen: dict[str, str] = {}
    for side, stems in split.sides.items():
        for stem in stems:
            if stem in seen:
                raise ValueError(f"{stem} is in {seen[stem]} and {side}")
            seen[stem] = side
            if split.group_side[split.group_of[stem]] != side:
                raise ValueError(f"{stem} is on {side} but its group is on {split.group_side[split.group_of[stem]]}")
    if digests:
        sides_of: dict[str, set[str]] = defaultdict(set)
        for side, stems in split.sides.items():
            for stem in stems:
                sides_of[digests[stem]].add(side)
        if crossing := [d for d, s in sides_of.items() if len(s) > 1]:
            raise ValueError(f"{len(crossing)} identical files cross the split")


# --- Near-duplicate search -------------------------------------------------------------
def dhash_poses(path: Path, crop: float = 0.9) -> list[int]:
    """64-bit difference hashes of the centre crop in all 8 flip/rotation poses (pose 0 = as stored)."""
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"unreadable image: {path}")
    h, w = image.shape
    dy, dx = int(h * (1 - crop) / 2), int(w * (1 - crop) / 2)
    small = cv2.resize(image[dy : h - dy, dx : w - dx], (64, 64), interpolation=cv2.INTER_AREA)
    hashes = []
    for k in range(4):
        for flip in (False, True):
            pose = np.rot90(small, k)
            pose = np.fliplr(pose) if flip else pose
            tiny = cv2.resize(np.ascontiguousarray(pose), (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
            bits = (tiny[:, 1:] > tiny[:, :-1]).flatten()
            hashes.append(int("".join("1" if b else "0" for b in bits), 2))
    return hashes


def near_duplicate_pairs(
    hashes: Mapping[str, list[int]], max_distance: int, block: int = 256
) -> list[tuple[str, str, int]]:
    """Pairs whose pose-0 hash is within max_distance bits of any pose of the other image."""
    import numpy as np

    names = sorted(hashes)
    base = np.array([hashes[n][0] for n in names], dtype=np.uint64)
    poses = np.array([hashes[n] for n in names], dtype=np.uint64)  # (N, 8)
    pairs = []
    for start in range(0, len(names), block):
        q = poses[start : start + block]  # (b, 8)
        dist = np.bitwise_count(q[:, :, None] ^ base[None, None, :]).min(axis=1)  # (b, N)
        for i, j in zip(*np.nonzero(dist <= max_distance), strict=True):
            a, b = start + int(i), int(j)
            if a < b:
                pairs.append((names[a], names[b], int(dist[i, j])))
            elif a > b and dist[i, j] <= max_distance:
                pairs.append((names[b], names[a], int(dist[i, j])))
    best: dict[tuple[str, str], int] = {}
    for a, b, d in pairs:
        best[(a, b)] = min(d, best.get((a, b), 64))
    return sorted((a, b, d) for (a, b), d in best.items())


def thumbnail(path: Path, size: int = 64, crop: float = 0.9):
    """Grayscale centre crop resized to size x size (float32), for pixel-level confirmation of candidates."""
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    dy, dx = int(h * (1 - crop) / 2), int(w * (1 - crop) / 2)
    return cv2.resize(image[dy : h - dy, dx : w - dx], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def best_pose_mad(a, b) -> float:
    """Smallest mean absolute pixel difference between a and any flip/rotation of b."""
    import numpy as np

    best = float("inf")
    for k in range(4):
        r = np.rot90(b, k)
        for pose in (r, np.fliplr(r)):
            best = min(best, float(np.abs(a - pose).mean()))
    return best


# --- Frozen test lock -------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_entries(image_paths: list[Path], label_of) -> list[dict]:
    return [
        {"image": p.name, "image_sha256": _sha256(p), "label_sha256": _sha256(label_of(p))}
        for p in sorted(image_paths, key=lambda x: x.name)
    ]


def combined_hash(entries: list[dict]) -> str:
    text = "".join(f"{e['image']}:{e['image_sha256']}:{e['label_sha256']}\n" for e in entries)
    return hashlib.sha256(text.encode()).hexdigest()


def write_test_lock(image_paths: list[Path], label_of, out: Path) -> str:
    entries = lock_entries(image_paths, label_of)
    digest = combined_hash(entries)
    out.write_text(json.dumps({"images": len(entries), "sha256": digest, "entries": entries}, indent=1))
    return digest


def verify_test_lock(image_paths: list[Path], label_of, lock: Path) -> str:
    """Raise unless the test images and labels are byte-identical to the frozen lock."""
    frozen = json.loads(lock.read_text())
    entries = lock_entries(image_paths, label_of)
    if len(entries) != frozen["images"] or combined_hash(entries) != frozen["sha256"]:
        changed = {e["image"] for e in entries} ^ {e["image"] for e in frozen["entries"]}
        raise ValueError(f"test set differs from its lock ({len(changed)} names differ)")
    return frozen["sha256"]
