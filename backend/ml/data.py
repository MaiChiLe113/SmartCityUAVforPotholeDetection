"""Dataset download, cleaning and the frozen 70/15/15 split (§3.1a). Owner: Chi (SCRUM-18)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import DATA, PATHS, DataConfig


@dataclass
class SplitManifest:
    """File lists for each subset plus the hash that proves the test set never changed."""

    train: list[Path]
    val: list[Path]
    test: list[Path]
    test_hash: str
    views: dict[str, str]  # image path -> "ground" | "aerial"


def download_dataset(cfg: DataConfig = DATA, dest: Path = PATHS.raw_dir) -> Path:
    """Download the Unified Road Defect Dataset from Hugging Face into `dest`.

    Use `huggingface_hub.snapshot_download(repo_id=cfg.hf_dataset_id, repo_type=cfg.hf_repo_type,
    revision=cfg.hf_revision, token=cfg.hf_token, local_dir=dest)`. Skip if already present.
    Returns the local dataset root.
    """
    raise NotImplementedError("TODO(SCRUM-18, Chi): download dataset from Hugging Face")


def deduplicate(images: list[Path], threshold: int = DATA.dedupe_hash_threshold) -> list[Path]:
    """Remove near-duplicate images using perceptual hashing (imagehash.phash)."""
    raise NotImplementedError("TODO(SCRUM-18, Chi): perceptual-hash dedupe")


def group_related(images: list[Path]) -> dict[str, list[Path]]:
    """Group frames of the same road/sequence so a group never crosses subsets."""
    raise NotImplementedError("TODO(SCRUM-18, Chi): group frames of the same road")


def detect_view(image: Path) -> str:
    """Return "ground" (dashcam/motorbike) or "aerial" (drone) for one image."""
    raise NotImplementedError("TODO(SCRUM-18, Chi): ground vs aerial label from source metadata")


def make_splits(groups: dict[str, list[Path]], cfg: DataConfig = DATA) -> SplitManifest:
    """Group-aware split stratified by view, using cfg.split and cfg.seed."""
    raise NotImplementedError("TODO(SCRUM-18, Chi): stratified group split 70/15/15")


def freeze_test_set(manifest: SplitManifest, splits_dir: Path = PATHS.splits_dir) -> None:
    """Write split lists to splits_dir and refuse to overwrite an existing frozen test set."""
    raise NotImplementedError("TODO(SCRUM-18, Chi): write split files, store test hash")


def load_manifest(splits_dir: Path = PATHS.splits_dir) -> SplitManifest:
    """Read split lists written by freeze_test_set and verify the test hash."""
    raise NotImplementedError("TODO(SCRUM-18, Chi): load frozen split")


def write_data_yaml(manifest: SplitManifest, out_dir: Path = PATHS.processed_dir) -> Path:
    """Write the Ultralytics data.yaml (train/val/test lists, nc, names) and return its path."""
    raise NotImplementedError("TODO(SCRUM-18, Chi): write Ultralytics data.yaml")
