"""Retraining helpers: training history, the small CPU-demo split, and model version bookkeeping."""

from __future__ import annotations

import csv
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from config import DATA, RETRAIN, RetrainConfig


def training_history(results_csv: Path) -> list[dict]:
    """Rows of an Ultralytics results.csv as dicts of floats (column names stripped)."""
    with results_csv.open() as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): float(v) for k, v in row.items() if v not in (None, "")} for row in rows]


def write_training_history(run_dir: Path, cfg: RetrainConfig = RETRAIN) -> Path:
    history = training_history(run_dir / "results.csv")
    out = cfg.metrics_dir / "training_history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run_dir": run_dir.name, "epochs": history}, indent=2))
    return out


def seeded_sample(items: list, n: int, seed: int) -> list:
    pool = sorted(items, key=str)
    random.Random(seed).shuffle(pool)
    return sorted(pool[:n], key=str)


def cpu_split(cfg: RetrainConfig = RETRAIN) -> tuple[Path, list[Path]]:
    """Small lists for the hosted CPU demo: new train images plus a replay sample of v1 train, and a
    validation sample. Returns the data.yaml and every image whose file must be present (with labels).
    """
    from ml import dataset_v1
    from ml.ingest import _relative

    train_all = dataset_v1.list_paths(cfg.retrain_dir / "train.txt")
    val_all = dataset_v1.list_paths(cfg.retrain_dir / "val.txt")
    new_train = [p for p in train_all if "new" in p.parts]
    old_train = [p for p in train_all if "new" not in p.parts]
    train = new_train + seeded_sample(old_train, cfg.cpu_replay, cfg.seed)
    new_val = [p for p in val_all if "new" in p.parts]
    old_val = [p for p in val_all if "new" not in p.parts]
    val = new_val + seeded_sample(old_val, max(cfg.cpu_val - len(new_val), 0), cfg.seed)

    out = cfg.cpu_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text("".join(_relative(p, out) + "\n" for p in train))
    (out / "val.txt").write_text("".join(_relative(p, out) + "\n" for p in val))
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(DATA.classes))
    yaml = out / "data.yaml"
    yaml.write_text(
        "# Hosted CPU demo split (not promotable): new train images, a v1 replay sample and a val sample.\n"
        f"train: train.txt\nval: val.txt\nnc: {len(DATA.classes)}\nnames:\n{names}"
    )
    return yaml, train + val


def current_version(cfg: RetrainConfig = RETRAIN) -> str:
    path = cfg.metadata_dir / "model_version.txt"
    return path.read_text().strip() if path.exists() else "v1"


def next_version(version: str) -> str:
    """v1 -> v2. Anything that is not v<number> starts again at v2."""
    try:
        return f"v{int(version.lstrip('v')) + 1}"
    except ValueError:
        return "v2"


def stamp_last_retrain(cfg: RetrainConfig = RETRAIN) -> str:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    cfg.metadata_dir.mkdir(parents=True, exist_ok=True)
    (cfg.metadata_dir / "last_retrain.txt").write_text(now + "\n")
    return now
