"""Build the handoff folder (data/handoff/yolo26n-v1).  python -m run.handoff [--run runs/train/yolo26n-v1]
Write README.md into that folder afterwards from evaluation.json."""

import argparse
from pathlib import Path

from config import DATASET_V1, PATHS
from ml.handoff import build_handoff


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=PATHS.runs_dir / "train" / "yolo26n-v1")
    a = p.parse_args()
    out = build_handoff(DATASET_V1, a.run / "weights" / "best.pt", a.run, PATHS.reports_dir / "v1")
    print(out)


if __name__ == "__main__":
    main()
