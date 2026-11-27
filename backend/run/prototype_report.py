"""Score the prototype on ground, aerial and per-source validation images.

python -m run.prototype_report [--weights path/to/best.pt]
"""

import argparse
import json
from pathlib import Path

from config import DATA, PATHS, PROTOTYPE_SPLIT
from ml.evaluate import evaluate_val_subsets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=PATHS.runs_dir / "train" / "yolo26n" / "weights" / "best.pt")
    a = p.parse_args()

    report = evaluate_val_subsets(
        a.weights, PROTOTYPE_SPLIT.output_dir / "val.txt", PATHS.reports_dir / "prototype" / "subsets", DATA.classes
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
