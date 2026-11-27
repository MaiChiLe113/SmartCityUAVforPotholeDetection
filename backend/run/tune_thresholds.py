"""Pick per-class confidence thresholds on val.  python -m run.tune_thresholds --weights PATH [--data YAML]
Copy the printed confidences into config.INFER.conf_thresholds."""

import argparse
import json
from pathlib import Path

from config import DATA, DATASET_V1, PATHS
from ml.thresholds import tune_thresholds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--data", type=Path, default=DATASET_V1.output_dir / "data.yaml")
    a = p.parse_args()
    out = PATHS.reports_dir / "v1" / "thresholds.json"
    print(json.dumps(tune_thresholds(a.weights, a.data.resolve(), DATA.classes, out), indent=2))


if __name__ == "__main__":
    main()
