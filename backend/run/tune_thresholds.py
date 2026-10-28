"""Pick per-class confidence thresholds on val (SCRUM-26).  python -m run.tune_thresholds --weights PATH
Copy the printed values into config.INFER.conf_thresholds."""

import argparse
from pathlib import Path

from ml.build import build_data_yaml_from_disk
from ml.thresholds import tune_thresholds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    a = p.parse_args()
    print(tune_thresholds(a.weights, build_data_yaml_from_disk()))


if __name__ == "__main__":
    main()
