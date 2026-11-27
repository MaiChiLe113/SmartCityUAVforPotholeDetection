"""Evaluate on the frozen test set: overall, per class, ground vs aerial (SCRUM-43).
python -m run.evaluate --weights PATH [--split test|field|val]"""

import argparse
from pathlib import Path

from ml.build import build_data_yaml_from_disk
from ml.evaluate import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--split", default="test", choices=["test", "field", "val"])
    a = p.parse_args()
    print(evaluate(a.weights, build_data_yaml_from_disk(), a.split))


if __name__ == "__main__":
    main()
