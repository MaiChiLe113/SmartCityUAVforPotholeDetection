"""Export the registered FP32 model to an INT8 edge package (SCRUM-35).
python -m run.export_edge --target jetson|pi --weights PATH --version V"""

import argparse
from pathlib import Path

from config import EXPORT
from ml.build import build_data_yaml_from_disk, build_exporter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, choices=list(EXPORT.formats))
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--version", required=True, help="registry model version")
    a = p.parse_args()
    print(build_exporter(a.target)(a.weights, build_data_yaml_from_disk(), a.version))


if __name__ == "__main__":
    main()
