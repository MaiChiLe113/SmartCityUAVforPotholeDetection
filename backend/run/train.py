"""Train one model on the frozen split (SCRUM-19).  python -m run.train --model yolov8n.pt [--smoke]"""

import argparse
from pathlib import Path

from config import TRAIN
from ml.build import build_data_yaml_from_disk, build_model, compile_train_args
from ml.train import train_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=TRAIN.baseline, choices=TRAIN.candidates)
    p.add_argument("--smoke", action="store_true", help="few epochs on a small subset")
    p.add_argument("--data", type=Path, help="data.yaml to train on, e.g. data/interim_v1/data.yaml")
    p.add_argument("--name", help="run folder name under runs/train (default: model name)")
    a = p.parse_args()

    data_yaml = a.data.resolve() if a.data else build_data_yaml_from_disk()
    args = compile_train_args(a.model, data_yaml, smoke=a.smoke)
    if a.name:
        args["name"] = a.name
    result = train_one(build_model(a.model), args)
    print(result)


if __name__ == "__main__":
    main()
