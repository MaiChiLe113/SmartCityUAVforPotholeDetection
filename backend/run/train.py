"""Train one model on the frozen split (SCRUM-19).  python -m run.train --model yolov8n.pt [--smoke]"""

import argparse

from config import TRAIN
from ml.build import build_data_yaml_from_disk, build_model, compile_train_args
from ml.train import train_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=TRAIN.baseline, choices=TRAIN.candidates)
    p.add_argument("--smoke", action="store_true", help="few epochs on a small subset")
    a = p.parse_args()

    args = compile_train_args(a.model, build_data_yaml_from_disk(), smoke=a.smoke)
    result = train_one(build_model(a.model), args)
    print(result)


if __name__ == "__main__":
    main()
