"""Train one model on the frozen split (SCRUM-19).

python -m run.train --model yolov8n.pt [--smoke]                       # from COCO weights
python -m run.train --init artifacts/models/champion.pt --epochs 20    # warm start from the champion
"""

import argparse
from pathlib import Path

from config import TRAIN
from ml.build import build_data_yaml_from_disk, build_model, build_model_from_weights, compile_train_args
from ml.train import train_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=TRAIN.baseline, choices=TRAIN.candidates)
    p.add_argument("--init", type=Path, help="start from these weights instead (only the champion is allowed)")
    p.add_argument("--epochs", type=int, help="override TRAIN.epochs")
    p.add_argument("--smoke", action="store_true", help="few epochs on a small subset")
    p.add_argument("--data", type=Path, help="data.yaml to train on, e.g. data/interim_v1/data.yaml")
    p.add_argument("--name", help="run folder name under runs/train (default: model name)")
    a = p.parse_args()

    data_yaml = a.data.resolve() if a.data else build_data_yaml_from_disk()
    model = build_model_from_weights(a.init) if a.init else build_model(a.model)
    args = compile_train_args(a.model, data_yaml, smoke=a.smoke, epochs=a.epochs)
    if a.name:
        args["name"] = a.name
    print(train_one(model, args))


if __name__ == "__main__":
    main()
