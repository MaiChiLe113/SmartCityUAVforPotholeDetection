"""Retrain on data/retrain (run.ingest_batches first).

python -m run.retrain --mode gpu           # warm start from the champion, params.yaml epochs, then tune
python -m run.retrain --mode cpu --plan    # write the CPU-demo lists and the files they need (for dvc pull)
python -m run.retrain --mode cpu           # hosted CPU demo: short run on a small split, never promoted
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from config import DATA, PATHS, RETRAIN, TRAIN
from ml.build import build_model_from_weights, compile_train_args
from ml.drift import label_for
from ml.retrain import cpu_split, stamp_last_retrain, write_training_history
from ml.thresholds import tune_thresholds
from ml.train import train_one


def train(data_yaml, name: str, **overrides) -> object:
    args = compile_train_args(TRAIN.candidates[-1], data_yaml, epochs=overrides.pop("epochs"))
    args.update({"name": name, "exist_ok": True, **overrides})
    run_dir = PATHS.runs_dir / "train" / name
    if run_dir.exists():
        shutil.rmtree(run_dir)  # each retrain starts from a clean run folder
    return train_one(build_model_from_weights(RETRAIN.champion), args)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gpu", "cpu"], required=True)
    p.add_argument("--plan", action="store_true", help="cpu only: write lists and required files, do not train")
    a = p.parse_args()
    stamp_last_retrain(RETRAIN)

    if a.mode == "gpu":
        data_yaml = RETRAIN.retrain_dir / "data.yaml"
        result = train(data_yaml, RETRAIN.run_name, epochs=RETRAIN.epochs, batch=RETRAIN.batch)
        run_dir = PATHS.runs_dir / "train" / RETRAIN.run_name
        write_training_history(run_dir, RETRAIN)
        tune_thresholds(result.best_weights, data_yaml, DATA.classes, RETRAIN.metrics_dir / "thresholds.json")
        print(json.dumps(result.metrics, indent=2))
        return

    data_yaml, needed = cpu_split(RETRAIN)
    if a.plan:
        paths = set(needed) | {label_for(p) for p in needed}
        files = sorted(Path(os.path.relpath(p, Path.cwd())).as_posix() for p in paths)
        out = RETRAIN.cpu_dir / "required_files.txt"
        out.write_text("\n".join(files) + "\n")
        print(f"{len(needed)} images listed in {out}")
        return
    name = f"{RETRAIN.run_name}-cpu"
    result = train(
        data_yaml,
        name,
        epochs=RETRAIN.cpu_epochs,
        imgsz=RETRAIN.cpu_imgsz,
        device="cpu",
        batch=16,
        workers=2,
        amp=False,
    )
    write_training_history(PATHS.runs_dir / "train" / name, RETRAIN)
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
