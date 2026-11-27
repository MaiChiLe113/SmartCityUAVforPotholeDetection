"""Check that this machine can run GPU retraining.  python -m run.check_env [--cpu]

Run it on any PC before registering it as the self-hosted `gpu` runner, and at the start of every GPU job.
Exits with status 1 unless every check passes.
"""

import argparse
import json
import shutil
import subprocess
import sys

from config import DATASET_V1, RETRAIN, TRAIN
from ml import dataset_v1

PINNED_ULTRALYTICS = "8.4.184"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", action="store_true", help="skip the GPU checks (hosted CPU runner)")
    a = p.parse_args()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    check("python 3.11+", sys.version_info >= (3, 11), sys.version.split()[0])
    try:
        import torch

        cuda = torch.cuda.is_available()
        detail = f"torch {torch.__version__}"
        if cuda:
            props = torch.cuda.get_device_properties(0)
            detail += f", {props.name}, {props.total_memory / 2**30:.1f} GB"
        check("CUDA GPU", cuda or a.cpu, detail if cuda else detail + ", no CUDA device")
    except ImportError:
        check("torch installed", False, "pip install the ml extra")
    try:
        import ultralytics

        check("ultralytics pinned", ultralytics.__version__ == PINNED_ULTRALYTICS, ultralytics.__version__)
    except ImportError:
        check("ultralytics installed", False, "pip install the ml extra")

    dvc = shutil.which("dvc")
    check("dvc on PATH", dvc is not None, dvc or "pip install the dvc extra")
    if dvc:
        remote = subprocess.run([dvc, "remote", "list"], capture_output=True, text=True).stdout.strip()
        check("dvc remote configured", bool(remote), remote.splitlines()[0] if remote else "run dvc remote add")

    summary_path = DATASET_V1.output_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for side in ("train", "val", "test"):
            paths = dataset_v1.list_paths(DATASET_V1.output_dir / f"{side}.txt")
            missing = sum(not q.exists() for q in paths)
            check(f"{side} images present", missing == 0, f"{len(paths)} listed, {missing} missing")
        if not a.cpu:
            try:
                digest = dataset_v1.verify_frozen_test(DATASET_V1)
                check("frozen test lock", digest == summary["test_lock_sha256"], digest[:12])
            except (ValueError, FileNotFoundError) as e:
                check("frozen test lock", False, str(e))
    else:
        check("dataset v1 lists", False, "run dvc pull")

    if RETRAIN.champion.exists():
        try:
            from ultralytics import YOLO

            model = YOLO(str(RETRAIN.champion))
            check("champion loads", True, f"{RETRAIN.champion.name}, {len(model.names)} classes")
        except Exception as e:  # noqa: BLE001 - report any load failure
            check("champion loads", False, str(e))
    else:
        check("champion present", False, f"{RETRAIN.champion} missing; run dvc pull")
    check("training device", True, f"DEVICE={TRAIN.device}, batch={RETRAIN.batch}")

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'OK  ' if ok else 'FAIL'} {name.ljust(width)}  {detail}")
    ready = all(ok for _, ok, _ in results)
    print("READY" if ready else "NOT READY")
    sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
