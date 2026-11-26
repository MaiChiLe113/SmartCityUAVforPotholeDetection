"""Build dataset v1 (ADR-007).

python -m run.build_dataset_v1 prepare      # extract archives, convert labels, tag negatives
python -m run.build_dataset_v1 calibrate    # near-duplicate distance report, to set dhash_max_distance
python -m run.build_dataset_v1 build       # dry run: writes lists and reports, no test lock
python -m run.build_dataset_v1 freeze      # same build, then writes test_lock.json (once)
"""

import argparse
import json

from config import DATASET_V1
from ml import dataset_v1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "calibrate", "build", "freeze"])
    p.add_argument("--frame-pattern", help="regex with a 'frame' group that links tiles of one UAV-PDD frame")
    a = p.parse_args()
    if a.stage == "prepare":
        result = dataset_v1.prepare(DATASET_V1)
    elif a.stage == "calibrate":
        result = dataset_v1.calibrate(DATASET_V1)
    else:
        result = dataset_v1.build(
            DATASET_V1, a.frame_pattern or DATASET_V1.uavpdd_frame_pattern, freeze=a.stage == "freeze"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
