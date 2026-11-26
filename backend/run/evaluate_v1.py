"""Score a v1 model once on the frozen test set (FP32), plus false alarms and the HighRPD pothole test.

python -m run.evaluate_v1 --weights runs/train/yolo26n-v1/weights/best.pt
Reads tuned confidences from config.INFER.conf_thresholds. Writes runs/reports/v1/.
"""

import argparse
import json
from pathlib import Path

from config import DATA, DATASET_V1, INFER, PATHS
from ml import dataset_v1
from ml.evaluate import evaluate_val_subsets, false_alarms


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    a = p.parse_args()
    out = PATHS.reports_dir / "v1"
    out.mkdir(parents=True, exist_ok=True)

    digest = dataset_v1.verify_frozen_test(DATASET_V1)
    tags = dataset_v1.negative_tags(DATASET_V1)
    test_list = DATASET_V1.output_dir / "test.txt"
    negatives = {p.stem for p in dataset_v1.list_paths(test_list) if p.stem in tags}
    thresholds = dict(INFER.conf_thresholds)

    report = {
        "weights": str(a.weights),
        "test_lock_sha256": digest,
        "conf_thresholds": thresholds,
        "test": evaluate_val_subsets(
            a.weights, test_list, out / "test", DATA.classes,
            negatives=negatives, conf_thresholds=thresholds, split_name="test",
        ),
        "false_alarms_test": false_alarms(
            a.weights,
            [p for p in dataset_v1.list_paths(test_list) if p.stem in negatives],
            tags,
            thresholds,
            DATA.classes,
        ),
    }
    hr_yaml = dataset_v1.highrpd_subset(DATASET_V1)
    hr = evaluate_val_subsets(
        a.weights, hr_yaml.parent / "subset.txt", out / "highrpd", DATA.classes,
        conf_thresholds=thresholds, split_name="highrpd", partition=False,
    )
    report["highrpd_external_d40"] = {
        "images": hr["all"]["images"],
        "AP50": hr["all"]["AP50"].get("D40"),
        "AP50-95": hr["all"]["AP50-95"].get("D40"),
        "at_tuned_conf": hr["all"].get("at_tuned_conf", {}).get("D40"),
        "note": "Only D40 is labelled in HighRPD; other classes are not scored.",
    }
    assert dataset_v1.verify_frozen_test(DATASET_V1) == digest  # nothing touched the test files
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "test"} | {"test_all": report["test"]["all"]}, indent=2))


if __name__ == "__main__":
    main()
