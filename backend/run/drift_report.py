"""Drift and data-quality report for new batches.

python -m run.drift_report --build-reference          # once: features of a seeded v1 train sample
python -m run.drift_report [--recent data/new]        # compare new images with the reference

Writes monitoring/reports/drift_report.{json,html}, appends monitoring/logs/monitoring.log, saves
monitoring/alerts/drift_*.json on an alert, and copies the summary to artifacts/metrics/monitoring_metrics.json.
"""

import argparse
import json
from pathlib import Path

from config import DATASET_V1, RETRAIN
from ml import dataset_v1
from ml.build import build_drift_monitor
from ml.drift import log_and_alert
from ml.grouping import IMAGE_SUFFIXES
from run.ingest_batches import github_output

REFERENCE_DIR = RETRAIN.monitoring_dir / "reference"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--build-reference", action="store_true", help="build the reference from dataset v1 train")
    p.add_argument("--recent", type=Path, default=RETRAIN.new_data_dir, help="folder searched for new images")
    a = p.parse_args()
    monitor = build_drift_monitor()

    if a.build_reference:
        train = dataset_v1.list_paths(DATASET_V1.output_dir / "train.txt")
        print(f"reference written to {monitor.build_reference(train, REFERENCE_DIR)}")
        return

    images = sorted(
        f for f in a.recent.rglob("*") if f.suffix.lower() in IMAGE_SUFFIXES and f.parent.name == "images"
    )
    reports = RETRAIN.monitoring_dir / "reports"
    result = monitor.run(images, REFERENCE_DIR, reports)
    log_and_alert(result, reports, RETRAIN.monitoring_dir)
    summary = json.loads(result.json_report.read_text())
    RETRAIN.metrics_dir.mkdir(parents=True, exist_ok=True)
    (RETRAIN.metrics_dir / "monitoring_metrics.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "tests"}, indent=2)
    )
    github_output(drift_alert=str(result.alert).lower(), drifted=",".join(result.drifted_features) or "none")
    print(
        f"{len(images)} images, drifted: {result.drifted_features or 'none'} "
        f"({result.share_drifted:.0%}), alert={result.alert}"
    )


if __name__ == "__main__":
    main()
