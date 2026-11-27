"""Ingest new batches into data/retrain.  python -m run.ingest_batches [--check-only]

--check-only lists batches the champion was not trained on and writes nothing else (used by the workflow
to decide whether to retrain). Without it, every batch is validated, checked for leaks into val/test and
split into train/val, and data/retrain/data.yaml is written.
"""

import argparse
import json
import os

from config import RETRAIN
from ml.ingest import batch_images, ingest, pending_batches, write_ingest_outputs, write_retrain_split


def github_output(**values) -> None:
    """Expose values to later workflow steps when running in GitHub Actions."""
    if path := os.getenv("GITHUB_OUTPUT"):
        with open(path, "a") as f:
            for key, value in values.items():
                f.write(f"{key}={value}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check-only", action="store_true", help="count pending batches and images only")
    a = p.parse_args()

    if a.check_only:
        pending = pending_batches(RETRAIN)
        images = sum(len(batch_images(b)) for b in pending)
        result = {"pending_batches": [b.name for b in pending], "pending_images": images}
        out = RETRAIN.monitoring_dir / "reports" / "pending.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        github_output(new_batches=len(pending), new_images=images)
        print(json.dumps(result, indent=2))
        return

    report = ingest(RETRAIN)
    yaml = write_retrain_split(report, RETRAIN)
    summary = write_ingest_outputs(report, RETRAIN)
    print(json.dumps({k: v for k, v in summary.items() if k != "rejected_images"}, indent=2))
    print(f"wrote {yaml}")


if __name__ == "__main__":
    main()
