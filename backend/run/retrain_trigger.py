"""Decide whether this workflow run retrains.  python -m run.retrain_trigger [--force]

Reads monitoring/reports/pending.json (run.ingest_batches --check-only), the drift report and
artifacts/metadata/last_retrain.txt. Prints the triggers and writes should_retrain to GITHUB_OUTPUT.
"""

import argparse
import json
from datetime import UTC, datetime

from config import RETRAIN
from ml.retrain_trigger import active_triggers
from run.ingest_batches import github_output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="retrain even without new data")
    a = p.parse_args()
    reports = RETRAIN.monitoring_dir / "reports"
    pending = json.loads((reports / "pending.json").read_text()) if (reports / "pending.json").exists() else {}
    drift = json.loads((reports / "drift_report.json").read_text()) if (reports / "drift_report.json").exists() else {}
    stamp = RETRAIN.metadata_dir / "last_retrain.txt"
    last = datetime.fromisoformat(stamp.read_text().strip()) if stamp.exists() else None

    result = active_triggers(
        pending.get("pending_images", 0), bool(drift.get("alert")), last, datetime.now(UTC), force=a.force
    )
    github_output(should_retrain=str(result.should_retrain).lower(), triggers=",".join(result.triggers) or "none")
    out = {"triggers": result.triggers, "should_retrain": result.should_retrain, "reason": result.reason}
    (reports / "triggers.json").parent.mkdir(parents=True, exist_ok=True)
    (reports / "triggers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
