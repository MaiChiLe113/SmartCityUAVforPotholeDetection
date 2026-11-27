"""Make the approved challenger the new champion (run by the workflow after human approval).

python -m run.promote --weights PATH [--decision artifacts/decision.json]

Copies the weights to artifacts/models/champion.pt and updates artifacts/metadata: model_version.txt,
batches.json (the batches the champion was trained on) and champion_history.json. The workflow then runs
`dvc add` / `dvc push` for the weights and opens a pull request with the metadata.
"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from config import RETRAIN


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True, help="challenger best.pt from the retrain job")
    p.add_argument("--decision", type=Path, default=RETRAIN.artifacts_dir / "decision.json")
    a = p.parse_args()
    decision = json.loads(a.decision.read_text())
    if not decision.get("promote"):
        raise SystemExit("decision.json does not recommend promotion; refusing to promote")

    RETRAIN.champion.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a.weights, RETRAIN.champion)
    meta = RETRAIN.metadata_dir
    meta.mkdir(parents=True, exist_ok=True)
    version = decision["challenger_version"]
    (meta / "model_version.txt").write_text(version + "\n")

    batches_path = meta / "batches.json"
    trained = json.loads(batches_path.read_text()) if batches_path.exists() else {}
    trained.update(decision.get("batches", {}))
    batches_path.write_text(json.dumps(trained, indent=2, sort_keys=True))

    history_path = meta / "champion_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(
        {
            "version": version,
            "replaced": decision["champion_version"],
            "promoted_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": decision.get("run_id"),
            "batches": sorted(decision.get("batches", {})),
            "reasons": decision["reasons"],
        }
    )
    history_path.write_text(json.dumps(history, indent=2))
    print(f"promoted {version} (was {decision['champion_version']})")


if __name__ == "__main__":
    main()
