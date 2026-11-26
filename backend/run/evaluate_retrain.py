"""Compare the retrained challenger with the champion and decide on promotion.

python -m run.evaluate_retrain              # GPU: val decision, then the frozen test set for the record
python -m run.evaluate_retrain --mode cpu   # hosted demo: val sample only, never promotable

Writes artifacts/metrics/evaluation_metrics.json and artifacts/decision.json.
"""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from config import DATA, DATASET_V1, PATHS, RETRAIN, TRAIN, TrainConfig
from ml import dataset_v1
from ml.evaluate import evaluate_val_subsets
from ml.promotion import check_promotion
from ml.retrain import current_version, next_version
from run.ingest_batches import github_output

KEEP = ("images", "mAP50-95", "mAP50", "precision", "recall", "AP50-95", "AP50")


def slim(report: dict) -> dict:
    return {name: {k: v[k] for k in KEEP if k in v} for name, v in report.items()}


def score(weights: Path, val_list: Path, out: Path, split: str, negatives: set[str], cfg: TrainConfig) -> dict:
    return evaluate_val_subsets(weights, val_list, out, DATA.classes, cfg=cfg, negatives=negatives, split_name=split)


def v1_negatives() -> set[str]:
    """Stems of the clean and hard-negative images kept in v1 validation and test."""
    stems: set[str] = set()
    for name in ("negatives_val.txt", "negatives_test.txt"):
        path = DATASET_V1.output_dir / name
        if path.exists():
            stems |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return stems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gpu", "cpu"], default="gpu")
    a = p.parse_args()
    cpu = a.mode == "cpu"
    cfg = replace(TRAIN, device="cpu", batch=16, workers=2) if cpu else TRAIN
    run_name = f"{RETRAIN.run_name}-cpu" if cpu else RETRAIN.run_name
    challenger = PATHS.runs_dir / "train" / run_name / "weights" / "best.pt"
    val_list = (RETRAIN.cpu_dir if cpu else RETRAIN.retrain_dir) / "val.txt"
    out = PATHS.reports_dir / "retrain"
    negatives = v1_negatives()

    val = {
        "champion": score(RETRAIN.champion, val_list, out / "val_champion", "val", negatives, cfg),
        "challenger": score(challenger, val_list, out / "val_challenger", "val", negatives, cfg),
    }
    decision = check_promotion(val["champion"], val["challenger"])
    version = current_version(RETRAIN)
    metrics = {"mode": a.mode, "val": {k: slim(v) for k, v in val.items()}}

    if not cpu:  # the test set is scored after the decision, for the record only
        digest = dataset_v1.verify_frozen_test(DATASET_V1)
        test_list = DATASET_V1.output_dir / "test.txt"
        metrics["test"] = {
            "champion": slim(score(RETRAIN.champion, test_list, out / "test_champion", "test", negatives, cfg)),
            "challenger": slim(score(challenger, test_list, out / "test_challenger", "test", negatives, cfg)),
        }
        assert dataset_v1.verify_frozen_test(DATASET_V1) == digest
        metrics["test_lock_sha256"] = digest

    ingest_report = RETRAIN.monitoring_dir / "reports" / "ingest_report.json"
    batches = json.loads(ingest_report.read_text())["batches"] if ingest_report.exists() else {}
    result = {
        "mode": a.mode,
        "promotable": not cpu,
        "promote": decision.promote and not cpu,
        "reasons": decision.reasons + (["FAIL hosted CPU demo models are never promoted"] if cpu else []),
        "champion_version": version,
        "challenger_version": next_version(version),
        "challenger_weights": str(challenger.relative_to(PATHS.runs_dir.parent)),
        "batches": batches,
        "run_id": os.getenv("GITHUB_RUN_ID", "local"),
    }
    RETRAIN.metrics_dir.mkdir(parents=True, exist_ok=True)
    (RETRAIN.metrics_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))
    (RETRAIN.artifacts_dir / "decision.json").write_text(json.dumps(result, indent=2))
    github_output(promote=str(result["promote"]).lower())
    print(json.dumps(result | {"batches": list(batches)}, indent=2))


if __name__ == "__main__":
    main()
