"""Markdown summary of a retraining run for the GitHub Actions job summary.

python -m run.workflow_summary [--title TEXT] >> "$GITHUB_STEP_SUMMARY"
Reads whichever reports exist: pending.json, triggers.json, drift_report.json, ingest_report.json,
artifacts/decision.json and artifacts/metrics/evaluation_metrics.json.
"""

import argparse
import json
from pathlib import Path

from config import DATA, RETRAIN

REPORTS = RETRAIN.monitoring_dir / "reports"


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def render(title: str) -> str:
    out = [f"## {title}", ""]
    if pending := load(REPORTS / "pending.json"):
        names = ", ".join(pending["pending_batches"]) or "none"
        out += [f"**New data:** {pending['pending_images']} images in batches not yet trained on: {names}", ""]
    if triggers := load(REPORTS / "triggers.json"):
        out += [
            f"**Triggers:** {', '.join(triggers['triggers']) or 'none'}. "
            f"Retrain: **{'yes' if triggers['should_retrain'] else 'no'}** ({triggers['reason']})",
            "",
        ]
    if drift := load(REPORTS / "drift_report.json"):
        out += [
            f"**Drift:** {'ALERT' if drift['alert'] else 'no alert'} on {drift['images']} images "
            f"({len(drift['drifted_features'])} of {len(drift['tests'])} tests drifted)",
            "",
            "| Test | p-value | Result |",
            "|---|---|---|",
        ]
        for name, t in drift["tests"].items():
            p = "n/a" if t.get("p_value") is None else f"{t['p_value']:.4f}"
            out.append(f"| {name} | {p} | {'drift' if t['drifted'] else 'ok'} |")
        out.append("")
    if ingest := load(REPORTS / "ingest_report.json"):
        out += [
            f"**Ingest:** {ingest['accepted_train']} to train, {ingest['accepted_val']} to val, "
            f"{ingest['rejected']} rejected {ingest['rejected_by_reason'] or ''}",
            "",
        ]
    if metrics := load(RETRAIN.metrics_dir / "evaluation_metrics.json"):
        out += ["| Model | Split | mAP50-95 | mAP50 | " + " | ".join(DATA.classes) + " |"]
        out += ["|---|---|---|---|" + "---|" * len(DATA.classes)]
        for split in ("val", "test"):
            for model, subsets in metrics.get(split, {}).items():
                a = subsets["all"]
                per_class = " | ".join(f"{a['AP50-95'].get(c, 0):.3f}" for c in DATA.classes)
                out.append(f"| {model} | {split} | {a['mAP50-95']:.4f} | {a['mAP50']:.4f} | {per_class} |")
        out.append("")
    if decision := load(RETRAIN.artifacts_dir / "decision.json"):
        verdict = "PROMOTE (waiting for approval)" if decision["promote"] else "keep the champion"
        out += [f"**Decision:** {verdict}: {decision['champion_version']} -> {decision['challenger_version']}", ""]
        out += [f"- {r}" for r in decision["reasons"]] + [""]
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--title", default="Retraining run")
    print(render(p.parse_args().title))


if __name__ == "__main__":
    main()
