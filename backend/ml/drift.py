"""Data drift and data-quality monitoring for new batches (§3.3).

The reference is a seeded sample of dataset v1 training images: per-image features (brightness, contrast,
blur, size) plus view and class counts. New images are compared with it:
- numeric features: two-sample Kolmogorov-Smirnov test each
- view and labelled classes: chi-square test of the two count tables
An alert is raised when more than MonitorConfig.drift_feature_share of the tests reject at ks_pvalue.
"""

from __future__ import annotations

import csv
import html
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from config import DATA, MONITOR, MonitorConfig

NUMERIC = ("brightness", "contrast", "blur", "width", "height", "aspect")


@dataclass
class DriftResult:
    drifted_features: list[str]
    share_drifted: float
    alert: bool
    html_report: Path
    json_report: Path


# --- features ----------------------------------------------------------------------------------
def image_features(image: Path) -> dict[str, float] | None:
    """Brightness (mean grey), contrast (grey std), blur (Laplacian variance), width, height, aspect.

    Returns None for an unreadable image.
    """
    import cv2

    bgr = cv2.imread(str(image))
    if bgr is None:
        return None
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = grey.shape
    return {
        "brightness": float(grey.mean()),
        "contrast": float(grey.std()),
        "blur": float(cv2.Laplacian(grey, cv2.CV_64F).var()),
        "width": float(w),
        "height": float(h),
        "aspect": float(w / h),
    }


def label_classes(label: Path) -> list[str]:
    if not label.exists():
        return []
    return [DATA.classes[int(line.split()[0])] for line in label.read_text().splitlines() if line.strip()]


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def view_of_name(stem: str) -> str:
    from ml.ingest import safe_view

    return safe_view(stem)


# --- statistics (pure, unit tested) -----------------------------------------------------------
def compare_numeric(ref: list[float], new: list[float], alpha: float) -> dict:
    from scipy.stats import ks_2samp

    if len(ref) < 2 or len(new) < 2:
        return {"test": "ks", "statistic": None, "p_value": None, "drifted": False, "note": "too few values"}
    result = ks_2samp(ref, new)
    return {
        "test": "ks",
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "drifted": bool(result.pvalue < alpha),
        "ref_mean": float(sum(ref) / len(ref)),
        "new_mean": float(sum(new) / len(new)),
    }


def compare_counts(ref: dict[str, int], new: dict[str, int], alpha: float) -> dict:
    """Chi-square test of two count tables over the union of categories."""
    from scipy.stats import chi2_contingency

    cats = sorted(c for c in set(ref) | set(new) if ref.get(c, 0) + new.get(c, 0) > 0)
    base = {
        "test": "chi2",
        "ref_counts": {c: ref.get(c, 0) for c in cats},
        "new_counts": {c: new.get(c, 0) for c in cats},
    }
    if len(cats) < 2 or sum(new.values()) == 0 or sum(ref.values()) == 0:
        return base | {"statistic": None, "p_value": None, "drifted": False, "note": "too few categories or counts"}
    table = [[ref.get(c, 0) for c in cats], [new.get(c, 0) for c in cats]]
    stat, p, _, _ = chi2_contingency(table)
    return base | {"statistic": float(stat), "p_value": float(p), "drifted": bool(p < alpha)}


def decide(tests: dict[str, dict], share: float) -> tuple[list[str], float, bool]:
    scored = {k: v for k, v in tests.items() if v.get("p_value") is not None}
    drifted = sorted(k for k, v in scored.items() if v["drifted"])
    ratio = len(drifted) / len(scored) if scored else 0.0
    return drifted, ratio, ratio > share


# --- reference and run -------------------------------------------------------------------------
def collect(images: list[Path]) -> dict:
    """Feature columns, view and class counts, and data-quality counts for a list of images."""
    columns: dict[str, list[float]] = {k: [] for k in NUMERIC}
    views: Counter = Counter()
    classes: Counter = Counter()
    unreadable, empty_labels, missing_labels = 0, 0, 0
    for path in images:
        feats = image_features(path)
        if feats is None:
            unreadable += 1
            continue
        for k in NUMERIC:
            columns[k].append(feats[k])
        views[view_of_name(path.stem)] += 1
        label = label_for(path)
        if not label.exists():
            missing_labels += 1
        found = label_classes(label)
        empty_labels += not found
        classes.update(found)
    return {
        "columns": columns,
        "view": dict(views),
        "class": dict(classes),
        "quality": {
            "images": len(images),
            "unreadable": unreadable,
            "missing_labels": missing_labels,
            "empty_labels": empty_labels,
            "boxes": sum(classes.values()),
        },
    }


class DriftMonitor:
    def __init__(self, cfg: MonitorConfig = MONITOR) -> None:
        self.cfg = cfg

    def build_reference(self, train_images: list[Path], out_dir: Path) -> Path:
        """Features for a seeded sample of training images, saved as features.csv and counts.json."""
        sample = sorted(train_images)
        random.Random(self.cfg.reference_seed).shuffle(sample)
        sample = sorted(sample[: self.cfg.reference_sample])
        data = collect(sample)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "features.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(NUMERIC)
            writer.writerows(zip(*(data["columns"][k] for k in NUMERIC), strict=True))
        counts = {"view": data["view"], "class": data["class"], "quality": data["quality"]}
        (out_dir / "counts.json").write_text(json.dumps(counts, indent=2))
        return out_dir

    @staticmethod
    def load_reference(ref_dir: Path) -> dict:
        with (ref_dir / "features.csv").open() as f:
            rows = list(csv.DictReader(f))
        counts = json.loads((ref_dir / "counts.json").read_text())
        return {"columns": {k: [float(r[k]) for r in rows] for k in NUMERIC}} | counts

    def run(self, recent_images: list[Path], ref_dir: Path, out_dir: Path) -> DriftResult:
        """Compare recent images with the reference and write the JSON and HTML reports."""
        ref = self.load_reference(ref_dir)
        new = collect(recent_images)
        alpha = self.cfg.ks_pvalue
        tests = {k: compare_numeric(ref["columns"][k], new["columns"][k], alpha) for k in self.cfg.drift_features}
        for cat in self.cfg.drift_categories:
            tests[cat] = compare_counts(ref.get(cat, {}), new[cat], alpha)
        drifted, share, alert = decide(tests, self.cfg.drift_feature_share)
        if not recent_images:
            alert = False
        report = {
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "images": len(recent_images),
            "reference_images": ref["quality"]["images"],
            "alpha": alpha,
            "alert_share": self.cfg.drift_feature_share,
            "drifted_features": drifted,
            "share_drifted": share,
            "alert": alert,
            "tests": tests,
            "data_quality": new["quality"],
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "drift_report.json"
        json_path.write_text(json.dumps(report, indent=2))
        html_path = out_dir / "drift_report.html"
        html_path.write_text(render_html(report))
        return DriftResult(drifted, share, alert, html_path, json_path)


def render_html(report: dict) -> str:
    """A plain, self-contained HTML table of the drift tests and data-quality counts."""
    rows = []
    for name, t in report["tests"].items():
        p = "n/a" if t.get("p_value") is None else f"{t['p_value']:.4f}"
        stat = "n/a" if t.get("statistic") is None else f"{t['statistic']:.3f}"
        if t["test"] == "ks":
            detail = f"ref mean {t.get('ref_mean', 0):.1f}, new mean {t.get('new_mean', 0):.1f}"
        else:
            detail = f"ref {t.get('ref_counts')}, new {t.get('new_counts')}"
        status = "DRIFT" if t["drifted"] else "ok"
        rows.append(
            f"<tr class='{status.lower()}'><td>{html.escape(name)}</td><td>{t['test']}</td><td>{stat}</td>"
            f"<td>{p}</td><td>{status}</td><td>{html.escape(str(t.get('note', detail)))}</td></tr>"
        )
    quality = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in report["data_quality"].items())
    verdict = "ALERT: drift detected" if report["alert"] else "No drift alert"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Drift report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1b1b1b; background: #fff; }}
table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
td, th {{ border: 1px solid #ccc; padding: .35rem .6rem; text-align: left; }}
tr.drift td {{ background: #fde2e1; }}
.verdict {{ font-weight: 600; font-size: 1.1rem; }}
</style></head><body>
<h1>Drift and data-quality report</h1>
<p class="verdict">{verdict}</p>
<p>{report['images']} new images compared with {report['reference_images']} reference training images.
Tests drifted: {len(report['drifted_features'])} ({report['share_drifted']:.0%}); alert above
{report['alert_share']:.0%}. Significance level {report['alpha']}. Created {report['created_utc']} UTC.</p>
<h2>Tests</h2>
<table><tr><th>Feature</th><th>Test</th><th>Statistic</th><th>p-value</th><th>Result</th><th>Detail</th></tr>
{''.join(rows)}</table>
<h2>Data quality</h2>
<table><tr><th>Check</th><th>Count</th></tr>{quality}</table>
</body></html>
"""


def log_and_alert(result: DriftResult, report_dir: Path, monitoring_dir: Path) -> None:
    """Append one line to monitoring.log and save an alert record when drift is detected."""
    report = json.loads(result.json_report.read_text())
    logs = monitoring_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    line = (
        f"{report['created_utc']} images={report['images']} drifted={','.join(result.drifted_features) or '-'} "
        f"share={result.share_drifted:.2f} alert={result.alert}\n"
    )
    with (logs / "monitoring.log").open("a") as f:
        f.write(line)
    if result.alert:
        alerts = monitoring_dir / "alerts"
        alerts.mkdir(parents=True, exist_ok=True)
        stamp = report["created_utc"].replace(":", "").replace("-", "")
        (alerts / f"drift_{stamp}.json").write_text(
            json.dumps({k: report[k] for k in ("created_utc", "images", "drifted_features", "share_drifted")}, indent=2)
        )
