"""Weekly drift report (SCRUM-36).  python -m run.drift_report --recent DIR"""

import argparse
from pathlib import Path

from config import PATHS
from ml.build import build_drift_monitor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recent", type=Path, required=True, help="folder of recent incoming images")
    a = p.parse_args()
    images = sorted(a.recent.glob("*.jpg")) + sorted(a.recent.glob("*.png"))
    print(build_drift_monitor().run(images, PATHS.reports_dir))


if __name__ == "__main__":
    main()
