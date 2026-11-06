"""Run an explicitly acknowledged local-only simulator demonstration."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

from config import DEMO, UAV
from ml.build import run_local_demo


def main() -> int:
    # Preserve the console's encoding while escaping only characters it cannot encode.
    # Redirected Windows streams may otherwise use a legacy code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description="Local recorded/synthetic fixtures and mock detection only; no live UAV."
    )
    parser.add_argument("--approve-demo", action="store_true", help="acknowledge this local-only demo")
    parser.add_argument("--scenario", choices=("positive", "all-empty", "error", "timeout"), default=DEMO.scenario)
    parser.add_argument("--output-dir", type=Path, default=DEMO.output_dir, help="parent for a new unique run folder")
    parser.add_argument("--images", type=Path, help="folder of exactly three development PNG/JPEG images")
    parser.add_argument("--delay", type=float, default=DEMO.processing_delay_s, help="simulated processing seconds")
    parser.add_argument(
        "--timeout", type=float, default=UAV.processing_timeout_s, help="processing deadline in seconds"
    )
    args = parser.parse_args()
    if not args.approve_demo:
        print(
            "Local demo acknowledgement required: pass --approve-demo. This does not approve a live mission.",
            file=sys.stderr,
        )
        return 2
    if not math.isfinite(args.delay) or not math.isfinite(args.timeout):
        parser.error("delay and timeout must be finite")
    cfg = replace(DEMO, scenario=args.scenario, output_dir=args.output_dir, processing_delay_s=args.delay)
    try:
        result = run_local_demo(cfg, replace(UAV, processing_timeout_s=args.timeout), image_dir=args.images)
    except (OSError, ValueError) as exc:
        print(f"Local demo could not start: {exc}", file=sys.stderr)
        return 2
    print(f"LOCAL DEMO ONLY | recorded/synthetic fixtures | mock-v1 | state: {result.state}")
    print(f"Artifacts: {result.output_dir}")
    if result.error is not None:
        print(f"Mission failed: {result.error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
