"""Isolated synthetic/recorded fixture demonstration; never dispatches a live mission."""

from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PureWindowsPath

from PIL import Image, ImageDraw, UnidentifiedImageError

from app.models import MissionState
from app.services.flight_planner import FlightPlanner
from config import PATHS, DemoConfig, UavConfig
from sim.camera import VirtualCamera
from sim.mock_detector import MockDetector
from sim.uav import SimulatedUAV


class VirtualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def timestamp(self) -> datetime:
        return datetime(2024, 1, 1, tzinfo=UTC) + timedelta(seconds=self.value)


@dataclass(frozen=True)
class DemoRun:
    output_dir: Path
    state: MissionState
    error: str | None


def _dump(path: Path, value: object) -> None:
    def convert(obj: object) -> str:
        if isinstance(obj, (Path, datetime)):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj).__name__}")

    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n", encoding="utf-8")


def _check_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.width < 90 or image.height < 90:
                raise ValueError(f"image must be at least 90 x 90 pixels for preset boxes: {path}")
    except (UnidentifiedImageError, OSError, SyntaxError, EOFError, Image.DecompressionBombError) as exc:
        raise ValueError(f"image is not readable: {path}") from exc


def _source_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"development image folder does not exist: {folder}")
    if folder.resolve().is_relative_to(PATHS.splits_dir.resolve()):
        raise ValueError("frozen split directory cannot be used for the demo")
    paths = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda p: (p.name.casefold(), p.name),
    )
    if len(paths) != 3:
        raise ValueError("development image folder must contain exactly three PNG/JPEG images")
    for path in paths:
        if path.resolve().is_relative_to(PATHS.splits_dir.resolve()):
            raise ValueError("frozen split images cannot be used for the demo")
        _check_image(path)
    return paths


def _create_fixtures(folder: Path, cfg: DemoConfig) -> list[Path]:
    folder.mkdir()
    paths = []
    for index, name in enumerate(cfg.fixture_names):
        path = folder / name
        image = Image.new("RGB", cfg.image_size, (215, 220, 220))
        draw = ImageDraw.Draw(image)
        draw.rectangle((8, 40, 120, 88), fill=(100 + index * 10, 100, 100))
        draw.line((15, 64, 110, 64), fill=(255, 220, 50), width=3)
        image.save(path, format="PNG")
        _check_image(path)
        paths.append(path)
    return paths


def _validate_fixture_names(cfg: DemoConfig) -> None:
    names = cfg.fixture_names
    if len(names) != 3 or len({name.casefold() for name in names}) != 3:
        raise ValueError("fixture names must contain exactly three unique names")
    for name in names:
        windows_path = PureWindowsPath(name)
        if (
            not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or any(char in name for char in ':*?"<>|')
            or any(ord(char) < 32 for char in name)
            or windows_path.is_reserved()
            or name.endswith((".", " "))
            or Path(name).name != name
            or windows_path.name != name
            or windows_path.is_absolute()
            or Path(name).suffix.lower() != ".png"
        ):
            raise ValueError("fixture names must be safe PNG basenames")
    # Verify containment before creating the run or fixture folders.
    parent = (cfg.output_dir.resolve() / "synthetic_fixtures").resolve()
    if any(not (parent / name).resolve().is_relative_to(parent) for name in names):
        raise ValueError("fixture names must stay inside the fixture folder")


def run_demo(cfg: DemoConfig, uav: UavConfig, *, image_dir: Path | None = None) -> DemoRun:
    """Create a new isolated output folder and run three local recorded-image fixtures."""
    _validate_fixture_names(cfg)
    if cfg.scenario not in {"positive", "all-empty", "error", "timeout"}:
        raise ValueError(f"unknown demo scenario: {cfg.scenario}")
    if not math.isfinite(cfg.processing_delay_s) or cfg.processing_delay_s < 0:
        raise ValueError("processing delay must be finite and non-negative")
    if not math.isfinite(uav.processing_timeout_s) or uav.processing_timeout_s <= 0:
        raise ValueError("processing timeout must be finite and positive")
    if cfg.processing_delay_s >= uav.processing_timeout_s:
        raise ValueError("processing delay must be less than the timeout")
    root = cfg.output_dir.resolve()
    images = None
    if image_dir is not None:
        source = image_dir.resolve()
        if root.is_relative_to(source) or source.is_relative_to(root):
            raise ValueError("output and input folders must be separate and not contain each other")
        images = _source_images(source)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="demo-", dir=root))
    if images is None:
        images = _create_fixtures(output / "synthetic_fixtures", cfg)
    source_label = "recorded_development_images" if image_dir else "synthetic_test_fixtures"
    scenarios = {
        "positive": ["positive", "empty", "empty"],
        "all-empty": ["empty", "empty", "empty"],
        "error": ["empty", "error", "empty"],
        "timeout": ["empty", "empty", "empty"],
    }[cfg.scenario]
    manifest = [
        {
            "order": i + 1,
            "path": str(path),
            "fixture_outcome": scenario,
            "expected": "timeout" if cfg.scenario == "timeout" and i == 1 else scenario,
        }
        for i, (path, scenario) in enumerate(zip(images, scenarios, strict=True))
    ]
    _dump(output / "manifest.json", {"input_source": source_label, "detector_mode": "mock", "frames": manifest})
    plan = FlightPlanner(uav).propose(uav.home_lat + cfg.target_offset_deg, uav.home_lon + cfg.target_offset_deg)
    _dump(output / "plan.json", asdict(plan))
    clock = VirtualClock()
    camera = VirtualCamera(uav, image_paths=images, clock=clock.timestamp)
    detectors = {
        path.name: MockDetector(
            {path.name: scenario},
            delay_s=(uav.processing_timeout_s + 1 if cfg.scenario == "timeout" and i == 1 else cfg.processing_delay_s),
        )
        for i, (path, scenario) in enumerate(zip(images, scenarios, strict=True))
    }

    class DemoProcessor:
        def start(self, frame, now):
            return detectors[frame.image_path.name].start(frame, now)

    sim = SimulatedUAV(
        plan.route, cfg=uav, camera=camera, processor=DemoProcessor(), clock=clock.now, sleep=clock.sleep
    )
    wall_start = time.perf_counter()
    results = sim.run()
    wall_elapsed = time.perf_counter() - wall_start
    _dump(output / "telemetry.json", [asdict(event) for event in sim.telemetry])
    _dump(output / "results.json", [asdict(result) for result in results])
    failed = sum(result.error is not None for result in results)
    summary = {
        "state": sim.state,
        "error": sim.error,
        "scenario": cfg.scenario,
        "input_source": source_label,
        "detector_mode": "mock",
        "model_version": "mock-v1",
        "disclaimer": "Synthetic/mock preset detections; confidence and boxes are not measured model accuracy.",
        "expected_frames": 3,
        "submitted_frames": len(results),
        "successful_frames": len(results) - failed,
        "failed_frames": failed,
        "detections": sum(len(r.detections) for r in results),
        "logical_elapsed_s": clock.value,
        "wall_elapsed_s": wall_elapsed,
        "processing_delay_s": cfg.processing_delay_s,
        "processing_timeout_s": uav.processing_timeout_s,
        "final_pose": asdict(sim.pose) if sim.pose else None,
        "home_ground_reached": bool(
            sim.state == MissionState.done
            and sim.pose
            and sim.pose.lat == uav.home_lat
            and sim.pose.lon == uav.home_lon
            and sim.pose.altitude_m == 0
        ),
    }
    _dump(output / "summary.json", summary)
    return DemoRun(output, sim.state, sim.error)
