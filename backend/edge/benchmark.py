"""Edge benchmark and deployment checks (§3.4 Table 3.8). Owner: Hoan (SCRUM-35)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import EXPORT, ExportConfig


@dataclass
class BenchmarkResult:
    device: str
    median_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float
    peak_memory_mb: float
    max_temperature_c: float | None
    crashes: int


def benchmark(package_path: Path, frames_dir: Path, device: str, cfg: ExportConfig = EXPORT) -> BenchmarkResult:
    """Warm up cfg.warmup_runs, then time cfg.benchmark_runs over the same recorded frames."""
    raise NotImplementedError("TODO(SCRUM-35, Hoan): latency/memory benchmark")


def stability_test(package_path: Path, frames_dir: Path, cfg: ExportConfig = EXPORT) -> bool:
    """Run continuous detection for cfg.stability_minutes; pass if no crash."""
    raise NotImplementedError("TODO(SCRUM-35, Hoan): 30-minute stability test")
