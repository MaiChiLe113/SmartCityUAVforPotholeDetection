"""CLI acceptance tests for the offline local simulator demonstration."""

import json
import os
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from config import DEMO, UAV
from sim.demo import run_demo

BACKEND = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run(
        [sys.executable, "-m", "run.demo_mission", *map(str, args)],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def artifacts(root):
    runs = list(root.iterdir())
    assert len(runs) == 1 and runs[0].is_dir()
    return runs[0]


def read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def test_approval_is_required_without_creating_output(tmp_path):
    output = tmp_path / "output"
    result = invoke("--output-dir", output)
    assert result.returncode != 0 and "--approve-demo" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("scenario,detections", [("positive", 1), ("all-empty", 0)])
def test_successful_demo_artifacts(tmp_path, scenario, detections):
    root = tmp_path / "output"
    result = invoke("--approve-demo", "--output-dir", root, "--scenario", scenario)
    assert result.returncode == 0, result.stderr
    assert "LOCAL DEMO ONLY" in result.stdout and "mock-v1" in result.stdout
    folder = artifacts(root)
    summary, frames = read(folder, "summary.json"), read(folder, "results.json")
    manifest, route, telemetry = (
        read(folder, "manifest.json"),
        read(folder, "plan.json")["route"],
        read(folder, "telemetry.json"),
    )
    assert summary["state"] == "done" and summary["home_ground_reached"]
    assert summary["input_source"] == "synthetic_test_fixtures" and summary["detector_mode"] == "mock"
    assert summary["model_version"] == "mock-v1" and summary["detections"] == detections
    assert (
        summary["expected_frames"],
        summary["submitted_frames"],
        summary["successful_frames"],
        summary["failed_frames"],
    ) == (3, 3, 3, 0)
    assert summary["logical_elapsed_s"] > 0 and summary["wall_elapsed_s"] >= 0
    assert len(frames) == len(manifest["frames"]) == 3
    assert all(f["error"] is None for f in frames)
    assert [e["pose"]["altitude_m"] for e in telemetry if e["phase"] == "done"] == [0]
    assert route[0]["lat"] == summary["final_pose"]["lat"] and route[0]["lon"] == summary["final_pose"]["lon"]
    for entry in manifest["frames"]:
        with Image.open(entry["path"]) as image:
            image.load()
    processing = [e for e in telemetry if e["phase"] == "processing"]
    assert len(processing) >= 6 and processing[0]["pose"] == processing[1]["pose"]


@pytest.mark.parametrize("scenario,message", [("error", "mock processing error"), ("timeout", "timed out")])
def test_failure_retains_partial_artifacts(tmp_path, scenario, message):
    root = tmp_path / "output"
    result = invoke("--approve-demo", "--output-dir", root, "--scenario", scenario, "--timeout", "3")
    assert result.returncode != 0 and message in result.stderr
    folder = artifacts(root)
    summary, frames = read(folder, "summary.json"), read(folder, "results.json")
    assert summary["state"] == "failed" and message in summary["error"]
    assert (summary["submitted_frames"], summary["successful_frames"], summary["failed_frames"]) == (2, 1, 1)
    assert frames[0]["error"] is None and frames[1]["error"] is not None
    assert not summary["home_ground_reached"] and read(folder, "telemetry.json")[-1]["state"] == "failed"


def test_development_images_are_not_mutated_and_output_is_unique(tmp_path):
    images, output = tmp_path / "images", tmp_path / "output"
    images.mkdir()
    for name in ("c.png", "a.png", "b.png"):
        Image.new("RGB", (100, 100), "white").save(images / name)
    before = {p.name: p.read_bytes() for p in images.iterdir()}
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    for _ in range(2):
        result = invoke("--approve-demo", "--output-dir", output, "--images", images)
        assert result.returncode == 0, result.stderr
    assert (output / "keep.txt").read_text() == "keep"
    assert {p.name: p.read_bytes() for p in images.iterdir()} == before
    runs = [p for p in output.iterdir() if p.is_dir()]
    assert len(runs) == 2
    manifest = read(runs[0], "manifest.json")
    assert manifest["input_source"] == "recorded_development_images"
    assert [Path(f["path"]).name for f in manifest["frames"]] == ["a.png", "b.png", "c.png"]


def test_invalid_inputs_and_collision_do_not_write(tmp_path):
    output = tmp_path / "output"
    missing = invoke("--approve-demo", "--output-dir", output, "--images", tmp_path / "missing")
    assert missing.returncode != 0 and "does not exist" in missing.stderr
    images = tmp_path / "images"
    images.mkdir()
    for n in range(3):
        (images / f"{n}.png").write_bytes(b"corrupt")
    corrupt = invoke("--approve-demo", "--output-dir", output, "--images", images)
    assert corrupt.returncode != 0 and "not readable" in corrupt.stderr
    collision = invoke("--approve-demo", "--output-dir", images / "outputs", "--images", images)
    assert collision.returncode != 0 and "separate" in collision.stderr
    invalid = invoke("--approve-demo", "--output-dir", output, "--scenario", "nonsense")
    assert invalid.returncode != 0
    assert not output.exists() and not (images / "outputs").exists()


def test_demo_imports_no_heavy_ml_or_network_client():
    script = (
        "import sys; import run.demo_mission, sim.demo; "
        "assert not any(x in sys.modules for x in "
        "('torch','ultralytics','mlflow','evidently','requests','httpx'))"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=BACKEND, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_redirected_legacy_encoding_handles_unicode_output_and_errors(tmp_path):
    env = dict(os.environ, PYTHONIOENCODING="cp1252:strict")
    output = tmp_path / "đường-dữ-liệu"
    command = [sys.executable, "-m", "run.demo_mission", "--approve-demo", "--output-dir", str(output)]
    success = subprocess.run(command, cwd=BACKEND, capture_output=True, env=env, timeout=30, check=False)
    assert success.returncode == 0, success.stderr
    assert b"Artifacts:" in success.stdout
    assert len(list(output.iterdir())) == 1
    missing = tmp_path / "ảnh-không-có"
    failure = subprocess.run(
        command + ["--images", str(missing)], cwd=BACKEND, capture_output=True, env=env, timeout=30, check=False
    )
    assert failure.returncode == 2
    assert b"does not exist" in failure.stderr and b"Traceback" not in failure.stderr


@pytest.mark.parametrize(
    "names",
    [
        ("same.png", "same.png", "other.png"),
        ("A.png", "a.PNG", "other.png"),
        ("../sentinel.png", "a.png", "b.png"),
        ("..\\sentinel.png", "a.png", "b.png"),
        ("C:\\sentinel.png", "a.png", "b.png"),
        ("a.jpg", "b.png", "c.png"),
        ("CON.png", "b.png", "c.png"),
        ("a.png", "b.png"),
    ],
)
def test_unsafe_fixture_names_refused_before_output(tmp_path, names):
    output = tmp_path / "out"
    sentinel = tmp_path / "sentinel.png"
    sentinel.write_bytes(b"do not change")
    with pytest.raises(ValueError, match="fixture names"):
        run_demo(replace(DEMO, output_dir=output, fixture_names=names), UAV)
    assert sentinel.read_bytes() == b"do not change"
    assert not output.exists()


def test_timeout_scenario_rejects_delay_that_times_out_first_frame(tmp_path):
    output = tmp_path / "output"
    result = invoke("--approve-demo", "--output-dir", output, "--scenario", "timeout", "--timeout", "1")
    assert result.returncode == 2 and "delay must be less" in result.stderr
    assert not output.exists()


def test_png_with_bad_idat_checksum_is_reported_cleanly(tmp_path):
    output, images = tmp_path / "out", tmp_path / "images"
    images.mkdir()
    for index in range(3):
        Image.new("RGB", (100, 100), "white").save(images / f"{index}.png")
    path = images / "1.png"
    data = bytearray(path.read_bytes())
    index = data.index(b"IDAT")
    length = struct.unpack(">I", data[index - 4 : index])[0]
    data[index + 4 + length] ^= 0x01  # corrupt IDAT CRC, retaining valid PNG header
    path.write_bytes(data)
    result = invoke("--approve-demo", "--output-dir", output, "--images", images)
    assert result.returncode == 2 and "not readable" in result.stderr
    assert "Traceback" not in result.stderr and not output.exists()
