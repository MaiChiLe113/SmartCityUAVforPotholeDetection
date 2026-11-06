"""Tests for local route planning and recorded-image camera replay."""

import math
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.services.flight_planner import FlightPlanner, _distance_m
from config import UAV
from ml import build
from ml.geo import Pose
from sim.camera import VirtualCamera


def test_route_structure_and_survey_radius():
    cfg = replace(UAV, home_lat=10.0, home_lon=106.0, survey_radius_m=25)
    plan = FlightPlanner(cfg).propose(10.001, 106.001)
    assert plan.drone_id == cfg.drone_ids[0]
    assert [p["action"] for p in plan.route] == [
        "takeoff",
        "takeoff",
        "fly",
        "survey",
        "survey",
        "survey",
        "fly",
        "land",
    ]
    assert [(p["lat"], p["lon"], p["alt"]) for p in (plan.route[0], plan.route[-1])] == [
        (cfg.home_lat, cfg.home_lon, 0),
        (cfg.home_lat, cfg.home_lon, 0),
    ]
    assert all(p["alt"] == cfg.default_altitude_m for p in plan.route[1:-1])
    assert (plan.route[2]["lat"], plan.route[2]["lon"]) == (10.001, 106.001)
    assert all(
        _distance_m(10.001, 106.001, p["lat"], p["lon"]) == pytest.approx(25, abs=0.001) for p in plan.route[3:6]
    )
    assert len({(p["lat"], p["lon"]) for p in plan.route[3:6]}) == 3
    assert FlightPlanner(cfg).propose(10.001, 106.001, "change").route == plan.route


def test_route_crosses_antimeridian_at_high_latitude():
    cfg = replace(UAV, home_lat=84.0, home_lon=179.9999, survey_radius_m=25)
    plan = FlightPlanner(cfg).propose(84.0, -179.9999)
    assert any(p["lon"] < 0 for p in plan.route)
    assert all(-180 <= p["lon"] <= 180 for p in plan.route)
    assert all(
        _distance_m(84.0, -179.9999, p["lat"], p["lon"]) == pytest.approx(25, abs=0.001) for p in plan.route[3:6]
    )


@pytest.mark.parametrize("lat,lon", [(math.nan, 1), (91, 1), (0, math.inf), (0, 181), (85.1, 0)])
def test_invalid_target_rejected(lat, lon):
    with pytest.raises(ValueError):
        FlightPlanner().propose(lat, lon)


@pytest.mark.parametrize(
    "changes",
    [
        {"home_lat": math.nan},
        {"home_lon": 181},
        {"default_altitude_m": 0},
        {"speed_mps": -1},
        {"survey_radius_m": math.inf},
        {"max_target_distance_m": 0},
        {"max_target_distance_m": 10001},
        {"drone_ids": ()},
        {"survey_radius_m": 20, "max_target_distance_m": 10},
    ],
)
def test_invalid_config_rejected(changes):
    cfg = replace(UAV, **changes)
    with pytest.raises(ValueError):
        FlightPlanner(cfg).propose(UAV.home_lat, UAV.home_lon)


def test_target_outside_local_range_rejected():
    with pytest.raises(ValueError, match="maximum distance"):
        FlightPlanner().propose(UAV.home_lat + 1, UAV.home_lon)


def test_camera_folder_order_metadata_and_exhaustion(tmp_path):
    for name in ("b.jpg", "A.png", "ignored.txt"):
        (tmp_path / name).write_bytes(b"fixture")
    (tmp_path / "directory.jpg").mkdir()
    now = datetime(2024, 1, 1, tzinfo=UTC)
    pose = Pose(1, 2, 30, 90)
    camera = VirtualCamera(image_dir=tmp_path, clock=lambda: now)
    first, second = camera.capture(pose), camera.capture(pose)
    assert [first.image_path.name, second.image_path.name] == ["A.png", "b.jpg"]
    assert [first.frame_id, second.frame_id] == ["frame-000001", "frame-000002"]
    assert first.pose == pose
    pose.lat = 8
    assert first.pose.lat == 1
    assert second.pose.lat == 1
    assert first.ts == now
    assert first.fov_deg == UAV.camera_fov_deg
    with pytest.raises(EOFError, match="exhausted"):
        camera.capture(pose)


def test_camera_explicit_order_and_builder(tmp_path):
    paths = [tmp_path / "b.jpg", tmp_path / "a.jpg"]
    for path in paths:
        path.write_bytes(b"fixture")
    pose = Pose(1, 2, 30, 0)
    camera = VirtualCamera(image_paths=paths)
    assert [camera.capture(pose).image_path for _ in paths] == paths
    assert build.build_camera(replace(UAV, recorded_images_dir=tmp_path)).capture(pose).image_path == paths[1]


def test_camera_rejects_missing_empty_and_conflicting_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        VirtualCamera(image_dir=tmp_path / "missing")
    with pytest.raises(ValueError, match="no recorded"):
        VirtualCamera(image_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        VirtualCamera(image_paths=[tmp_path / "missing.jpg"])
    with pytest.raises(ValueError, match="either"):
        VirtualCamera(image_dir=tmp_path, image_paths=[])
    path = tmp_path / "gone.jpg"
    path.write_bytes(b"fixture")
    camera = VirtualCamera(image_paths=[path])
    path.unlink()
    with pytest.raises(FileNotFoundError):
        camera.capture(Pose(1, 2, 3, 4))
