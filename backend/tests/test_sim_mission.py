"""Mission simulation checks using virtual time and local recorded fixtures."""

import math
from dataclasses import replace

import pytest

from app.models import MissionState
from app.services.detector import DetectionResult
from app.services.flight_planner import FlightPlanner
from config import UAV
from sim.camera import VirtualCamera
from sim.mock_detector import FrameResult, MockDetector
from sim.uav import SimulatedUAV


class VirtualTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0
        self.now += seconds


def local_distance(a, b):
    lat1, lat2 = math.radians(a.lat), math.radians(b.lat)
    dlat = lat2 - lat1
    dlon = math.radians(b.lon - a.lon)
    haversine = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    horizontal = 2 * 6_371_000 * math.asin(math.sqrt(haversine))
    return math.hypot(horizontal, b.altitude_m - a.altitude_m)


def make_sim(tmp_path, scenarios=("positive", "empty", "positive"), delay=2.5, sink=None):
    cfg = replace(
        UAV, home_lat=10, home_lon=106, default_altitude_m=10, survey_radius_m=5, speed_mps=5, telemetry_interval_s=1
    )
    route = FlightPlanner(cfg).propose(10.0001, 106.0001).route
    paths = [tmp_path / f"{i}.jpg" for i in range(len(scenarios))]
    for path in paths:
        path.write_bytes(b"recorded fixture")
    camera = VirtualCamera(cfg, image_paths=paths)
    detector = MockDetector({p.name: s for p, s in zip(paths, scenarios, strict=True)}, delay_s=delay)
    clock = VirtualTime()
    sim = SimulatedUAV(
        route, cfg=cfg, camera=camera, processor=detector, clock=clock.clock, sleep=clock.sleep, telemetry_sink=sink
    )
    return sim, route


def test_complete_mission_movement_results_and_hold(tmp_path):
    observed = []
    sim, route = make_sim(tmp_path, sink=observed.append)
    results = sim.run()
    assert sim.state == MissionState.done
    assert sim.error is None
    assert [r.frame.image_path.name for r in results] == ["0.jpg", "1.jpg", "2.jpg"]
    assert [len(r.detections) for r in results] == [1, 0, 1]
    assert all(r.source == "recorded" and r.model_version == "mock-v1" and r.error is None for r in results)
    assert results[0].detections[0].frame_id == results[0].frame.frame_id
    assert results[0].detections[0].model_version == "mock-v1"
    assert [(r.frame.pose.lat, r.frame.pose.lon, r.frame.pose.altitude_m) for r in results] == [
        (p["lat"], p["lon"], p["alt"]) for p in route[3:6]
    ]
    assert (sim.pose.lat, sim.pose.lon, sim.pose.altitude_m) == (route[0]["lat"], route[0]["lon"], 0)
    assert {e.state for e in observed} == {
        MissionState.pending,
        MissionState.takeoff,
        MissionState.surveying,
        MissionState.landing,
        MissionState.done,
    }
    assert {e.phase for e in observed} >= {"outbound", "survey", "processing", "return", "landing"}
    assert all(
        local_distance(a.pose, b.pose) <= sim.cfg.speed_mps * (b.elapsed_s - a.elapsed_s) + 0.002
        for a, b in zip(observed, observed[1:], strict=False)
    )
    takeoff = [e for e in observed if e.phase == "takeoff"]
    assert [e.pose.altitude_m for e in takeoff] == [0, 5, 10]
    processing = [e for e in observed if e.phase == "processing"]
    assert len(processing) == 12  # initial event plus three ticks for each 2.5-second job
    for group in (processing[:4], processing[4:8], processing[8:]):
        assert len({(e.pose.lat, e.pose.lon, e.pose.altitude_m) for e in group}) == 1
        assert group[-1].elapsed_s - group[0].elapsed_s == pytest.approx(3)
    observed[0].pose.lat = -1
    assert sim.telemetry[0].pose.lat == 10
    sim.pose.lat = -2
    assert results[0].frame.pose.lat == route[3]["lat"]
    with pytest.raises(RuntimeError, match="single-use"):
        sim.run()


def test_mock_error_preserves_partial_results_and_fails_before_return(tmp_path):
    sim, route = make_sim(tmp_path, ("positive", "empty", "error"), delay=0)
    results = sim.run()
    assert sim.state == MissionState.failed
    assert len(results) == 3 and [r.error for r in results] == [None, None, "mock processing error"]
    assert len(results[0].detections) == 1 and not results[1].detections
    assert (sim.pose.lat, sim.pose.lon) == (route[5]["lat"], route[5]["lon"])
    assert not any(e.phase == "return" for e in sim.telemetry)


def test_missing_camera_frame_preserves_partial_results(tmp_path):
    sim, _ = make_sim(tmp_path, ("positive", "empty"), delay=0)
    assert len(sim.run()) == 2
    assert sim.state == MissionState.failed
    assert "exhausted" in sim.error


def test_unmapped_mock_frame_is_recorded_as_failure(tmp_path):
    sim, _ = make_sim(tmp_path, delay=0)
    sim.processor.scenarios.pop("1.jpg")
    results = sim.run()
    assert len(results) == 2 and "no mock scenario" in results[1].error
    assert sim.state == MissionState.failed


def test_invalid_route_and_missing_processor_fail_before_flight(tmp_path):
    sim, route = make_sim(tmp_path)
    sim.route[-1]["lat"] += 0.001
    assert sim.run() == []
    assert sim.state == MissionState.failed and "at home" in sim.error
    sim2 = SimulatedUAV(route, camera=sim.camera)
    assert sim2.run() == []
    assert sim2.state == MissionState.failed and "processor" in sim2.error


def test_mock_requires_known_scenarios_and_nonnegative_delay():
    with pytest.raises(ValueError):
        MockDetector({"a.jpg": "unknown"})
    with pytest.raises(ValueError):
        MockDetector({}, delay_s=-1)


def test_no_more_than_one_processing_job(tmp_path):
    sim, _ = make_sim(tmp_path)
    inner = sim.processor

    class TrackingProcessor:
        active = 0
        maximum = 0

        def start(self, frame, now):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            job = inner.start(frame, now)
            owner = self

            class TrackingJob:
                def poll(self, now):
                    result = job.poll(now)
                    if result is not None:
                        owner.active -= 1
                    return result

            return TrackingJob()

    tracker = TrackingProcessor()
    sim.processor = tracker
    sim.run()
    assert sim.state == MissionState.done
    assert tracker.maximum == 1 and tracker.active == 0


def test_never_ready_job_times_out_with_partial_results(tmp_path):
    sim, _ = make_sim(tmp_path, delay=0)
    original = sim.processor

    class NeverReady:
        def poll(self, now):
            return None

    class Mixed:
        def start(self, frame, now):
            return original.start(frame, now) if frame.frame_id == "frame-000001" else NeverReady()

    sim.cfg = replace(sim.cfg, processing_timeout_s=2.5)
    sim.processor = Mixed()
    results = sim.run()
    assert sim.state == MissionState.failed
    assert len(results) == 2 and results[0].error is None
    assert "timed out" in results[1].error
    processing = [e for e in sim.telemetry if e.phase == "processing"]
    assert len(processing) == 5


@pytest.mark.parametrize(
    "kind",
    [
        "none_detections",
        "wrong_frame",
        "nan",
        "wrong_path",
        "bad_box",
        "bad_class",
        "bad_version",
        "empty_error",
        "mutated_pose",
    ],
)
def test_malformed_processor_result_fails_and_keeps_canonical_frame(tmp_path, kind):
    sim, route = make_sim(tmp_path, delay=0)

    class Processor:
        def start(self, frame, now):
            class Job:
                def poll(self, now):
                    d = DetectionResult("D40", 0.5, [1, 2, 3, 4], frame.frame_id, "mock-v1")
                    result = FrameResult(frame, [d])
                    if kind == "none_detections":
                        result.detections = None
                    elif kind == "wrong_frame":
                        d.frame_id = "other"
                    elif kind == "nan":
                        d.confidence = math.nan
                    elif kind == "wrong_path":
                        frame.image_path = frame.image_path.with_name("other.jpg")
                    elif kind == "bad_box":
                        d.box_xyxy = [3, 2, 1, 4]
                    elif kind == "bad_class":
                        d.cls = "fake"
                    elif kind == "bad_version":
                        d.model_version = ""
                    elif kind == "empty_error":
                        result.error, result.detections = "", []
                    elif kind == "mutated_pose":
                        frame.pose.lat = -10
                    return result

            return Job()

    sim.processor = Processor()
    results = sim.run()
    assert sim.state == MissionState.failed
    assert len(results) == 1 and results[0].error is not None
    assert results[0].frame.image_path.name == "0.jpg"
    assert results[0].frame.pose.lat == route[3]["lat"]


def test_returned_result_is_detached_from_processor_objects(tmp_path):
    sim, _ = make_sim(tmp_path, delay=0)
    saved = []

    class Processor:
        def start(self, frame, now):
            class Job:
                def poll(self, now):
                    d = DetectionResult("D40", 0.5, [1, 2, 3, 4], frame.frame_id, "mock-v1")
                    result = FrameResult(frame, [d])
                    saved.append(result)
                    return result

            return Job()

    sim.processor = Processor()
    results = sim.run()
    assert sim.state == MissionState.done
    saved[0].frame.pose.lat = -30
    saved[0].detections[0].box_xyxy[0] = 99
    assert results[0].frame.pose.lat != -30
    assert results[0].detections[0].box_xyxy[0] == 1


@pytest.mark.parametrize("clock", [lambda: math.nan, lambda: (_ for _ in ()).throw(RuntimeError("clock failed"))])
def test_invalid_starting_clock_fails_cleanly(tmp_path, clock):
    sim, _ = make_sim(tmp_path)
    sim.clock = clock
    assert sim.run() == []
    assert sim.state == MissionState.failed and sim.error


def test_clock_regression_during_processing_fails(tmp_path):
    sim, _ = make_sim(tmp_path)
    original = sim.clock

    def clock():
        if sim.phase == "processing":
            return -1
        return original()

    sim.clock = clock
    results = sim.run()
    assert sim.state == MissionState.failed
    assert "backwards" in sim.error
    assert len(results) == 1 and results[0].error is not None


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf])
def test_invalid_processing_timeout_rejected(tmp_path, timeout):
    sim, _ = make_sim(tmp_path)
    sim.cfg = replace(sim.cfg, processing_timeout_s=timeout)
    sim.run()
    assert sim.state == MissionState.failed and "timeout" in sim.error


def test_stalled_clock_fails_instead_of_looping(tmp_path):
    sim, _ = make_sim(tmp_path)
    sim.clock = lambda: 0
    sim.sleep = lambda seconds: None
    sim.run()
    assert sim.state == MissionState.failed
    assert "advance" in sim.error
