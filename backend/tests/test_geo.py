import pytest

from ml.geo import Pose, box_to_latlon


@pytest.mark.xfail(raises=NotImplementedError, reason="SCRUM-34 (Hai)")
def test_centre_box_maps_to_drone_position():
    pose = Pose(lat=10.0, lon=106.0, altitude_m=30.0, heading_deg=0.0)
    lat, lon, _ = box_to_latlon((950, 530, 970, 550), (1920, 1080), pose)
    assert lat == pytest.approx(10.0, abs=1e-5)
    assert lon == pytest.approx(106.0, abs=1e-5)
