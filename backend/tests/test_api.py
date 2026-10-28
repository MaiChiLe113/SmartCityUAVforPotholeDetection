from fastapi.testclient import TestClient

from app.main import create_app


def test_health():
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_unbuilt_endpoint_returns_501():
    with TestClient(create_app()) as client:
        assert client.get("/reports").status_code == 501
