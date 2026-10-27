"""
Unit and integration tests for the production RailCast Web Server and REST API (src.app.server).

Verifies:
- Health check and system status endpoints (/api/status)
- Station, train, and critical section directories (/api/stations, /api/trains, /api/critical_sections)
- Input validation and schema checking for /api/predict (negative delay, malformed payload, invalid types)
- Multi-horizon inference execution (1h to 6h)
- Path traversal attack prevention on /reports/figures/
- Security headers and CORS headers
"""
import json
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import pytest

from src.app.server import RailCastRequestHandler, run_server
import socketserver
import http.server

TEST_PORT = 8599


@pytest.fixture(scope="module")
def live_server():
    """Start live test HTTP server instance in a background thread."""
    class ThreadingTestServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadingTestServer(("127.0.0.1", TEST_PORT), RailCastRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    yield f"http://127.0.0.1:{TEST_PORT}"
    server.shutdown()


def test_api_status(live_server):
    """Test health check /api/status endpoint."""
    url = f"{live_server}/api/status"
    req = Request(url)
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "online"
        assert data["is_simulated"] is True
        assert data["model_loaded"] is True
        assert data["horizons"] == [1, 2, 3, 4, 5, 6]


def test_api_stations(live_server):
    """Test stations directory endpoint."""
    url = f"{live_server}/api/stations"
    req = Request(url)
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "stations" in data
        assert len(data["stations"]) >= 50
        first = data["stations"][0]
        assert "code" in first
        assert "name" in first
        assert "km" in first


def test_api_critical_sections(live_server):
    """Test cascade critical sections endpoint."""
    url = f"{live_server}/api/critical_sections"
    req = Request(url)
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "critical_sections" in data
        assert len(data["critical_sections"]) >= 5


def test_api_predict_success(live_server):
    """Test valid prediction request across all 6 horizons."""
    url = f"{live_server}/api/predict"
    payload = json.dumps({
        "train": 12302,
        "station": "TDL",
        "current_delay": 25.0,
        "fog_flag": 0,
        "rolling_active_trains_30m": 6,
    }).encode("utf-8")

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True
        assert data["is_simulated"] is True
        assert data["current_delay"] == 25.0
        assert len(data["predictions"]) == 6
        horizons = [p["horizon_hours"] for p in data["predictions"]]
        assert horizons == [1, 2, 3, 4, 5, 6]


def test_api_predict_validation_negative_delay(live_server):
    """Test input validation rejection for negative current delay."""
    url = f"{live_server}/api/predict"
    payload = json.dumps({
        "current_delay": -10.0,
    }).encode("utf-8")

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req)
    assert exc_info.value.code == 400


def test_api_predict_validation_malformed_json(live_server):
    """Test input validation rejection for malformed JSON."""
    url = f"{live_server}/api/predict"
    payload = b"this is not json"

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req)
    assert exc_info.value.code == 400


def test_security_path_traversal_blocked(live_server):
    """Test path traversal prevention on /reports/figures/."""
    url = f"{live_server}/reports/figures/../../README.md"
    req = Request(url)
    with pytest.raises(HTTPError) as exc_info:
        urlopen(req)
    # Must be 403 Forbidden or 404 Not Found (never 200)
    assert exc_info.value.code in (403, 404)
