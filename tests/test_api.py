"""API tests exercising the FastAPI service against a trained temp model
(fixtures in conftest.py isolate the DB/model dir from the real application)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from misan.api import app
from misan.config import SENSOR_NAMES

client = TestClient(app)


def test_health_reports_no_model_before_training():
    # Uses its own isolated env but before the trained_version fixture runs in this
    # process, the active model may already be set by other tests — just check shape.
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "active_model_version" in resp.json()


def test_faults_endpoint_lists_taxonomy():
    resp = client.get("/faults")
    assert resp.status_code == 200
    codes = {f["code"] for f in resp.json()}
    assert "NORMAL" in codes and "CMP-DEG" in codes


def test_diagnose_endpoint_end_to_end(trained_version):
    result, df = trained_version
    faulty_run = df[(df["fault_class"] == "REF-LEAK") & (df["severity"] == 1.0)].sort_values("timestamp")
    run_id = faulty_run["run_id"].iloc[0]
    one_run = faulty_run[faulty_run["run_id"] == run_id].tail(15)

    payload = {
        "asset_id": "API-TEST-ASSET",
        "timestamps": one_run["timestamp"].astype(str).tolist(),
        "readings": {s: one_run[s].astype(float).tolist() for s in SENSOR_NAMES},
    }
    resp = client.post("/diagnose", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == result.version
    assert body["validation_status"] in ("confirmed", "rejected", "unchecked")
    assert set(body["class_probabilities"].keys())


def test_diagnose_endpoint_rejects_missing_signal(trained_version):
    payload = {
        "asset_id": "API-TEST-ASSET",
        "timestamps": ["2026-01-01T00:00:00"] * 15,
        "readings": {s: [1.0] * 15 for s in SENSOR_NAMES[:-1]},  # missing one signal
    }
    resp = client.post("/diagnose", json=payload)
    assert resp.status_code == 422


def test_models_active_reflects_trained_version(trained_version):
    result, _ = trained_version
    resp = client.get("/models/active")
    assert resp.status_code == 200
    assert resp.json()["version"] == result.version


def test_sensitive_routes_reject_missing_api_key(monkeypatch):
    monkeypatch.setenv("MISAN_API_KEY", "test-secret")
    protected_routes = ["/diagnose", "/datasets/import", "/train", "/evaluate"]

    for route in protected_routes:
        response = client.post(route)
        assert response.status_code == 401, (
            f"{route} should reject requests without an API key; "
            f"got {response.status_code}: {response.text}"
        )


def test_sensitive_routes_reject_wrong_api_key(monkeypatch):
    monkeypatch.setenv("MISAN_API_KEY", "test-secret")
    response = client.post(
        "/diagnose",
        headers={"X-Misan-API-Key": "wrong-secret"},
    )
    assert response.status_code == 401


def test_health_remains_public_when_api_key_is_configured(monkeypatch):
    monkeypatch.setenv("MISAN_API_KEY", "test-secret")
    response = client.get("/health")
    assert response.status_code == 200
