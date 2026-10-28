"""Thin HTTP client the dashboard uses to talk to the Misan FastAPI service.
All dashboard data is real — this module never fabricates a response."""
from __future__ import annotations

import os

import requests

BASE_URL = os.environ.get("MISAN_API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 15
API_KEY = os.environ.get("MISAN_API_KEY", "")

def _headers() -> dict[str, str]:
    return {"X-Misan-API-Key": API_KEY} if API_KEY else {}


class ApiError(RuntimeError):
    def __init__(self, status_code: int, detail):
        super().__init__(f"API error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


def _get(path: str, **params):
    try:
        resp = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
    except requests.ConnectionError as exc:
        raise ApiError(503, "cannot reach the Misan API — is it running? (`python -m misan.cli serve`)") from exc
    if resp.status_code >= 400:
        raise ApiError(resp.status_code, _safe_json(resp))
    return resp.json()



def _post(path: str, json_body=None, files=None, params=None):
    try:
        resp = requests.post(
            f"{BASE_URL}{path}",
            json=json_body,
            files=files,
            params=params,
            headers=_headers(),
            timeout=300 if path == "/train" else 60,
        )
    except requests.Timeout as exc:
        raise ApiError(
            504,
            "The request timed out. The server may still be processing it. "
            "Do not retry training until you check /health and the model version."
        ) from exc
    except requests.ConnectionError as exc:
        raise ApiError(
            503,
            "Cannot reach the Misan API — is it running? "
            "(`python -m misan.cli serve`)"
        ) from exc

    if resp.status_code >= 400:
        raise ApiError(resp.status_code, _safe_json(resp))
    return resp.json()



def _safe_json(resp):
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text


def health() -> dict:
    return _get("/health")


def faults() -> list[dict]:
    return _get("/faults")


def recent_alerts(limit: int = 10) -> list[dict]:
    return _get("/alerts/recent", limit=limit)


def recent_diagnostics(limit: int = 10) -> list[dict]:
    return _get("/diagnostics/recent", limit=limit)


def active_model() -> dict | None:
    return _get("/models/active")


def evaluations(limit: int = 10) -> list[dict]:
    return _get("/models/evaluations", limit=limit)


def datasets() -> list[dict]:
    return _get("/datasets")


def import_dataset(filename: str, file_bytes: bytes) -> dict:
    files = {"file": (filename, file_bytes)}
    return _post("/datasets/import", files=files)


def diagnose(asset_id: str, timestamps: list[str], readings: dict[str, list[float]]) -> dict:
    return _post("/diagnose", json_body={"asset_id": asset_id, "timestamps": timestamps, "readings": readings})


def train(dataset_path: str) -> dict:
    return _post("/train", params={"dataset_path": dataset_path})


def evaluate(version: str | None = None) -> dict:
    params = {"version": version} if version else None
    return _post("/evaluate", params=params)
