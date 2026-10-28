
"""HTTP client for the MISAN dashboard."""
from __future__ import annotations

import os
from typing import Any

import requests

BASE_URL = os.environ.get(
    "MISAN_API_BASE_URL", "http://127.0.0.1:8000"
).rstrip("/")
TIMEOUT = 15
API_KEY = os.environ.get("MISAN_API_KEY", "")


def _headers() -> dict[str, str]:
    return {"X-Misan-API-Key": API_KEY} if API_KEY else {}


class ApiError(RuntimeError):
    """A user-displayable API failure with a useful recovery message."""

    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail
        super().__init__(self._format_detail(detail))

    @staticmethod
    def _format_detail(detail: Any) -> str:
        if isinstance(detail, dict):
            message = detail.get("message")
            notes = detail.get("notes")
            if message and notes:
                if isinstance(notes, list):
                    return f"{message}: {'; '.join(map(str, notes))}"
                return f"{message}: {notes}"
            if message:
                return str(message)
            if "detail" in detail:
                return ApiError._format_detail(detail["detail"])
        if isinstance(detail, list):
            return "; ".join(map(str, detail))
        return str(detail) if detail else "The request could not be completed."

    def __str__(self) -> str:
        if self.status_code == 401:
            return "Access was denied. Check the API key configuration."
        if self.status_code == 413:
            return "The uploaded file exceeds the 25 MB limit."
        if self.status_code == 422:
            return (
                f"{self._format_detail(self.detail)} "
                "Check the input and try again."
            )
        if self.status_code == 503:
            return (
                f"{self._format_detail(self.detail)} "
                "Check that the MISAN API and required model are available."
            )
        if self.status_code == 504:
            return self._format_detail(self.detail)
        return f"MISAN request failed (HTTP {self.status_code}): {self._format_detail(self.detail)}"


def _safe_json(response: requests.Response) -> Any:
    try:
        body = response.json()
    except (ValueError, requests.exceptions.JSONDecodeError):
        return response.text[:2000] or "The server returned an empty response."

    if isinstance(body, dict):
        return body.get("detail", body.get("message", body))
    return body


def _request_error(exc: requests.RequestException, operation: str) -> ApiError:
    if isinstance(exc, requests.Timeout):
        if operation == "training":
            message = (
                "Training timed out while waiting for a response. The server may "
                "still be processing the request. Check the active model version "
                "before attempting training again."
            )
        else:
            message = (
                "The MISAN API took too long to respond. Check the service status "
                "and retry when it is available."
            )
        return ApiError(504, message)

    if isinstance(exc, requests.ConnectionError):
        return ApiError(
            503,
            "Cannot connect to the MISAN API. Check the API service and its URL.",
        )

    return ApiError(
        502,
        f"Communication with the MISAN API failed: {type(exc).__name__}.",
    )


def _get(path: str, **params: Any) -> Any:
    try:
        response = requests.get(
            f"{BASE_URL}{path}",
            params=params,
            headers=_headers(),
            timeout=TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            raise ApiError(
                exc.response.status_code, _safe_json(exc.response)
            ) from exc
        raise _request_error(exc, "reading data") from exc

    try:
        return response.json()
    except ValueError as exc:
        raise ApiError(
            502, "The MISAN API returned an invalid response. Try again later."
        ) from exc


def _post(
    path: str,
    json_body: dict | None = None,
    files: dict | None = None,
    params: dict | None = None,
) -> Any:
    operation = "training" if path == "/train" else "request"
    timeout = 300 if path == "/train" else 60

    try:
        response = requests.post(
            f"{BASE_URL}{path}",
            json=json_body,
            files=files,
            params=params,
            headers=_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            raise ApiError(
                exc.response.status_code, _safe_json(exc.response)
            ) from exc
        raise _request_error(exc, operation) from exc

    try:
        return response.json()
    except ValueError as exc:
        raise ApiError(
            502, "The MISAN API returned an invalid response. Try again later."
        ) from exc


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
    return _post("/datasets/import", files={"file": (filename, file_bytes)})


def diagnose(
    asset_id: str,
    timestamps: list[str],
    readings: dict[str, list[float]],
) -> dict:
    return _post(
        "/diagnose",
        json_body={
            "asset_id": asset_id,
            "timestamps": timestamps,
            "readings": readings,
        },
    )


def train(dataset_path: str) -> dict:
    return _post("/train", params={"dataset_path": dataset_path})


def evaluate(version: str | None = None) -> dict:
    params = {"version": version} if version else None
    return _post("/evaluate", params=params)
