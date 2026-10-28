import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import api_client


def test_get_connection_failure_has_friendly_message(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(api_client.requests, "get", fail)

    with pytest.raises(api_client.ApiError) as exc:
        api_client._get("/health")

    assert exc.value.status_code == 503
    assert "Cannot connect" in str(exc.value)


def test_get_timeout_has_friendly_message(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(api_client.requests, "get", fail)

    with pytest.raises(api_client.ApiError) as exc:
        api_client._get("/health")

    assert exc.value.status_code == 504
    assert "too long to respond" in str(exc.value)


def test_get_http_error_preserves_status(monkeypatch):
    response = Mock()
    response.status_code = 422
    response.json.return_value = {"detail": "Invalid dataset"}
    response.text = '{"detail": "Invalid dataset"}'
    response.raise_for_status.side_effect = requests.HTTPError(
        response=response
    )

    monkeypatch.setattr(
        api_client.requests, "get", lambda *args, **kwargs: response
    )

    with pytest.raises(api_client.ApiError) as exc:
        api_client._get("/health")

    assert exc.value.status_code == 422
    assert "Invalid dataset" in str(exc.value)


def test_get_invalid_json_becomes_api_error(monkeypatch):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.side_effect = ValueError("invalid JSON")

    monkeypatch.setattr(
        api_client.requests, "get", lambda *args, **kwargs: response
    )

    with pytest.raises(api_client.ApiError) as exc:
        api_client._get("/health")

    assert exc.value.status_code == 502
    assert "invalid response" in str(exc.value)


def test_train_timeout_warns_not_to_retry_blindly(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.Timeout("training timed out")

    monkeypatch.setattr(api_client.requests, "post", fail)

    with pytest.raises(api_client.ApiError) as exc:
        api_client.train("/tmp/example.csv")

    assert exc.value.status_code == 504
    assert "active model version" in str(exc.value)
    assert "training again" in str(exc.value)


def test_error_response_with_html_does_not_break_client(monkeypatch):
    response = Mock()
    response.status_code = 500
    response.text = "<html>Server error</html>"
    response.json.side_effect = ValueError("not JSON")
    response.raise_for_status.side_effect = requests.HTTPError(
        response=response
    )

    monkeypatch.setattr(
        api_client.requests, "get", lambda *args, **kwargs: response
    )

    with pytest.raises(api_client.ApiError) as exc:
        api_client._get("/health")

    assert exc.value.status_code == 500
    assert "MISAN request failed" in str(exc.value)
