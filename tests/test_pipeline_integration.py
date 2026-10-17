"""End-to-end integration test: simulate -> train -> evaluate -> live diagnose ->
API, run against an isolated temp DB/model dir (see conftest.py)."""
from __future__ import annotations

import pytest

from misan.evaluation import evaluate
from misan.pipeline import diagnose_window
from misan.schemas import SensorWindow

from .conftest import FAULT_SUBSET


def test_training_produces_artifacts(trained_version):
    result, _ = trained_version
    assert (result.model_dir / "diagnosis_model.joblib").exists()
    assert (result.model_dir / "detection_model.joblib").exists()
    assert (result.model_dir / "severity_model.joblib").exists()
    assert 0.0 <= result.validation_macro_f1 <= 1.0


def test_evaluation_summary_is_sane(trained_version):
    result, _ = trained_version
    summary = evaluate(result.version)
    assert 0.0 <= summary.macro_f1 <= 1.0
    assert 0.0 <= summary.false_alarm_rate <= 1.0
    assert 0.0 <= summary.calibration_error <= 1.0
    assert summary.n_test_windows > 0
    for code, metrics in summary.per_class.items():
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0


def test_detection_run_metrics_counts_missed_runs():
    import numpy as np
    import pandas as pd
    from misan.evaluation import _detection_run_metrics

    frame = pd.DataFrame({
        "run_id": ["fault-a"] * 3 + ["fault-b"] * 3 + ["healthy"] * 3,
        "timestamp": list(range(3)) * 3,
        "fault_class": ["CMP-DEG"] * 3 + ["REF-LEAK"] * 3 + ["NORMAL"] * 3,
    })
    predictions = np.array([
        True, True, True,   # fault-a triggers after persistence
        False, False, False,  # fault-b is missed
        False, False, False,  # healthy run is excluded from faulty-run recall
    ])

    result = _detection_run_metrics(frame, predictions, persistence_n=2)

    assert result["n_faulty_runs"] == 2
    assert result["n_detected_faulty_runs"] == 1
    assert result["detection_miss_rate"] == 0.5
    assert result["detection_run_recall"] == 0.5


def test_diagnose_window_on_faulty_run(trained_version):
    _, df = trained_version
    faulty_run = df[(df["fault_class"] == "CMP-DEG") & (df["severity"] == 1.0)].sort_values("timestamp")
    run_id = faulty_run["run_id"].iloc[0]
    one_run = faulty_run[faulty_run["run_id"] == run_id].tail(15)

    from misan.config import SENSOR_NAMES

    window = SensorWindow(
        asset_id="TEST-ASSET",
        timestamps=one_run["timestamp"].astype(str).tolist(),
        readings={s: one_run[s].astype(float).tolist() for s in SENSOR_NAMES},
    )
    record = diagnose_window(window)
    assert record.fault_class in FAULT_SUBSET + ["UNKNOWN"]
    assert 0.0 <= record.confidence <= 1.0
    assert abs(sum(record.class_probabilities.values()) - 1.0) < 0.05
    assert record.validation_status in ("confirmed", "rejected", "unchecked")


def test_diagnose_window_insufficient_data_raises(trained_version):
    from misan.pipeline import InsufficientDataError
    from misan.config import SENSOR_NAMES

    window = SensorWindow(
        asset_id="TEST-ASSET-SHORT",
        timestamps=["2026-01-01T00:00:00"] * 3,
        readings={s: [1.0, 2.0, 3.0] for s in SENSOR_NAMES},
    )
    with pytest.raises(InsufficientDataError):
        diagnose_window(window)


def test_alert_raised_after_three_consecutive_faulty_windows(trained_version):
    import misan.pipeline as pipeline
    from misan.config import SENSOR_NAMES
    from misan.detection import PersistenceTracker

    _, df = trained_version
    faulty_run = df[
        (df["fault_class"] == "CMP-DEG") & (df["severity"] == 1.0)
    ].sort_values("timestamp")

    run_id = faulty_run["run_id"].iloc[0]
    one_run = faulty_run[faulty_run["run_id"] == run_id].tail(15)

    window = SensorWindow(
        asset_id="ALERT-TEST-ASSET",
        timestamps=one_run["timestamp"].astype(str).tolist(),
        readings={s: one_run[s].astype(float).tolist() for s in SENSOR_NAMES},
    )

    original_tracker = pipeline._persistence_tracker
    pipeline._persistence_tracker = PersistenceTracker(n=3)

    try:
        records = [pipeline.diagnose_window(window) for _ in range(3)]

        assert records[0].fault_detected is False
        assert records[1].fault_detected is False
        assert records[2].fault_detected is True
    finally:
        pipeline._persistence_tracker = original_tracker
