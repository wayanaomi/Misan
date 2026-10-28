import pandas as pd
import pytest

from misan.config import SENSOR_NAMES
from misan.training import _validate_training_dataset


def make_dataset(run_count=4, rows_per_run=15):
    rows = []
    for run_number in range(run_count):
        for sample in range(rows_per_run):
            row = {
                "timestamp": pd.Timestamp("2026-10-01")
                + pd.Timedelta(minutes=sample),
                "asset_id": f"asset-{run_number}",
                "run_id": f"run-{run_number}",
                "fault_class": "NORMAL" if run_number < 2 else "CMP-DEG",
                "severity": 0.0 if run_number < 2 else 1.0,
            }
            row.update({sensor: 10.0 for sensor in SENSOR_NAMES})
            rows.append(row)
    return pd.DataFrame(rows)


def test_training_rejects_missing_training_columns():
    df = make_dataset().drop(columns=["severity"])

    with pytest.raises(ValueError, match="severity"):
        _validate_training_dataset(df)


def test_training_rejects_missing_physical_sensor():
    df = make_dataset().drop(columns=[SENSOR_NAMES[-1]])

    with pytest.raises(ValueError, match="sensor schema"):
        _validate_training_dataset(df)


def test_training_rejects_too_few_run_groups():
    df = make_dataset(run_count=3)

    with pytest.raises(ValueError, match="at least 4 distinct"):
        _validate_training_dataset(df)


def test_training_accepts_compatible_dataset():
    df = make_dataset()

    validated = _validate_training_dataset(df)

    assert len(validated) == len(df)
    assert set(SENSOR_NAMES).issubset(validated.columns)
    assert validated["run_id"].nunique() == 4
