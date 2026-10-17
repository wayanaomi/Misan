import pandas as pd
import pytest

from misan import ingestion
from misan.config import SENSOR_NAMES


def _sample_df(n=20):
    data = {
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        "asset_id": "A1",
    }
    defaults = {
        "suction_pressure": 900,
        "discharge_pressure": 2200,
        "suction_temperature": 5,
        "discharge_temperature": 70,
        "condenser_inlet_temperature": 35,
        "condenser_outlet_temperature": 45,
        "evaporator_inlet_temperature": 20,
        "evaporator_outlet_temperature": 12,
        "refrigerant_mass_flow": 400,
        "compressor_power": 5,
        "ambient_temperature": 35,
        "load_setpoint": 18,
    }
    for name in SENSOR_NAMES:
        data[name] = [defaults[name]] * n
    return pd.DataFrame(data)


def test_validate_schema_ok():
    df = _sample_df()
    assert ingestion.validate_schema(df) == []


def test_validate_schema_missing_signal():
    df = _sample_df().drop(columns=["suction_pressure"])
    problems = ingestion.validate_schema(df)
    assert problems
    assert "suction_pressure" in problems[0]


def test_flag_out_of_range():
    df = _sample_df()
    df.loc[0, "suction_pressure"] = -500  # below valid_range
    flagged = ingestion.flag_out_of_range(df)
    assert flagged.loc[0, "suction_pressure__oor"]
    assert not flagged.loc[1, "suction_pressure__oor"]


def test_fill_missing_short_gap():
    df = _sample_df()
    df.loc[2, "suction_pressure"] = None
    filled, missing_before = ingestion.fill_missing(df)
    assert missing_before == 1
    assert not filled["suction_pressure"].isna().any()


def test_clean_end_to_end():
    df = _sample_df()
    clean_df, quality = ingestion.clean(df)
    assert quality.sufficient
    assert quality.sample_count == len(df)


def test_clean_rejects_bad_schema():
    df = _sample_df().drop(columns=["asset_id"])
    with pytest.raises(ValueError):
        ingestion.clean(df)
