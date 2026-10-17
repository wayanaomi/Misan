import numpy as np
import pandas as pd

from misan.features import (
    FEATURE_COLUMNS,
    add_thermo_features,
    build_feature_table,
    saturation_temperature_kpa_to_c,
    window_feature_vector,
)


def _window_df(n=15):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
            "asset_id": "A1",
            "suction_pressure": 900 + rng.normal(0, 2, n),
            "discharge_pressure": 2200 + rng.normal(0, 5, n),
            "suction_temperature": 5 + rng.normal(0, 0.2, n),
            "discharge_temperature": 70 + rng.normal(0, 1, n),
            "condenser_inlet_temperature": 35 + rng.normal(0, 0.1, n),
            "condenser_outlet_temperature": 45 + rng.normal(0, 0.1, n),
            "evaporator_inlet_temperature": 20 + rng.normal(0, 0.1, n),
            "evaporator_outlet_temperature": 12 + rng.normal(0, 0.1, n),
            "refrigerant_mass_flow": 400 + rng.normal(0, 3, n),
            "compressor_power": 5 + rng.normal(0, 0.1, n),
            "ambient_temperature": 35.0,
            "load_setpoint": 18.0,
        }
    )


def test_saturation_temperature_monotonic():
    pressures = np.array([500, 1000, 2000, 3000])
    temps = saturation_temperature_kpa_to_c(pressures)
    assert np.all(np.diff(temps) > 0)


def test_add_thermo_features_columns():
    df = add_thermo_features(_window_df())
    for col in ["compression_ratio", "superheat", "subcooling", "condenser_approach", "evaporator_approach", "efficiency_indicator"]:
        assert col in df.columns
        assert df[col].notna().all()


def test_window_feature_vector_keys():
    vec = window_feature_vector(_window_df())
    assert "suction_pressure_mean" in vec
    assert "compression_ratio_mean" in vec
    assert "suction_pressure_slope" in vec


def test_build_feature_table_sliding_windows():
    df = _window_df(n=40)
    table = build_feature_table(df, window_length=15, step=5, group_col="asset_id")
    assert len(table) == len(range(0, 40 - 15 + 1, 5))
    for col in FEATURE_COLUMNS:
        assert col in table.columns
