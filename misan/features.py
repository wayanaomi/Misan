"""Feature engineering: sliding-window statistics and derived thermodynamic indicators
(PRD FR-3, FR-4).

Thermodynamic formulas and their assumptions
---------------------------------------------
- compression_ratio = discharge_pressure / suction_pressure (standard definition).
- superheat = suction_temperature - saturation_temperature(suction_pressure).
- subcooling = saturation_temperature(discharge_pressure) - condenser_outlet_temperature.
- condenser_approach = saturation_temperature(discharge_pressure) - condenser_outlet_temperature
  (terminal temperature difference at the condenser outlet).
- evaporator_approach = evaporator_inlet_temperature - saturation_temperature(suction_pressure).
- efficiency_indicator = (evaporator_inlet_temperature - evaporator_outlet_temperature)
  * refrigerant_mass_flow / compressor_power, a simplified cooling-effect-per-unit-power
  proxy. It is NOT a true COP (no enthalpy/specific-heat data is available in the MVP
  sensor list) — documented approximation for relative comparison only.

`saturation_temperature_kpa_to_c` is a degree-2 polynomial fit through public, approximate
R410A saturation pressure/temperature reference points. It is a demonstration proxy for
the MVP, not a certified refrigerant property library, and should not be relied on for
real equipment sizing or certification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from misan.config import SENSOR_NAMES

# Approximate R410A saturation reference points: (pressure kPa, temperature degC)
_SAT_REF_PRESSURE = np.array([400, 573, 798, 1086, 1444, 1882, 2408, 3026, 3749], dtype=float)
_SAT_REF_TEMP = np.array([-20, -10, 0, 10, 20, 30, 40, 50, 60], dtype=float)
_SAT_COEFFS = np.polyfit(_SAT_REF_PRESSURE, _SAT_REF_TEMP, deg=2)

THERMO_COLUMNS = [
    "compression_ratio",
    "superheat",
    "subcooling",
    "condenser_approach",
    "evaporator_approach",
    "efficiency_indicator",
]


def saturation_temperature_kpa_to_c(pressure_kpa: pd.Series | np.ndarray) -> np.ndarray:
    pressure = np.clip(np.asarray(pressure_kpa, dtype=float), _SAT_REF_PRESSURE.min(), _SAT_REF_PRESSURE.max())
    return np.polyval(_SAT_COEFFS, pressure)



def validate_feature_inputs(df: pd.DataFrame) -> None:
    """Reject absent physical inputs instead of producing NaN thermodynamic features."""
    required = list(SENSOR_NAMES)
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(
            "Cannot calculate HVAC features: missing required sensor signals: "
            + ", ".join(missing)
        )

    numeric = df[required].apply(pd.to_numeric, errors="coerce")
    invalid = [
        name for name in required
        if numeric[name].isna().any()
        or not np.isfinite(numeric[name].to_numpy(dtype=float)).all()
    ]
    if invalid:
        raise ValueError(
            "Cannot calculate HVAC features: missing or non-finite readings in: "
            + ", ".join(invalid)
        )


def add_thermo_features(df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    validate_feature_inputs(df)
    df = df.copy()
    t_sat_suction = saturation_temperature_kpa_to_c(df["suction_pressure"])
    t_sat_discharge = saturation_temperature_kpa_to_c(df["discharge_pressure"])

    df["compression_ratio"] = df["discharge_pressure"] / df["suction_pressure"].clip(lower=eps)
    df["superheat"] = df["suction_temperature"] - t_sat_suction
    df["subcooling"] = t_sat_discharge - df["condenser_outlet_temperature"]
    df["condenser_approach"] = t_sat_discharge - df["condenser_outlet_temperature"]
    df["evaporator_approach"] = df["evaporator_inlet_temperature"] - t_sat_suction
    df["efficiency_indicator"] = (
        (df["evaporator_inlet_temperature"] - df["evaporator_outlet_temperature"]) * df["refrigerant_mass_flow"]
    ) / df["compressor_power"].clip(lower=eps)
    return df


def _slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2 or np.isnan(values).all():
        return 0.0
    x = np.arange(n, dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return 0.0
    coeffs = np.polyfit(x[mask], values[mask], deg=1)
    return float(coeffs[0])


def window_feature_vector(window_df: pd.DataFrame) -> dict[str, float]:
    """Mean/std/slope/range per signal and per derived thermodynamic feature (FR-3, FR-4)."""
    enriched = add_thermo_features(window_df)
    columns = [c for c in SENSOR_NAMES if c in enriched.columns] + THERMO_COLUMNS
    features: dict[str, float] = {}
    for col in columns:
        values = enriched[col].to_numpy(dtype=float)
        features[f"{col}_mean"] = float(np.nanmean(values)) if len(values) else 0.0
        features[f"{col}_std"] = float(np.nanstd(values)) if len(values) else 0.0
        features[f"{col}_slope"] = _slope(values)
        features[f"{col}_range"] = (
            float(np.nanmax(values) - np.nanmin(values)) if len(values) and not np.isnan(values).all() else 0.0
        )
    return features


FEATURE_COLUMNS: list[str] = [
    f"{col}_{stat}" for col in (SENSOR_NAMES + THERMO_COLUMNS) for stat in ("mean", "std", "slope", "range")
]


def build_feature_table(
    df: pd.DataFrame,
    window_length: int,
    step: int = 1,
    group_col: str = "asset_id",
    passthrough_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Slide a window of `window_length` samples (stride `step`) over each group and
    emit one feature row per window. `passthrough_cols` (e.g. asset_id, fault_class,
    severity) are copied from the window's last sample, used when building training sets."""
    rows: list[dict[str, float | str]] = []
    group_iter = df.groupby(group_col) if group_col in df.columns else [(None, df)]
    for group_key, group_df in group_iter:
        group_df = group_df.sort_values("timestamp").reset_index(drop=True)
        for start in range(0, max(len(group_df) - window_length + 1, 0), step):
            window = group_df.iloc[start : start + window_length]
            if len(window) < window_length:
                continue
            row = window_feature_vector(window)
            row[group_col] = group_key if group_key is not None else "unknown"
            row["timestamp"] = str(window.iloc[-1]["timestamp"])
            for col in passthrough_cols:
                if col in window.columns:
                    row[col] = window.iloc[-1][col]
            rows.append(row)
    return pd.DataFrame(rows)
