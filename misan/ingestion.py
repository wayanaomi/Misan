"""Data ingestion and quality checks (PRD FR-1, FR-2).

Training/evaluation datasets and live diagnostic submissions both pass through this
module so cleaning behaviour stays identical between training and inference (PRD
Section 7, "keep preprocessing identical...to avoid data leakage").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from misan.config import SENSOR_BY_NAME, SENSOR_NAMES
from misan.schemas import DataQuality


REQUIRED_META_COLUMNS = {"timestamp", "asset_id"}

# Only direct, semantically defensible mappings. No physical sensor values
# are inferred or synthesized by this mapping.
HVAC_SOURCE_COLUMN_MAP = {
    "unit_id": "asset_id",
    "suction_pressure_kpa": "suction_pressure",
    "discharge_pressure_kpa": "discharge_pressure",
    "suction_temp_c": "suction_temperature",
    "discharge_temp_c": "discharge_temperature",
    "compressor_power_kw": "compressor_power",
    "ambient_temp_c": "ambient_temperature",
}

def assess_dataset_schema(df: pd.DataFrame) -> dict:
    """Assess import compatibility without modifying or fabricating measurements."""
    source_columns = list(df.columns)
    mapped = {
        source: target for source, target in HVAC_SOURCE_COLUMN_MAP.items()
        if source in df.columns and target not in df.columns
    }
    available_after_mapping = set(source_columns)
    available_after_mapping.update(mapped.values())

    missing_meta = sorted(REQUIRED_META_COLUMNS - available_after_mapping)
    missing_signals = [name for name in SENSOR_NAMES if name not in available_after_mapping]
    missing_training = [
        name for name in ("run_id", "fault_class")
        if name not in available_after_mapping
    ]
    return {
        "source_columns": source_columns,
        "proposed_column_mapping": mapped,
        "missing_required_metadata": missing_meta,
        "missing_sensor_signals": missing_signals,
        "missing_training_columns": missing_training,
        "schema_compatible": not missing_meta and not missing_signals,
        "ready_for_training": not missing_meta and not missing_signals and not missing_training,
        "note": "Missing physical measurements are not generated or imputed.",
    }



def read_raw(path: str | Path) -> pd.DataFrame:
    """Read a simulator-exported CSV or Parquet file into a DataFrame."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported file type: {path.suffix} (expected .csv or .parquet)")
    return df



def normalize_source_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known source headers only; never synthesize missing sensor readings."""
    df = df.copy()
    rename = {
        source: target
        for source, target in HVAC_SOURCE_COLUMN_MAP.items()
        if source in df.columns and target not in df.columns
    }
    return df.rename(columns=rename)


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable schema problems; empty list means valid."""
    problems: list[str] = []
    missing_meta = REQUIRED_META_COLUMNS - set(df.columns)
    if missing_meta:
        problems.append(f"missing required columns: {sorted(missing_meta)}")
    missing_signals = [s for s in SENSOR_NAMES if s not in df.columns]
    if missing_signals:
        problems.append(f"missing sensor signals: {missing_signals}")
    return problems


def align_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps, drop exact duplicates, and sort chronologically per asset
    (and per run, when a `run_id` column is present — distinct simulated runs for the
    same asset can share a synthetic timestamp range and must not be deduplicated
    against each other)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    n_unparsable = int(df["timestamp"].isna().sum())
    if n_unparsable:
        df = df.dropna(subset=["timestamp"])
    key_cols = [c for c in ("run_id", "asset_id") if c in df.columns] + ["timestamp"]
    df = df.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last")
    return df.reset_index(drop=True)


def flag_out_of_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean `<signal>__oor` column for readings outside the configured valid range."""
    df = df.copy()
    for name in SENSOR_NAMES:
        if name not in df.columns:
            continue
        lo, hi = SENSOR_BY_NAME[name].valid_range
        df[f"{name}__oor"] = (df[name] < lo) | (df[name] > hi)
    return df


def fill_missing(df: pd.DataFrame, max_gap: int = 3) -> tuple[pd.DataFrame, int]:
    """Linearly interpolate short gaps (<= max_gap samples) per signal; longer gaps are
    left as NaN so downstream windowing can flag the window as data-insufficient."""
    df = df.copy()
    missing_before = int(df[SENSOR_NAMES].isna().sum().sum()) if set(SENSOR_NAMES) <= set(df.columns) else 0
    for name in SENSOR_NAMES:
        if name not in df.columns:
            continue
        df[name] = df[name].interpolate(method="linear", limit=max_gap, limit_area="inside")
    return df, missing_before


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, DataQuality]:
    """Run the full cleaning sequence and return the cleaned frame plus a quality summary."""
    problems = validate_schema(df)
    if problems:
        raise ValueError("; ".join(problems))
    df = align_timestamps(df)
    df, missing_before = fill_missing(df)
    df = flag_out_of_range(df)
    oor_cols = [c for c in df.columns if c.endswith("__oor")]
    out_of_range_count = int(df[oor_cols].sum().sum()) if oor_cols else 0
    remaining_missing = int(df[SENSOR_NAMES].isna().sum().sum())
    quality = DataQuality(
        missing_value_count=missing_before,
        out_of_range_count=out_of_range_count,
        sample_count=len(df),
        sufficient=remaining_missing == 0,
        notes=(["unrecoverable gaps remain after interpolation"] if remaining_missing else []),
    )
    return df, quality


def window_quality(window_df: pd.DataFrame, min_samples: int) -> DataQuality:
    """Data-quality summary scoped to a single diagnostic window."""
    oor_cols = [c for c in window_df.columns if c.endswith("__oor")]
    out_of_range_count = int(window_df[oor_cols].sum().sum()) if oor_cols else 0
    missing = int(window_df[SENSOR_NAMES].isna().sum().sum()) if set(SENSOR_NAMES) <= set(window_df.columns) else 0
    sufficient = len(window_df) >= min_samples and missing == 0
    notes = []
    if len(window_df) < min_samples:
        notes.append(f"only {len(window_df)}/{min_samples} samples available")
    if missing:
        notes.append(f"{missing} missing readings in window")
    return DataQuality(
        missing_value_count=missing,
        out_of_range_count=out_of_range_count,
        sample_count=len(window_df),
        sufficient=sufficient,
        notes=notes,
    )


def sensor_window_to_frame(asset_id: str, timestamps: list[str], readings: dict[str, list[float]]) -> pd.DataFrame:
    """Convert an API `SensorWindow` payload into a DataFrame compatible with `clean`."""
    data = {"timestamp": timestamps, "asset_id": asset_id}
    data.update(readings)
    df = pd.DataFrame(data)
    lengths = {len(v) for v in readings.values()} | {len(timestamps)}
    if len(lengths) > 1:
        raise ValueError("all signals and timestamps must have the same length")
    return df
