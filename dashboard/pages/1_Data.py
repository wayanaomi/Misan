"""Misan dashboard — Data (PRD FR-1, FR-2, FR-25, Section 12 "Data")."""
from __future__ import annotations

import streamlit as st
import pandas as pd
from api_client import ApiError, datasets, diagnose, import_dataset, train
from theme import inject_css, page_header

from misan import ingestion
from misan.config import SENSOR_NAMES, WINDOW_LENGTH_SAMPLES

st.set_page_config(page_title="Misan · Data", layout="wide")
inject_css()

from auth import require_demo_password
require_demo_password()

st.sidebar.markdown("### MISAN")
st.sidebar.markdown("---")
st.sidebar.page_link("app.py", label="Overview")
st.sidebar.page_link("pages/1_Data.py", label="Data")
st.sidebar.page_link("pages/2_Model_Evaluation.py", label="Model Evaluation")

page_header("Data", "Import, inspect and act on datasets")

st.markdown("##### Import a dataset")
st.caption("CSV or Parquet, matching the schema in config/sensors.yaml. Labelled training "
           "datasets (run_id + fault_class + severity columns) can also be trained on directly.")
uploaded = st.file_uploader("Upload sensor data", type=["csv", "parquet"])
if uploaded is not None and st.button("Import"):
    try:
        result = import_dataset(uploaded.name, uploaded.getvalue())
        st.success(f'Imported {result["n_rows"]} rows (dataset id {result["dataset_id"]}).')
        q = result["data_quality"]
        st.json(q)

        compatible = result.get("compatible_with_current_model", True)
        if compatible:
            st.success("Dataset schema is compatible with the current model.")
        else:
            st.warning(
                "Dataset imported for inspection only. It cannot be used for diagnosis "
                "with the current model until the missing sensor signals are addressed."
            )

        schema_report = result.get("schema_report")
        if schema_report:
            st.markdown("**Schema compatibility report**")
            st.json(schema_report)
    except ApiError as exc:
        st.error(str(exc))


def render_historian_inspection(dataset_path: str) -> None:
    """Inspect historian-style CSV data without treating labels as verified diagnoses."""
    st.markdown("##### Historian dataset inspection")
    st.caption(
        "Exploratory analysis of the uploaded readings and their supplied labels. "
        "These labels are not independently verified fault diagnoses."
    )

    try:
        df = ingestion.read_raw(dataset_path)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read this dataset for inspection: {exc}")
        return

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("This dataset is empty or could not be interpreted as a table.")
        return

    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    timestamp_col = next(
        (column for column in ("timestamp", "time", "datetime")
         if column in df.columns),
        None,
    )
    asset_col = next(
        (column for column in ("unit_id", "asset_id", "equipment_id")
         if column in df.columns),
        None,
    )
    label_col = "fault_class" if "fault_class" in df.columns else None

    if timestamp_col:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{len(df):,}")
    metric_cols[1].metric(
        "Assets",
        str(df[asset_col].nunique()) if asset_col else "Not identified",
    )
    metric_cols[2].metric(
        "Time range",
        (
            f'{df[timestamp_col].min():%Y-%m-%d} to '
            f'{df[timestamp_col].max():%Y-%m-%d}'
        )
        if timestamp_col and df[timestamp_col].notna().any()
        else "Unavailable",
    )
    metric_cols[3].metric(
        "Columns",
        str(len(df.columns)),
    )

    if asset_col:
        assets = sorted(df[asset_col].dropna().astype(str).unique().tolist())
        selected_assets = st.multiselect(
            "Filter equipment",
            assets,
            default=assets,
            key="historian_asset_filter",
        )
        df = df[df[asset_col].astype(str).isin(selected_assets)]

    if df.empty:
        st.info("No rows match the selected equipment.")
        return

    if label_col:
        st.markdown("**Supplied fault-label distribution**")
        counts = (
            df[label_col]
            .fillna("Missing label")
            .astype(str)
            .value_counts()
            .rename_axis("Provided label")
            .reset_index(name="Rows")
        )
        st.bar_chart(counts.set_index("Provided label")["Rows"])
        st.dataframe(counts, width="stretch", hide_index=True)
        st.warning(
            "Labels shown here come from the file. Their provenance and "
            "whether they represent technician-confirmed faults are unknown."
        )
    else:
        st.info("No `fault_class` column was found; label distribution is unavailable.")

    excluded = {
        timestamp_col, asset_col, label_col, "run_id", "severity"
    }
    numeric_cols = [
        column for column in df.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(df[column])
    ]

    if timestamp_col and numeric_cols:
        st.markdown("**Sensor trends**")
        chosen_signals = st.multiselect(
            "Choose numeric signals",
            numeric_cols,
            default=numeric_cols[: min(3, len(numeric_cols))],
            key="historian_signal_filter",
        )
        plot_df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
        if chosen_signals and not plot_df.empty:
            trend = plot_df.set_index(timestamp_col)[chosen_signals]
            st.line_chart(trend)
        elif not chosen_signals:
            st.caption("Select at least one signal to display its trend.")
    else:
        st.info("No usable timestamp and numeric sensor columns were found for trend charts.")

    with st.expander("Preview rows and columns"):
        st.write("**Columns detected**")
        st.code(", ".join(df.columns))
        st.dataframe(df.head(100), width="stretch", hide_index=True)



st.markdown("##### Registered datasets")
try:
    ds_list = datasets()
except ApiError as exc:
    st.error(str(exc))
    ds_list = []

if not ds_list:
    st.markdown('<div class="misan-card">No dataset has been imported or simulated yet.</div>', unsafe_allow_html=True)
else:
    st.dataframe(ds_list, width="stretch", hide_index=True)
    options = {f'{d["name"]} ({d["source"]}, {d["n_rows"]} rows)': d["path"] for d in ds_list}
    selected_label = st.selectbox("Select a dataset to act on", list(options.keys()))
    selected_path = options[selected_label]

    render_historian_inspection(selected_path)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Train a model**")
        st.caption("Requires run_id, fault_class and severity columns (e.g. a simulated dataset).")
        if st.button("Train on selected dataset"):
            with st.spinner("Training — splitting by run, comparing model families, calibrating..."):
                try:
                    result = train(selected_path)
                    st.success(
                        f'Trained {result["version"]} ({result["chosen_model"]}), '
                        f'validation macro-F1 {result["validation_macro_f1"]:.3f}'
                    )
                    st.json(result["all_candidate_scores"])
                except ApiError as exc:
                    st.error(str(exc))

    with col_b:
        st.markdown("**Diagnose latest window per run/asset**")
        st.caption("Runs the real diagnostic pipeline on the most recent window of each run "
                   "(or asset, if the file has no run_id column) in this file.")
        max_windows = st.number_input("Maximum windows to diagnose", min_value=1, max_value=200, value=10)
        if st.button("Run diagnosis"):
            try:
                raw_df = ingestion.read_raw(selected_path)
                problems = ingestion.validate_schema(raw_df)
                if problems:
                    st.error("; ".join(problems))
                else:
                    clean_df, _ = ingestion.clean(raw_df)
                    # Group by run_id when present: multiple simulated runs can share the
                    # same asset_id, and grouping by asset_id alone would interleave them.
                    group_col = "run_id" if "run_id" in clean_df.columns else "asset_id"
                    produced = 0
                    for group_key, group in clean_df.groupby(group_col):
                        if produced >= max_windows:
                            break
                        window = group.sort_values("timestamp").tail(WINDOW_LENGTH_SAMPLES)
                        if len(window) < WINDOW_LENGTH_SAMPLES:
                            continue
                        real_asset = window["asset_id"].iloc[-1] if "asset_id" in window.columns else str(group_key)
                        label = f"{real_asset}::{group_key}" if group_col == "run_id" else str(group_key)
                        timestamps = window["timestamp"].astype(str).tolist()
                        readings = {s: window[s].astype(float).tolist() for s in SENSOR_NAMES}
                        record = diagnose(label, timestamps, readings)
                        produced += 1
                        with st.expander(f'{label} → {record["fault_class"]} ({record["confidence"]:.0%})'):
                            st.json(record)
                    if produced == 0:
                        st.warning(f"No run/asset in this file has at least {WINDOW_LENGTH_SAMPLES} samples.")
                    else:
                        st.success(f"Produced {produced} diagnostic record(s) — see Overview for the latest.")
            except ApiError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 — surface any pipeline error to the user, never hide it
                st.error(str(exc))
