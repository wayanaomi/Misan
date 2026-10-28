"""Misan dashboard — Model Evaluation (PRD FR-32, Section 11, Section 12)."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from api_client import ApiError, active_model, evaluate, evaluations
from theme import inject_css, page_header

st.set_page_config(page_title="Misan · Model Evaluation", layout="wide")
inject_css()

from auth import require_demo_password
require_demo_password()

st.sidebar.markdown("### MISAN")
st.sidebar.markdown("---")
st.sidebar.page_link("app.py", label="Overview")
st.sidebar.page_link("pages/1_Data.py", label="Data")
st.sidebar.page_link("pages/2_Model_Evaluation.py", label="Model Evaluation")

page_header("Model Evaluation", "Benchmark against the rule-based baseline")

try:
    active = active_model()
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if not active:
    st.markdown(
        '<div class="misan-card">No model has been trained yet. Train one from the '
        '<b>Data</b> page first.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown(f'<div class="misan-card">Active model <b>{active["version"]}</b> ({active["model_type"]})</div>', unsafe_allow_html=True)

if st.button("Run evaluation on the active model"):
    with st.spinner("Evaluating on the held-out test split..."):
        try:
            summary = evaluate(active["version"])
            st.success("Evaluation complete.")
        except ApiError as exc:
            if exc.status_code == 422:
                st.warning(
                    "The evaluation request could not be processed. "
                    "Check that the active model and evaluation dataset are valid."
                )
            else:
                st.error(str(exc))
            summary = None
        except Exception as exc:
            st.error(
                "Evaluation could not be completed. "
                f"Please check the service and try again ({type(exc).__name__})."
            )
            summary = None
    if summary:
        st.session_state["last_summary"] = summary

try:
    past_runs = evaluations(limit=10)
except ApiError as exc:
    past_runs = []
    st.warning(f"Previous evaluation records could not be loaded: {exc}")

summary = st.session_state.get("last_summary")
if summary is None and past_runs:
    latest = past_runs[0]
    if isinstance(latest, dict):
        summary = latest.get("summary")

if not isinstance(summary, dict) or not summary:
    st.info(
        "No usable evaluation summary is available yet. Run an evaluation "
        "when the active model and test dataset are ready."
    )
    st.stop()

required_metrics = {
    "macro_f1",
    "baseline_macro_f1",
    "false_alarm_rate",
    "calibration_error",
    "inference_time_ms",
    "per_class",
    "confusion_matrix",
    "noise_robustness",
    "physics_confirmation_rate",
    "physics_rejection_rate",
    "baseline_false_alarm_rate",
    "n_test_windows",
    "generated_at",
}
missing_metrics = sorted(required_metrics - summary.keys())
if missing_metrics:
    st.warning(
        "The evaluation result is incomplete, so the report cannot be "
        "displayed safely. Missing fields: " + ", ".join(missing_metrics)
    )
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Macro-F1", f'{summary["macro_f1"]:.3f}', delta=f'{summary["macro_f1"] - summary["baseline_macro_f1"]:+.3f} vs baseline')
c2.metric("False-alarm rate", f'{summary["false_alarm_rate"]:.1%}')
c3.metric("Calibration error", f'{summary["calibration_error"]:.3f}')
c4.metric("Inference time", f'{summary["inference_time_ms"]:.1f} ms')

st.markdown("##### Per-class performance")
per_class_df = pd.DataFrame(summary["per_class"]).T.reset_index(names="fault_class")
st.dataframe(per_class_df, width="stretch", hide_index=True)

st.markdown("##### Confusion matrix")
cm_df = pd.DataFrame(summary["confusion_matrix"]).T
st.dataframe(cm_df, width="stretch")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("##### Noise robustness (macro-F1 by noise level)")
    if summary["noise_robustness"]:
        st.bar_chart(pd.Series(summary["noise_robustness"], name="macro_f1"))
    else:
        st.caption("No noise-level metadata available in this test split.")
with col_b:
    st.markdown("##### Physics-check outcomes on test predictions")
    st.markdown(
        f'- Confirmed: **{summary["physics_confirmation_rate"]:.1%}**\n'
        f'- Rejected: **{summary["physics_rejection_rate"]:.1%}**\n'
        f'- Unchecked: **{max(0.0, 1 - summary["physics_confirmation_rate"] - summary["physics_rejection_rate"]):.1%}**'
    )

st.markdown("##### Versus rule-based baseline")
st.markdown(
    f'- Misan macro-F1 **{summary["macro_f1"]:.3f}** vs baseline **{summary["baseline_macro_f1"]:.3f}**\n'
    f'- Misan false-alarm rate **{summary["false_alarm_rate"]:.1%}** vs baseline **{summary["baseline_false_alarm_rate"]:.1%}**'
)
st.caption(f'{summary["n_test_windows"]} held-out test windows · generated {summary["generated_at"]}')
