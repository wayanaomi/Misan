"""Misan dashboard — Overview (PRD Section 13 Minimal User Experience, FR-26)."""
from __future__ import annotations

import streamlit as st
from api_client import ApiError, active_model, faults, health, recent_alerts, recent_diagnostics
from theme import SEVERITY_COLOR, STATUS_COLOR, VALIDATION_COLOR, badge, inject_css, page_header

st.set_page_config(page_title="Misan · Overview", layout="wide", initial_sidebar_state="expanded")
inject_css()

from auth import require_demo_password
require_demo_password()

st.sidebar.markdown("### MISAN")
st.sidebar.caption("Detect · Diagnose · Explain · Hand off")
st.sidebar.markdown("---")
st.sidebar.page_link("app.py", label="Overview")
st.sidebar.page_link("pages/1_Data.py", label="Data")
st.sidebar.page_link("pages/2_Model_Evaluation.py", label="Model Evaluation")

page_header("Overview", "Misan · HVAC fault detection")

try:
    status = health()
except ApiError as exc:
    st.error(str(exc))
    st.stop()

try:
    fault_meta = {f["code"]: f for f in faults()}
except ApiError:
    fault_meta = {}

if not status.get("active_model_version"):
    st.markdown(
        '<div class="misan-card">'
        '<span class="misan-eyebrow">Model status</span><br>'
        "No model has been trained yet. Generate a dataset and train a model from the "
        "<b>Data</b> page before diagnostics can run."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

try:
    diagnostics = recent_diagnostics(limit=10)
    alerts = recent_alerts(limit=10)
except ApiError as exc:
    st.error(str(exc))
    st.stop()

if not diagnostics:
    st.markdown(
        '<div class="misan-card">'
        '<span class="misan-eyebrow">System status</span><br>'
        "No diagnoses have been recorded yet. Submit a window for diagnosis from the "
        "<b>Data</b> page to populate this view."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

latest = diagnostics[0]

if latest["fault_class"] == "NORMAL":
    system_status = "normal"
elif latest["fault_detected"]:
    system_status = "fault"
else:
    system_status = "warning"

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown(
        f'<div class="misan-card">{badge(system_status, STATUS_COLOR[system_status])}'
        f'<div style="margin-top:0.5rem;font-size:0.8rem" class="misan-muted">'
        f'Model {latest["model_version"]}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    name = fault_meta.get(latest["fault_class"], {}).get("name", latest["fault_class"])
    st.markdown(
        f'<div class="misan-card"><b>{name}</b> &middot; {latest["component"]}'
        f'<div class="misan-muted">asset {latest["asset_id"]} &middot; {latest["timestamp"]}</div></div>',
        unsafe_allow_html=True,
    )

c1, c2, c3 = st.columns(3)
c1.metric("Confidence", f'{latest["confidence"]:.0%}')
if latest.get("severity_level"):
    c2.markdown(
        f'**Severity**<br>{badge(latest["severity_level"], SEVERITY_COLOR[latest["severity_level"]])} '
        f'<span class="misan-muted">{latest.get("severity_trend", "unknown")}</span>',
        unsafe_allow_html=True,
    )
else:
    c2.markdown("**Severity**<br><span class='misan-muted'>not applicable</span>", unsafe_allow_html=True)
c3.markdown(
    f'**Physics check**<br>{badge(latest["validation_status"], VALIDATION_COLOR[latest["validation_status"]])}',
    unsafe_allow_html=True,
)

st.markdown("##### Why")
st.write(latest["explanation"])
if latest.get("top_features"):
    for f in latest["top_features"][:3]:
        st.markdown(f'- `{f["feature"]}` — **{f["direction"]}** (weight {f["weight"]:+.3f})')
st.caption(latest.get("validation_note", ""))

st.markdown("##### Recent alerts")
if not alerts:
    st.caption("No fault alerts have been raised in the recorded diagnostics yet.")
else:
    rows = [
        {
            "Timestamp": a["timestamp"],
            "Asset": a["asset_id"],
            "Fault": a["fault_class"],
            "Severity": a.get("severity_level") or "—",
            "Confidence": f'{a["confidence"]:.0%}',
            "Physics validation": a["validation_status"].title(),
            "Validation note": a.get("validation_note", ""),
        }
        for a in alerts
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "An alert means the detector and persistence rule identified a fault. "
        "Physics validation is a separate check; a rejected result is not physics-confirmed."
    )
