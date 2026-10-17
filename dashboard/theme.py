"""Brand tokens and dark theme styling for the Misan dashboard."""
from __future__ import annotations

import streamlit as st

INK = "#10203F"
SURFACE = "#141B2D"
BORDER = "#2D374B"
MUTED = "#A3AEC2"

RED = "#D1503B"
AMBER = "#DB9A34"
GREEN = "#2C9166"
TEAL = "#1C7D7D"
PURPLE = "#8878F0"

STATUS_COLOR = {"normal": GREEN, "warning": AMBER, "fault": RED}
VALIDATION_COLOR = {"confirmed": GREEN, "rejected": RED, "unchecked": MUTED}
SEVERITY_COLOR = {"low": GREEN, "medium": AMBER, "high": RED}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1500px;
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: #0D111A;
        }}

        [data-testid="stSidebar"] {{
            background-color: {INK};
            border-right: 1px solid {BORDER};
        }}

        [data-testid="stSidebar"] * {{
            color: #E8EAF0;
        }}

        h1, h2, h3, h4, h5 {{
            color: #F3F5FA !important;
            font-weight: 650;
            letter-spacing: -0.02em;
        }}

        p, label, [data-testid="stMetricLabel"] {{
            color: #D7DCE7;
        }}

        .misan-eyebrow {{
            color: {PURPLE};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }}

        .misan-card {{
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            background: {SURFACE};
            color: #F3F5FA;
            margin-bottom: 0.75rem;
            overflow-wrap: anywhere;
        }}

        .misan-card b {{
            color: #F3F5FA;
        }}

        .misan-badge {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 700;
            color: #FFFFFF;
            white-space: nowrap;
        }}

        .misan-muted {{
            color: {MUTED};
            font-size: 0.85rem;
            overflow-wrap: anywhere;
        }}

        [data-testid="stDataFrame"] {{
            border: 1px solid {BORDER};
            border-radius: 8px;
            overflow: hidden;
        }}

        [data-testid="stMetricValue"] {{
            color: #F3F5FA;
        }}

        hr {{
            margin: 0.7rem 0;
            border-color: {BORDER};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, color: str) -> str:
    safe_text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<span class="misan-badge" '
        f'style="background:{color}">{safe_text.upper()}</span>'
    )


def page_header(title: str, eyebrow: str) -> None:
    st.markdown(
        f'<div class="misan-eyebrow">{eyebrow}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"### {title}")
