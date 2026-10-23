"""Optional shared-password gate for the public demonstration dashboard."""
from __future__ import annotations

import hmac
import os
import streamlit as st


def require_demo_password() -> None:
    expected = os.environ.get("MISAN_DEMO_PASSWORD", "")
    production = os.environ.get("MISAN_ENV", "").lower() == "production"

    if production and not expected:
        st.error("Demo access is not configured. Contact the MISAN administrator.")
        st.stop()

    if not expected:
        return

    if st.session_state.get("misan_authenticated") is True:
        return

    st.title("MISAN Access")
    supplied = st.text_input("Demo password", type="password")
    if st.button("Unlock dashboard", type="primary"):
        if hmac.compare_digest(supplied, expected):
            st.session_state["misan_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()
