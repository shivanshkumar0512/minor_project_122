"""SEAF dashboard entry point.

    streamlit run app.py

Pages live in ``app_pages/``; shared UI code in ``ui/``; the framework itself
in the reusable ``seaf/`` library.
"""
from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import layout, state, theme  # noqa: E402
from ui.components import md  # noqa: E402

st.set_page_config(
    page_title="SEAF · Secure Explainable AI",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="auto",
)

log = logging.getLogger("seaf")


def ensure_artifacts() -> None:
    """Artifacts are committed/pre-built; train only as a last-resort fallback."""
    if state.available_domains():
        return
    with st.status("First start: building models (one-off, ~5-8 min on a free CPU)…", expanded=True) as s:
        from scripts.train_all import train_domain
        from seaf.config import DOMAINS
        for key in DOMAINS:
            s.write(f"Training {key}…")
            train_domain(key)
        s.update(label="Models ready", state="complete")
    st.rerun()


theme.inject()
pages = layout.build_pages()
nav = st.navigation(pages, position="hidden")

try:
    ensure_artifacts()
    state.db()  # initialise + seed the audit DB once per process
    layout.sidebar(pages)
    nav.run()
except Exception as exc:  # graceful error instead of a stack trace
    log.error("Unhandled error on page %s: %s\n%s", nav.title, exc, traceback.format_exc())
    md(f"""<div class="card" style="border-color:rgba(239,68,68,.45)">
        <h4>Something went wrong on this page</h4>
        <div class="muted">The error has been logged. Try another page or reload. If it persists, the
        artifacts may be out of date - rebuild them with <code>python scripts/train_all.py</code>.</div>
        <div class="muted mono" style="margin-top:8px">{type(exc).__name__}: {str(exc)[:200]}</div></div>""")
