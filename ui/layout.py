"""Custom sidebar (wordmark, domain switcher, navigation, theme) and page guard."""
from __future__ import annotations

import streamlit as st

from . import state
from .components import esc, md

PAGES = [
    ("app_pages/home.py", "Home", ":material/space_dashboard:"),
    ("app_pages/predict.py", "Predict", ":material/psychology:"),
    ("app_pages/attack.py", "Attack Simulator", ":material/crisis_alert:"),
    ("app_pages/monitor.py", "Live Monitor", ":material/monitor_heart:"),
    ("app_pages/review.py", "Review Queue", ":material/fact_check:"),
    ("app_pages/analytics.py", "Analytics", ":material/insights:"),
    ("app_pages/audit.py", "Audit Log", ":material/receipt_long:"),
    ("app_pages/admin.py", "Admin", ":material/admin_panel_settings:"),
]


def build_pages() -> list:
    return [st.Page(p, title=t, icon=i, default=(n == 0)) for n, (p, t, i) in enumerate(PAGES)]


def sidebar(pages: list) -> None:
    with st.sidebar:
        md("""<div class="brand"><div class="brand-mark">S</div><div>
              <div class="brand-name">SEAF</div>
              <div class="brand-sub">Secure Explainable AI Framework</div></div></div>""")

        doms = state.available_domains()
        if doms:
            cur = state.current_domain()
            if st.session_state.get("domain_select") != cur:
                st.session_state["domain_select"] = cur
            choice = st.selectbox("Active domain", doms, format_func=state.domain_title, key="domain_select",
                                  on_change=lambda: state.set_domain(st.session_state["domain_select"]),
                                  help="Every page works on the selected domain's model and baseline.")
            m = state.meta(choice) or {}
            sub = m.get("subtitle") or (state.DOMAINS[choice].subtitle if choice in state.DOMAINS else "Custom dataset")
            pending = state.db().counts(choice).get("pending", 0)
            md(f"""<div class="domain-pill"><span class="dot"></span><div>
                   <div class="t">{esc(state.domain_title(choice))} <span class="s">· {esc(sub)}</span></div>
                   <div class="s">threshold <span class="mono">{state.threshold(choice):.3f}</span> ·
                   {pending} pending review</div></div></div>""")

        md('<div class="side-label">Navigate</div>')
        for p in pages:
            st.page_link(p, label=p.title, icon=p.icon, width="stretch")

        md('<div class="side-label">Appearance</div>')
        light = st.toggle("Light theme", value=st.session_state.get("theme_mode") == "light", key="theme_toggle")
        want = "light" if light else "dark"
        if want != st.session_state.get("theme_mode", "dark"):
            st.session_state["theme_mode"] = want
            st.rerun()
        md('<div class="side-foot">Research prototype for decision <b>triage</b>, not automated rejection. '
           'Free-tier hosting: audit data resets on restart.</div>')


def guard() -> bool:
    """Show a friendly message when no artifacts are available."""
    if state.available_domains():
        return True
    from .components import empty_state
    empty_state("No trained models found",
                "Run <code>python scripts/train_all.py</code> to build the artifacts, then reload.", "model_training")
    return False
