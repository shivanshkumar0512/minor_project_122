"""Theme: CSS injection, light/dark tokens and the shared Plotly template."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"

PALETTE = {
    "dark": {
        "bg": "#0B0F17", "surface": "#111725", "surface2": "#161E2F", "grid": "rgba(148,163,184,0.10)",
        "axis": "#2F3A52", "text": "#E6EAF2", "text2": "#B4BDCC", "muted": "#8591A6",
    },
    "light": {
        "bg": "#F5F7FB", "surface": "#FFFFFF", "surface2": "#F1F4F9", "grid": "rgba(15,23,42,0.07)",
        "axis": "#CBD3E1", "text": "#0F172A", "text2": "#334155", "muted": "#64748B",
    },
}
ACCENT = "#5B8CFF"
SAFE, REVIEW, ATTACK, DRIFT = "#22C55E", "#F59E0B", "#EF4444", "#A78BFA"
# Categorical order validated for CVD separation (dataviz validator) per mode
SERIES = {"dark": ["#5B8CFF", "#C9761F", "#8B7CF6", "#1FA37A"],
          "light": ["#3B6BFF", "#C8741C", "#7C5CE6", "#15875F"]}

LIGHT_OVERRIDES = """
:root {
  --bg:#F5F7FB; --surface:#FFFFFF; --surface-2:#F1F4F9; --surface-3:#E7ECF4;
  --border:#E2E8F0; --border-strong:#CBD5E1; --text:#0F172A; --text-2:#334155; --muted:#64748B;
  --accent:#3B6BFF; --accent-2:#2F5BEA; --accent-soft:rgba(59,107,255,.10);
  --safe:#16A34A; --review:#D97706; --attack:#DC2626; --drift:#7C3AED;
  --shadow: 0 1px 2px rgba(15,23,42,.06), 0 6px 18px rgba(15,23,42,.06);
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg) !important; color: var(--text) !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; }
.stApp p, .stApp li, .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"],
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] { color: var(--text-2); }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, [data-testid="stSidebar"] p { color: var(--text) !important; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"], .stNumberInput input,
.stTextInput input, [data-baseweb="base-input"] { background: var(--surface) !important; color: var(--text) !important; border-color: var(--border-strong) !important; }
[data-baseweb="input"] input, [data-baseweb="select"] span, [data-baseweb="select"] div { color: var(--text) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: var(--surface) !important; }
[data-baseweb="popover"] li { color: var(--text) !important; }
.stButton > button:not([kind="primary"]), .stDownloadButton > button { background: var(--surface) !important; color: var(--text) !important; }
[data-testid="stNumberInputContainer"] button { background: var(--surface-2) !important; color: var(--text) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--text) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] p { color: var(--text) !important; }
.stTabs [data-baseweb="tab"] p { color: var(--text-2) !important; }
[data-testid="stSegmentedControl"] button, [data-testid="stPills"] button { background: var(--surface) !important; color: var(--text) !important; }
/* react-aria selects / multiselects (Streamlit >= 1.5x) */
.react-aria-ComboBox [role="group"], .stSelectbox [role="group"], .stMultiSelect [role="group"],
.stMultiSelect div[class*="st-emotion"] { background: var(--surface) !important; color: var(--text) !important; }
.stSelectbox input, .stMultiSelect input, [role="combobox"] { color: var(--text) !important; background: transparent !important; }
[data-testid="stSelectboxVirtualDropdown"], [data-testid*="Dropdown"], [role="listbox"] { background: var(--surface) !important; color: var(--text) !important; }
[role="option"] { color: var(--text) !important; }
[role="option"]:hover, [role="option"][data-focused="true"] { background: var(--surface-2) !important; }
.stButton button[data-testid="stBaseButton-secondary"], .stButton button[data-testid="stBaseButton-secondary"]:hover,
.stButton button[data-testid="stBaseButton-secondary"]:focus, .stButton button[data-testid="stBaseButton-secondary"]:active,
[data-testid="stPopover"] button { background: var(--surface) !important; color: var(--text) !important; border-color: var(--border-strong) !important; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p { color: #fff !important; }
[data-testid="stTooltipIcon"] svg { color: var(--muted) !important; }
"""


@lru_cache(maxsize=1)
def _css() -> str:
    return (ASSETS / "style.css").read_text(encoding="utf-8")


def mode() -> str:
    return st.session_state.get("theme_mode", "dark")


def inject() -> None:
    css = _css()
    if mode() == "light":
        css += LIGHT_OVERRIDES
    st.html(f"<style>{css}</style>")
    for m in PALETTE:  # both registered once; each chart picks the viewer's mode
        if f"seaf_{m}" not in pio.templates:
            _register_plotly(m)


def _register_plotly(m: str) -> None:
    p = PALETTE[m]
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", size=12.5, color=p["text2"]),
        title=dict(font=dict(size=14, color=p["text"]), x=0, xanchor="left", y=0.97),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES[m],
        margin=dict(l=8, r=8, t=36, b=8),
        xaxis=dict(gridcolor=p["grid"], zerolinecolor=p["axis"], linecolor=p["axis"], showline=False,
                   tickfont=dict(color=p["muted"]), title=dict(font=dict(color=p["muted"], size=12))),
        yaxis=dict(gridcolor=p["grid"], zerolinecolor=p["axis"], linecolor=p["axis"], showline=False,
                   tickfont=dict(color=p["muted"]), title=dict(font=dict(color=p["muted"], size=12))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=p["text2"], size=12)),
        hoverlabel=dict(bgcolor=p["surface2"], bordercolor=p["axis"],
                        font=dict(family="Inter, sans-serif", color=p["text"], size=12.5)),
        bargap=0.28,
        polar=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates[f"seaf_{m}"] = tpl


def plotly_template() -> str:
    """Per-session template name (the global Plotly default is shared by all sessions)."""
    return f"seaf_{mode()}"


SEMANTIC = {
    "dark": {"accent": "#5B8CFF", "safe": "#22C55E", "review": "#F59E0B", "attack": "#EF4444", "drift": "#A78BFA"},
    "light": {"accent": "#3B6BFF", "safe": "#16A34A", "review": "#D97706", "attack": "#DC2626", "drift": "#7C3AED"},
}


def colors() -> dict:
    """Current-mode tokens: surfaces/text + semantic colours + series order."""
    m = mode()
    return {**PALETTE[m], **SEMANTIC[m], "series": SERIES[m]}
