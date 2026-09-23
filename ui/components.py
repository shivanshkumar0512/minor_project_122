"""Reusable HTML components (cards, KPIs, badges, verdicts, tooltips)."""
from __future__ import annotations

import html
from typing import Iterable

import streamlit as st

GLOSSARY = {
    "SHAP": "SHAP splits a prediction into per-feature contributions. They add up exactly to "
            "the model's output, so they are an exact receipt of how the decision was reached.",
    "trust": "Trust score T (0-1): the average of Confidence, (1 - Anomaly) and (1 - Stability "
             "deviation). Low trust sends the decision to a human reviewer.",
    "sigma": "σ deviation: how many standard deviations a feature's SHAP contribution is from its "
             "usual value on normal traffic. |σ| above ~3 is unusual.",
    "C": "Confidence C = max(p, 1-p). How sure the model is about its own decision.",
    "A": "Anomaly A: how unusual the explanation looks compared with normal model reasoning "
         "(Isolation Forest on standardised SHAP vectors), scaled 0-1 on clean data.",
    "S": "Stability deviation S: we nudge the numeric inputs by ±2% and see how much the explanation "
         "wobbles, relative to the usual amount. Too stable or too volatile are both suspicious.",
    "threshold": "Decisions with trust below this threshold go to the review queue. It is recalibrated "
                 "from reviewer outcomes to catch as many attacks as possible at ≤10% false alarms.",
    "pp_attack": "Prediction-preserving attack: the adversary changes inputs so the decision stays the "
                 "same but the stated reasons change. Output-only audits cannot see it.",
    "evasion": "Evasion attack: the adversary nudges inputs just enough to flip the decision.",
    "fpr": "False-positive rate: share of legitimate decisions sent to review.",
    "detection": "Detection rate: share of attacked decisions sent to review.",
    "auc": "ROC-AUC: probability a randomly chosen attacked record scores as more suspicious than a "
           "randomly chosen clean one. 0.5 = coin flip, 1.0 = perfect.",
    "drift": "Drift: the population slowly changes (e.g. incomes rise). It also moves explanations, "
             "but gradually, whereas attacks arrive as sudden, isolated spikes.",
}


def esc(s: object) -> str:
    return html.escape(str(s))


def tip(key_or_text: str) -> str:
    text = GLOSSARY.get(key_or_text, key_or_text)
    return f'<span class="tip" data-tip="{esc(text)}">?</span>'


def md(h: str) -> None:
    st.markdown(h, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "", icon: str | None = None) -> None:
    ic = f'<span class="material-symbols-rounded" style="font-size:16px">{esc(icon)}</span>' if icon else ""
    md(f"""<div class="page-head"><div>
        <div class="eyebrow">{ic}{esc(eyebrow)}</div>
        <h1>{esc(title)}</h1>{f'<p>{subtitle}</p>' if subtitle else ''}
        </div></div>""")


def section(title: str, tip_key: str | None = None) -> None:
    md(f'<div class="section-title">{esc(title)}{tip(tip_key) if tip_key else ""}</div>')


def kpi_html(label: str, value: str, delta: str | None = None, tone: str = "accent",
             delta_tone: str = "", tip_key: str | None = None) -> str:
    color = {"accent": "var(--accent)", "safe": "var(--safe)", "review": "var(--review)",
             "attack": "var(--attack)", "drift": "var(--drift)", "neutral": "var(--border-strong)"}[tone]
    d = f'<div class="d {delta_tone}">{esc(delta)}</div>' if delta else '<div class="d">&nbsp;</div>'
    return (f'<div class="kpi" style="--kpi-c:{color}"><div class="l">{esc(label)}'
            f'{tip(tip_key) if tip_key else ""}</div><div class="v">{esc(value)}</div>{d}</div>')


def kpi_row(items: Iterable[str]) -> None:
    md(f'<div class="kpi-row">{"".join(items)}</div>')


def badge(text: str, tone: str = "neutral", dot: bool = True) -> str:
    return f'<span class="badge {tone}">{"<span class=bdot></span>" if dot else ""}{esc(text)}</span>'


def verdict_badge(verdict: str) -> str:
    return badge("Accept", "safe") if verdict == "Accept" else badge("Review", "review")


def verdict_panel(verdict: str, trust: float, threshold: float, sub: str | None = None) -> None:
    if verdict == "Accept":
        cls, ic, t = "safe", "✓", "Accept"
        s = sub or f"Trust {trust:.3f} ≥ threshold {threshold:.3f}: explanation consistent with normal reasoning."
    else:
        cls, ic, t = "review", "!", "Route to review"
        s = sub or f"Trust {trust:.3f} < threshold {threshold:.3f}: explanation deviates from normal reasoning."
    md(f'<div class="verdict {cls}"><div class="ic">{ic}</div><div><div class="t">{t}</div>'
       f'<div class="s">{esc(s)}</div></div></div>')


def empty_state(title: str, text: str, icon: str = "inbox") -> None:
    md(f'<div class="empty"><div class="ic">{esc(icon)}</div><h4>{esc(title)}</h4><div>{text}</div></div>')


def skeleton(lines: int = 3, tall: bool = False) -> str:
    return "".join('<div class="skeleton tall"></div>' if tall else '<div class="skeleton"></div>'
                   for _ in range(lines))


def callout(text: str, warn: bool = False) -> None:
    md(f'<div class="callout{" warn" if warn else ""}">{text}</div>')


def fmt_pct(x: float, d: int = 1) -> str:
    return f"{100 * x:.{d}f}%"


def delta_str(new: float, old: float, pct: bool = False, d: int = 3) -> tuple[str, str]:
    diff = new - old
    if abs(diff) < 1e-12:
        return "no change", ""
    s = f"{'+' if diff > 0 else '−'}{100 * abs(diff):.1f} pp" if pct else f"{'+' if diff > 0 else '−'}{abs(diff):.{d}f}"
    return s, ("up" if diff > 0 else "down")
