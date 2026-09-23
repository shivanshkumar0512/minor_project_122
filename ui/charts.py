"""Plotly figures.  Every figure is rendered with the viewer's 'seaf_<mode>' template (ui/theme.py)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .theme import colors, plotly_template

CONFIG = {"displayModeBar": False, "responsive": True}
MONO = "JetBrains Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def show(fig: go.Figure, key: str | None = None) -> None:
    # Enforce the transparent, uncluttered look regardless of per-figure layout.
    fig.update_layout(template=plotly_template(), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, width="stretch", config=CONFIG, theme=None, key=key)


def _rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


# --------------------------------------------------------------------------- #
# Predict page
# --------------------------------------------------------------------------- #
def probability_gauge(p: float, class_names: Sequence[str], height: int = 220) -> go.Figure:
    c = colors()
    col = c["attack"] if p >= 0.5 else c["safe"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p * 100,
        number=dict(suffix="%", font=dict(family=MONO, size=38, color=c["text"])),
        gauge=dict(
            shape="angular",
            axis=dict(range=[0, 100], tickwidth=0, tickcolor=c["muted"], tickfont=dict(size=10, color=c["muted"]),
                      tickvals=[0, 50, 100]),
            bar=dict(color=col, thickness=0.28),
            bgcolor=c["surface2"], borderwidth=0,
            steps=[dict(range=[0, 50], color=_rgba(c["safe"], .10)), dict(range=[50, 100], color=_rgba(c["attack"], .10))],
            threshold=dict(line=dict(color=c["text2"], width=2), thickness=0.8, value=50),
        ),
        domain=dict(x=[0, 1], y=[0.08, 1]),
    ))
    fig.add_annotation(x=0.02, y=0.0, text=class_names[0], showarrow=False, font=dict(size=11, color=c["muted"]), xanchor="left")
    fig.add_annotation(x=0.98, y=0.0, text=class_names[1], showarrow=False, font=dict(size=11, color=c["muted"]), xanchor="right")
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=16, b=8))
    return fig


def shap_waterfall(phi: np.ndarray, labels: Sequence[str], values: Sequence[str], base: float,
                   p: float, top: int = 9, height: int | None = None) -> go.Figure:
    """Horizontal waterfall: base rate -> contributions -> predicted probability."""
    c = colors()
    phi = np.asarray(phi, float)
    order = np.argsort(-np.abs(phi))
    keep, rest = order[:top], order[top:]
    ys = [f"{labels[i]} = {values[i]}" for i in keep][::-1]
    xs = [float(phi[i]) for i in keep][::-1]
    if len(rest):
        ys = [f"{len(rest)} other features"] + ys
        xs = [float(phi[rest].sum())] + xs
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(xs) + ["total"],
        y=["Base rate"] + ys + ["Prediction"], x=[base] + xs + [0],
        base=0, connector=dict(line=dict(color=c["axis"], width=1, dash="dot")),
        increasing=dict(marker=dict(color=c["attack"])), decreasing=dict(marker=dict(color=c["safe"])),
        totals=dict(marker=dict(color=c["accent"])),
        text=[f"{base:.3f}"] + [f"{v:+.3f}" for v in xs] + [f"{p:.3f}"], textposition="outside",
        textfont=dict(family=MONO, size=11, color=c["text2"]),
        hovertemplate="%{y}<br>contribution %{x:+.4f}<extra></extra>",
    ))
    cum = np.cumsum([base] + xs)
    hi = max(float(cum.max()), p, base) + 0.12
    lo = min(0.0, float(cum.min()) - 0.05)
    fig.update_layout(height=height or 90 + 30 * (len(xs) + 2), showlegend=False,
                      xaxis=dict(title="probability of positive class", zeroline=False, range=[lo, min(hi, 1.15)]),
                      yaxis=dict(automargin=True), margin=dict(l=8, r=40, t=10, b=8))
    return fig


def trust_breakdown(C: float, A: float, S: float, T: float, threshold: float, height: int = 170) -> go.Figure:
    """Segmented bar: T is the sum of three equal-weight contributions."""
    c = colors()
    parts = [("Confidence  C/3", C / 3, c["series"][0]), ("Normal reasoning  (1−A)/3", (1 - A) / 3, c["series"][2]),
             ("Stable explanation  (1−S)/3", (1 - S) / 3, c["series"][3])]
    fig = go.Figure()
    for name, v, col in parts:
        fig.add_trace(go.Bar(y=["Trust"], x=[v], name=name, orientation="h", marker=dict(color=col, line=dict(color=c["surface"], width=2)),
                             hovertemplate=f"{name}: %{{x:.3f}}<extra></extra>", text=[f"{v:.2f}"], textposition="inside",
                             insidetextanchor="middle", textfont=dict(family=MONO, size=11, color="#fff")))
    fig.add_shape(type="line", x0=threshold, x1=threshold, y0=-0.5, y1=0.5, line=dict(color=c["review"], width=2, dash="dash"))
    fig.add_annotation(x=threshold, y=0.62, text=f"threshold {threshold:.3f}", showarrow=False,
                       font=dict(size=11, color=c["review"]))
    fig.add_annotation(x=T, y=-0.62, text=f"T = {T:.3f}", showarrow=False, xanchor="right" if T > 0.8 else "left",
                       font=dict(family=MONO, size=12, color=c["text"]))
    fig.update_layout(barmode="stack", height=height, xaxis=dict(range=[0, 1], dtick=0.25, title=None),
                      yaxis=dict(visible=False), legend=dict(y=-0.45, yanchor="top", x=0, xanchor="left"),
                      margin=dict(l=8, r=8, t=24, b=8))
    return fig


def signal_radial(C: float, A: float, S: float, ref: dict | None = None, height: int = 260) -> go.Figure:
    """Radar of the three 'good' directions; optional baseline-median overlay."""
    c = colors()
    cats = ["Confidence", "Normal reasoning", "Stable explanation"]
    fig = go.Figure()
    if ref:
        rv = [ref["C"], 1 - ref["A"], 1 - ref["S"]]
        fig.add_trace(go.Scatterpolar(r=rv + rv[:1], theta=cats + cats[:1], name="Typical clean",
                                      line=dict(color=c["muted"], width=1.5, dash="dot"), fill="none"))
    v = [C, 1 - A, 1 - S]
    fig.add_trace(go.Scatterpolar(r=v + v[:1], theta=cats + cats[:1], name="This decision", fill="toself",
                                  line=dict(color=c["accent"], width=2), fillcolor=_rgba(c["accent"], .18)))
    fig.update_layout(height=height, polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor=c["grid"]),
                                                angularaxis=dict(gridcolor=c["grid"], tickfont=dict(size=11, color=c["text2"]))),
                      margin=dict(l=30, r=30, t=30, b=20), legend=dict(y=-0.1, yanchor="top", x=0.5, xanchor="center"))
    return fig


def deviation_bars(top: list[dict], labels: dict | None = None, height: int = 170) -> go.Figure:
    """Mini bar chart of the top deviating features (baseline sigma)."""
    c = colors()
    labels = labels or {}
    top = list(reversed(top))
    ys = [labels.get(t["feature"], t["feature"]) for t in top]
    xs = [t["sigma"] for t in top]
    cols = [c["attack"] if abs(x) >= 3 else c["review"] if abs(x) >= 2 else c["muted"] for x in xs]
    fig = go.Figure(go.Bar(x=xs, y=ys, orientation="h", marker=dict(color=cols, cornerradius=4),
                           text=[f"{x:+.1f}σ" for x in xs], textposition="outside",
                           textfont=dict(family=MONO, size=11, color=c["text2"]),
                           hovertemplate="%{y}: %{x:+.2f}σ<extra></extra>"))
    m = max(3.5, max((abs(x) for x in xs), default=1) * 1.3)
    fig.update_layout(height=height, xaxis=dict(range=[-m, m], title=None, zeroline=True, zerolinecolor=c["axis"]),
                      yaxis=dict(automargin=True), margin=dict(l=4, r=4, t=4, b=4), showlegend=False)
    return fig


# --------------------------------------------------------------------------- #
# Attack simulator
# --------------------------------------------------------------------------- #
def shap_compare(phi0: np.ndarray, phi1: np.ndarray, labels: Sequence[str], top: int = 10) -> go.Figure:
    c = colors()
    phi0, phi1 = np.asarray(phi0, float), np.asarray(phi1, float)
    order = np.argsort(-(np.abs(phi0) + np.abs(phi1)))[:top][::-1]
    ys = [labels[i] for i in order]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=ys, x=phi0[order], name="Original", orientation="h",
                         marker=dict(color=c["muted"], cornerradius=3),
                         hovertemplate="%{y}<br>original %{x:+.4f}<extra></extra>"))
    fig.add_trace(go.Bar(y=ys, x=phi1[order], name="Attacked", orientation="h",
                         marker=dict(color=c["attack"], cornerradius=3),
                         hovertemplate="%{y}<br>attacked %{x:+.4f}<extra></extra>"))
    fig.update_layout(barmode="group", height=110 + 34 * len(ys), bargap=0.25, bargroupgap=0.08,
                      xaxis=dict(title="SHAP contribution", zeroline=True, zerolinecolor=c["axis"]),
                      yaxis=dict(automargin=True), margin=dict(l=8, r=8, t=30, b=8))
    return fig


def search_trace(trace: Sequence[float], height: int = 180) -> go.Figure:
    c = colors()
    fig = go.Figure(go.Scatter(y=list(trace), x=list(range(1, len(trace) + 1)), mode="lines",
                               line=dict(color=c["attack"], width=2, shape="hv"),
                               fill="tozeroy", fillcolor=_rgba(c["attack"], .10),
                               hovertemplate="candidate %{x}<br>best displacement %{y:.2f}σ<extra></extra>"))
    fig.update_layout(height=height, xaxis=dict(title="surviving candidate #"), yaxis=dict(title="best displacement (σ)"),
                      margin=dict(l=8, r=8, t=10, b=8), showlegend=False)
    return fig


def trust_compare(before: dict, after: dict, threshold: float, height: int = 240) -> go.Figure:
    c = colors()
    cats = ["C", "1 − A", "1 − S", "Trust T"]
    b = [before["C"], 1 - before["A"], 1 - before["S"], before["T"]]
    a = [after["C"], 1 - after["A"], 1 - after["S"], after["T"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=b, name="Original", marker=dict(color=c["muted"], cornerradius=4)))
    fig.add_trace(go.Bar(x=cats, y=a, name="Attacked", marker=dict(color=c["attack"], cornerradius=4)))
    fig.add_hline(y=threshold, line=dict(color=c["review"], width=1.5, dash="dash"),
                  annotation_text=f"threshold {threshold:.3f}", annotation_font=dict(color=c["review"], size=11))
    fig.update_layout(barmode="group", height=height, yaxis=dict(range=[0, 1.05], title=None),
                      margin=dict(l=8, r=8, t=30, b=8))
    return fig


# --------------------------------------------------------------------------- #
# Live monitor
# --------------------------------------------------------------------------- #
def trust_timeline(df: pd.DataFrame, threshold_series: Sequence[float], height: int = 280) -> go.Figure:
    c = colors()
    fig = go.Figure()
    if len(df):
        fig.add_trace(go.Scatter(x=df["n"], y=threshold_series, name="Threshold", mode="lines",
                                 line=dict(color=c["review"], width=1.5, dash="dash"), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=df["n"], y=df["T"].rolling(10, min_periods=1).mean(), name="Rolling mean",
                                 mode="lines", line=dict(color=c["accent"], width=2), hoverinfo="skip"))
        for lab, col, name, sym in (("clean", c["safe"], "Clean", "circle"), ("drift", c["drift"], "Drifted", "diamond"),
                                    ("attack", c["attack"], "Injected attack", "x")):
            d = df[df["label"] == lab]
            if len(d):
                fig.add_trace(go.Scatter(
                    x=d["n"], y=d["T"], mode="markers", name=name,
                    marker=dict(color=col, size=8 if lab != "clean" else 6, symbol=sym,
                                line=dict(color=c["surface"], width=1), opacity=0.9 if lab != "clean" else 0.55),
                    customdata=np.c_[d["verdict"]],
                    hovertemplate="#%{x} · T=%{y:.3f}<br>%{customdata[0]}<extra>" + name + "</extra>"))
    fig.update_layout(height=height, xaxis=dict(title="decision #"), yaxis=dict(title="trust T", range=[0, 1]),
                      margin=dict(l=8, r=8, t=30, b=8))
    return fig


def flag_rate_chart(hist: pd.DataFrame, expected: float, height: int = 250) -> go.Figure:
    """Short vs long flag rate, with drift / spike periods shaded."""
    c = colors()
    fig = go.Figure()
    if len(hist):
        # shade contiguous state regions
        for state, col in (("spike", c["attack"]), ("drift", c["drift"])):
            on = (hist["state"] == state).to_numpy()
            start = None
            for i, v in enumerate(np.r_[on, False]):
                if v and start is None:
                    start = i
                if not v and start is not None:
                    fig.add_vrect(x0=hist["n"].iloc[start] - 0.5, x1=hist["n"].iloc[i - 1] + 0.5,
                                  fillcolor=_rgba(col, .16), line_width=0, layer="below")
                    start = None
        fig.add_trace(go.Scatter(x=hist["n"], y=hist["short_rate"], name="Recent 20", mode="lines",
                                 line=dict(color=c["accent"], width=2)))
        fig.add_trace(go.Scatter(x=hist["n"], y=hist["long_rate"], name="Last 100", mode="lines",
                                 line=dict(color=c["muted"], width=2, dash="dot")))
        fig.add_hline(y=expected, line=dict(color=c["muted"], width=1, dash="dot"),
                      annotation_text="expected clean rate", annotation_font=dict(size=10, color=c["muted"]),
                      annotation_position="top left")
    fig.update_layout(height=height, xaxis=dict(title="decision #"), yaxis=dict(title="flag rate", range=[0, 1], tickformat=".0%"),
                      margin=dict(l=8, r=8, t=30, b=8))
    return fig


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def grouped_bars(cats: Sequence[str], series: dict[str, Sequence[float]], ref_line: float | None = 0.5,
                 y_title: str = "ROC-AUC", height: int = 300, errors: dict[str, Sequence[Sequence[float]]] | None = None,
                 y_range: Sequence[float] = (0, 1), text_fmt: str = "{:.3f}") -> go.Figure:
    c = colors()
    fig = go.Figure()
    n_ours = sum("paper" not in k.lower() for k in series)
    for i, (name, vals) in enumerate(series.items()):
        err = None
        if errors and name in errors:
            lo = [v - e[0] for v, e in zip(vals, errors[name])]
            hi = [e[1] - v for v, e in zip(vals, errors[name])]
            err = dict(type="data", array=hi, arrayminus=lo, color=c["text2"], thickness=1.2, width=4)
        # "(paper)" series reuse the hue of their "(ours)" counterpart, hatched and faded
        is_ref = "paper" in name.lower()
        slot = (i - n_ours) if is_ref else i
        fig.add_trace(go.Bar(x=list(cats), y=list(vals), name=name, error_y=err, opacity=0.55 if is_ref else 1,
                             marker=dict(color=c["series"][slot % len(c["series"])], cornerradius=4,
                                         pattern=dict(shape="/", fgcolor=c["surface"], size=6) if is_ref else None,
                                         line=dict(color=c["surface"], width=2)),
                             text=[text_fmt.format(v) for v in vals], textposition="outside",
                             textfont=dict(family=MONO, size=11, color=c["text2"]),
                             hovertemplate="%{x} · " + name + ": %{y:.3f}<extra></extra>"))
    if ref_line is not None:
        fig.add_hline(y=ref_line, line=dict(color=c["muted"], width=1, dash="dot"),
                      annotation_text="chance" if ref_line == 0.5 else "", annotation_font=dict(size=10, color=c["muted"]))
    fig.update_layout(barmode="group", height=height, yaxis=dict(range=list(y_range), title=y_title),
                      margin=dict(l=8, r=8, t=36, b=8))
    return fig


def line_chart(x: Sequence, ys: dict[str, Sequence[float]], x_title: str, y_title: str, height: int = 260,
               y_range: Sequence[float] | None = None, fmt: str | None = None) -> go.Figure:
    c = colors()
    fig = go.Figure()
    for i, (name, y) in enumerate(ys.items()):
        fig.add_trace(go.Scatter(x=list(x), y=list(y), name=name, mode="lines+markers",
                                 line=dict(color=c["series"][i % len(c["series"])], width=2),
                                 marker=dict(size=8, line=dict(color=c["surface"], width=2))))
    fig.update_layout(height=height, xaxis=dict(title=x_title, tickformat=fmt), yaxis=dict(title=y_title, range=y_range),
                      margin=dict(l=8, r=8, t=30, b=8), showlegend=len(ys) > 1)
    return fig


def distribution(clean: np.ndarray, attack: np.ndarray, threshold: float, height: int = 260) -> go.Figure:
    c = colors()
    fig = go.Figure()
    bins = dict(start=0, end=1, size=0.025)
    fig.add_trace(go.Histogram(x=clean, name="Clean", xbins=bins, marker=dict(color=_rgba(c["safe"], .6)), histnorm="probability"))
    fig.add_trace(go.Histogram(x=attack, name="Attacked", xbins=bins, marker=dict(color=_rgba(c["attack"], .6)), histnorm="probability"))
    fig.add_vline(x=threshold, line=dict(color=c["review"], width=2, dash="dash"),
                  annotation_text=f"threshold {threshold:.3f}", annotation_font=dict(color=c["review"], size=11))
    fig.update_layout(barmode="overlay", height=height, xaxis=dict(title="trust T", range=[0, 1]),
                      yaxis=dict(title="share", tickformat=".0%"), margin=dict(l=8, r=8, t=30, b=8))
    return fig


def donut(labels: Sequence[str], values: Sequence[float], cols: Sequence[str], height: int = 230) -> go.Figure:
    c = colors()
    fig = go.Figure(go.Pie(labels=list(labels), values=list(values), hole=0.66, sort=False,
                           marker=dict(colors=list(cols), line=dict(color=c["surface"], width=2)),
                           textinfo="none", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b>{int(sum(values))}</b><br><span style='font-size:11px'>decisions</span>",
                       showarrow=False, font=dict(family=MONO, size=18, color=c["text"]))
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8), legend=dict(orientation="v", x=1, y=0.5, xanchor="left"))
    return fig
