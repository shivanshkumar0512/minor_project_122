"""Record-level helpers: input forms, scoring, display and audit logging."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from seaf.audit import AuditLog

from . import state


def score(rt: state.DomainRuntime, X: np.ndarray, stability: bool | np.ndarray = True):
    """Score with the *current* shared threshold (never mutates the cached SEAF)."""
    return rt.seaf.score(np.atleast_2d(X), stability=stability, threshold=state.threshold(rt.key))


def log_records(rt: state.DomainRuntime, res, source: str, refs: list[str] | None = None,
                sim_labels: list[str | None] | None = None) -> list[int]:
    rows = []
    for i in range(len(res)):
        rows.append(AuditLog.make_row(rt.key, source, res.record(i),
                                      sim_label=sim_labels[i] if sim_labels else None,
                                      record_ref=refs[i] if refs else None))
    return state.db().log(rows)


def display_values(rt: state.DomainRuntime, x: np.ndarray) -> list[str]:
    s = rt.schema
    return [s.display_value(f, v) for f, v in zip(s.feature_names, x)]


def labels(rt: state.DomainRuntime) -> list[str]:
    return [rt.schema.label(f) for f in rt.schema.feature_names]


def label_map(rt: state.DomainRuntime) -> dict[str, str]:
    return {f: rt.schema.label(f) for f in rt.schema.feature_names}


def record_options(rt: state.DomainRuntime, limit: int = 400) -> list[int]:
    """Evaluation-split rows exposed in the UI (tuning half only; held-out stays hidden)."""
    return [int(i) for i in rt.tune_idx[:limit]]


def record_caption(rt: state.DomainRuntime, i: int) -> str:
    y = int(rt.bundle["y_eval"][i])
    return f"Record #{i} · actual: {rt.schema.class_names[y]}"


def feature_form(rt: state.DomainRuntime, x0: np.ndarray, key: str, cols: int = 2) -> np.ndarray:
    """Widgets for every feature, initialised from ``x0``; returns the edited vector."""
    s = rt.schema
    out = np.array(x0, dtype=float)
    grid = st.columns(cols)
    for j, f in enumerate(s.feature_names):
        lab = s.label(f)
        v = float(x0[j])
        with grid[j % cols]:
            wkey = f"{key}_{f}"
            if f in s.categorical_maps:
                cats = s.categorical_maps[f]
                idx = int(np.clip(round(v), 0, len(cats) - 1))
                out[j] = float(cats.index(st.selectbox(lab, cats, index=idx, key=wkey)))
            elif f in s.choices:
                opts = list(s.choices[f])
                if v not in opts:
                    opts = sorted(set(opts) | {v})
                vl = s.value_labels.get(f, {})
                out[j] = float(st.selectbox(lab, opts, index=opts.index(v), key=wkey,
                                            format_func=lambda o, vl=vl: vl.get(int(o), f"{o:g}")))
            elif f in s.display_units:
                factor, unit = s.display_units[f]
                shown = st.number_input(f"{lab.split(' (')[0]} ({unit})", value=round(v * factor, 1), step=1.0,
                                        format="%.1f", key=wkey)
                out[j] = round(shown / factor)
            else:
                is_int = f in s.integer_features
                lo, hi = s.ranges.get(f, (None, None))
                step = 1.0 if is_int else max(abs(hi - lo) / 100, 0.01) if lo is not None else 1.0
                if is_int and hi is not None and hi - lo > 1e5:
                    step = float(10 ** int(np.log10(max(hi - lo, 1)) - 2))
                out[j] = st.number_input(lab, value=float(v), step=float(step), format="%.0f" if is_int else "%.2f",
                                         key=wkey, help=f"Observed range {lo:,.0f} – {hi:,.0f}" if lo is not None else None)
    return out


def diff_table(rt: state.DomainRuntime, x0: np.ndarray, x1: np.ndarray, phi0: np.ndarray, phi1: np.ndarray) -> str:
    """HTML table of feature changes and attribution shifts, changed rows highlighted."""
    s = rt.schema
    rows = []
    order = np.argsort(-np.abs(np.asarray(phi1) - np.asarray(phi0)))
    for j in order:
        f = s.feature_names[j]
        a, b = float(x0[j]), float(x1[j])
        changed = abs(a - b) > 1e-9
        rel = (b - a) / abs(a) * 100 if changed and abs(a) > 1e-9 else 0.0
        dphi = float(phi1[j] - phi0[j])
        cls = "changed" if changed else ""
        arrow = f'<span class="{"up" if rel > 0 else "dn"}">{rel:+.1f}%</span>' if changed else "—"
        rows.append(f"<tr class='{cls}'><td>{s.label(f)}</td><td class='num'>{s.display_value(f, a)}</td>"
                    f"<td class='num'>{s.display_value(f, b)}</td><td class='num'>{arrow}</td>"
                    f"<td class='num'>{phi0[j]:+.4f}</td><td class='num'>{phi1[j]:+.4f}</td>"
                    f"<td class='num'>{dphi:+.4f}</td></tr>")
    return ("<table class='diff'><thead><tr><th>Feature</th><th style='text-align:right'>Original</th>"
            "<th style='text-align:right'>Attacked</th><th style='text-align:right'>Change</th>"
            "<th style='text-align:right'>SHAP before</th><th style='text-align:right'>SHAP after</th>"
            "<th style='text-align:right'>Δ SHAP</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def baseline_reference(rt: state.DomainRuntime) -> dict:
    b = rt.seaf.baseline_
    return {"C": float(np.median(b["C"])), "A": float(np.median(b["A"])), "S": float(np.median(b["S"]))}


def records_frame(rt: state.DomainRuntime, x: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([x], columns=rt.schema.feature_names)
