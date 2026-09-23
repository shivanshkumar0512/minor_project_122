"""Predict: score one record end to end and show the full SEAF breakdown."""
from __future__ import annotations

import json

import numpy as np
import streamlit as st

from ui import charts, layout, state
from ui import record as R
from ui.components import (callout, esc, fmt_pct, kpi_html, kpi_row, md, page_header, section, tip,
                           verdict_panel)

if not layout.guard():
    st.stop()

rt = state.runtime()
names = rt.schema.class_names
page_header(f"{rt.title} · {rt.subtitle}", "Predict & verify",
            f"Score a single record. SEAF shows the model's decision, its SHAP explanation and whether that explanation "
            f"looks like the model's normal reasoning {tip('trust')}.", "psychology")

ss = st.session_state
key = f"pred_{rt.key}"
opts = R.record_options(rt)
if f"{key}_x" not in ss:
    ss[f"{key}_x"] = rt.x_clean(opts[0])
    ss[f"{key}_v"] = 0


def load(x: np.ndarray, src: str) -> None:
    ss[f"{key}_x"] = x
    ss[f"{key}_v"] += 1
    ss[f"{key}_src"] = src
    ss.pop(f"{key}_res", None)


left, right = st.columns([1, 1.45], gap="large")

with left:
    with st.container(key="card-form"):
        st.markdown("##### Input record")
        b1, b2, b3 = st.columns(3)
        rng = np.random.default_rng()
        if b1.button("Clean", icon=":material/shuffle:", width="stretch"):
            i = int(rng.choice(opts))
            load(rt.x_clean(i), f"eval-{i}")
        if b2.button("Attacked", icon=":material/bug_report:", width="stretch",
                     help="A pre-computed prediction-preserving attack on a real record"):
            pairs = rt.attack_pairs()
            j = int(rng.choice(pairs.index))
            load(rt.x_adv(j), f"adv-{int(pairs.loc[j, 'eval_idx'])}")
        if b3.button("Reset", icon=":material/restart_alt:", width="stretch"):
            load(rt.x_clean(opts[0]), f"eval-{opts[0]}")
        src = ss.get(f"{key}_src", f"eval-{opts[0]}")
        tag = " · <span class='badge attack'>attacked sample</span>" if src.startswith("adv") else ""
        md(f"<div class='muted' style='margin:2px 0 8px'>Loaded: <span class='mono'>{esc(src)}</span>{tag}</div>")
        with st.form(f"{key}_form", border=False):
            x = R.feature_form(rt, ss[f"{key}_x"], key=f"{key}_{ss[f'{key}_v']}")
            submitted = st.form_submit_button("Score with SEAF", type="primary", icon=":material/bolt:", width="stretch")

if submitted:
    with right:
        ph = st.empty()
        with ph.container():
            st.skeleton(height=90)
            st.skeleton(height=220)
            st.skeleton(height=260)
        res = R.score(rt, x[None])
        rid = R.log_records(rt, res, "predict", [src], ["attack" if src.startswith("adv") else None])[0]
        ss[f"{key}_res"] = {"rec": res.record(0), "phi": res.phi[0], "x": x, "id": rid}
        ph.empty()
        if res.flagged[0]:
            st.toast(f"Decision #{rid} routed to review (T={res.T[0]:.3f})", icon=":material/flag:")

with right:
    out = ss.get(f"{key}_res")
    if not out:
        md("""<div class="empty"><div class="ic">bolt</div><h4>No decision yet</h4>
           <div>Edit the record (or load a random one) and press <b>Score with SEAF</b>. You'll see the prediction,
           its SHAP explanation and a trust verdict.</div></div>""")
    else:
        rec, phi, xv = out["rec"], out["phi"], out["x"]
        thr = rec["threshold"]
        verdict_panel(rec["verdict"], rec["T"], thr)
        st.write("")
        kpi_row([
            kpi_html("Prediction", names[rec["pred"]], f"audit #{out['id']}", "accent"),
            kpi_html(rt.verb.capitalize(), f"{rec['proba']:.3f}", None, "neutral"),
            kpi_html("Trust T", f"{rec['T']:.3f}", f"threshold {thr:.3f}",
                     "safe" if rec["verdict"] == "Accept" else "review", "", "trust"),
        ])
        g, t = st.columns([1, 1.25])
        with g:
            section("Probability")
            charts.show(charts.probability_gauge(rec["proba"], names), key="p_gauge")
        with t:
            section("Trust signals", "trust")
            charts.show(charts.signal_radial(rec["C"], rec["A"], rec["S"], R.baseline_reference(rt), 250), key="p_radar")
        section("Trust composition")
        md(f"<div class='muted'>C = {rec['C']:.3f}{tip('C')} &nbsp; A = {rec['A']:.3f}{tip('A')} &nbsp; "
           f"S = {rec['S']:.3f}{tip('S')}</div>")
        charts.show(charts.trust_breakdown(rec["C"], rec["A"], rec["S"], rec["T"], thr), key="p_trust")

        section("Why? SHAP explanation", "SHAP")
        charts.show(charts.shap_waterfall(phi, R.labels(rt), R.display_values(rt, xv), rec["base_value"], rec["proba"]),
                    key="p_wf")
        err = abs(rec["base_value"] + float(np.sum(phi)) - rec["proba"])
        callout(f"Summation identity: base rate <span class='mono'>{rec['base_value']:.4f}</span> + Σφ "
                f"<span class='mono'>{float(np.sum(phi)):+.4f}</span> = <span class='mono'>{rec['proba']:.4f}</span> "
                f"(error {err:.1e}). Red bars push towards <b>{names[1]}</b>, green towards <b>{names[0]}</b>.")

        section("Audit evidence: most deviating features", "sigma")
        charts.show(charts.deviation_bars(rec["top_deviations"], R.label_map(rt), 210), key="p_dev")
        with st.expander("Full audit record (JSON)", icon=":material/data_object:"):
            st.code(json.dumps(rec, indent=2), language="json")
        if rec["verdict"] == "Review":
            st.page_link("app_pages/review.py", label="Open review queue", icon=":material/fact_check:")
