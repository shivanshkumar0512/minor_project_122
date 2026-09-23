"""Attack Simulator: run a live prediction-preserving or evasion attack."""
from __future__ import annotations

import time

import numpy as np
import streamlit as st

from seaf.attack import evasion_attack, prediction_preserving_attack
from ui import charts, layout, state
from ui import record as R
from ui.components import (badge, callout, esc, kpi_html, kpi_row, md, page_header, section, tip,
                           verdict_badge)

if not layout.guard():
    st.stop()

rt = state.runtime()
names = rt.schema.class_names
page_header(f"{rt.title} · red team", "Attack simulator",
            f"Act as the adversary. A <b>prediction-preserving</b> attack{tip('pp_attack')} keeps the decision but "
            f"rewrites its reasons; an <b>evasion</b> attack{tip('evasion')} flips the decision with the smallest nudge. "
            "Then see whether SEAF notices.", "crisis_alert")

ss = st.session_state
opts = R.record_options(rt)

with st.container(key="card-atk-setup"):
    c1, c2, c3 = st.columns([1.3, 1.1, 1.4], gap="large")
    with c1:
        st.markdown("##### 1 · Pick a record")
        if st.button("Random record", icon=":material/shuffle:"):
            ss["atk_idx"] = int(np.random.default_rng().choice(opts))
        cur = ss.get("atk_idx", opts[0])
        idx = st.selectbox("Evaluation record", opts, index=opts.index(cur) if cur in opts else 0,
                           format_func=lambda i: R.record_caption(rt, i), label_visibility="collapsed")
        ss["atk_idx"] = idx
    with c2:
        st.markdown("##### 2 · Choose the attack")
        kind = st.segmented_control("Attack type", ["Prediction-preserving", "Evasion"], default="Prediction-preserving",
                                    label_visibility="collapsed", key="atk_kind") or "Prediction-preserving"
        numeric = [rt.schema.label(f) for f in rt.schema.numeric_features]
        md(f"<div class='muted'>Perturbs {len(numeric)} continuous features: {esc(', '.join(numeric[:6]))}"
           f"{'…' if len(numeric) > 6 else ''}</div>")
    with c3:
        st.markdown("##### 3 · Budget")
        if kind == "Prediction-preserving":
            budget = st.slider("Max change per feature", 0.05, 0.50, 0.35, 0.05, format="±%.2f",
                               help="Paper setting: ±35%")
            n_cand = st.slider("Candidates", 10, 120, 40, 10, help="Paper setting: 40 random candidates")
            tau = st.slider("Max probability change |Δp|", 0.01, 0.15, 0.06, 0.01, help="Paper setting: 0.06")
        else:
            budget, n_cand, tau = 0.75, 300, None
            md("<div class='muted'>Searches radii 2% → 75% with 300 candidates each and keeps the smallest "
               "perturbation that flips the decision.</div>")
    run = st.button("Launch attack", type="primary", icon=":material/rocket_launch:")

if run:
    x0 = rt.x_clean(idx)
    seaf = rt.seaf
    with st.status("Running attack…", expanded=True) as status:
        bar = st.progress(0.0, "Generating candidates")
        for k in range(1, 5):  # visual pacing only; the real work below is vectorised
            time.sleep(0.12)
            bar.progress(k * 0.1, f"Generating and screening candidates with predict_proba ({k * 25}%)")
        t0 = time.perf_counter()
        if kind == "Prediction-preserving":
            atk = prediction_preserving_attack(rt.model, seaf.explainer, seaf.detector.standardiser.scale_, x0,
                                               seaf.numeric_idx, rt.schema.integer_idx, n_candidates=n_cand,
                                               budget=budget, tau=tau, seed=int(time.time()) % 10_000)
            bar.progress(0.6, f"{atk.n_survivors}/{n_cand} candidates kept the decision within |Δp| ≤ {tau:.2f}; "
                              "explaining survivors with TreeSHAP")
        else:
            atk = evasion_attack(rt.model, x0, seaf.numeric_idx, rt.schema.integer_idx,
                                 seed=int(time.time()) % 10_000)
            bar.progress(0.6, f"Screened {atk.n_candidates} candidates; {atk.n_survivors} flip the decision")
        bar.progress(0.8, "Scoring original and attacked record with SEAF")
        res = R.score(rt, np.vstack([x0, atk.x_adv]))
        bar.progress(1.0, "Done")
        dt = time.perf_counter() - t0
        status.update(label=f"Attack {'succeeded' if atk.success else 'failed'} in {dt:.2f}s",
                      state="complete" if atk.success else "error", expanded=False)
    ss["atk_out"] = {"atk": atk, "rec0": res.record(0), "rec1": res.record(1), "phi0": res.phi[0],
                     "phi1": res.phi[1], "kind": kind, "idx": idx, "logged": None}
    if atk.success and res.flagged[1]:
        st.toast("SEAF flagged the attacked decision", icon=":material/gpp_maybe:")
    elif atk.success:
        st.toast("The attack slipped past the current threshold", icon=":material/warning:")

out = ss.get("atk_out")
if not out:
    md("""<div class="empty"><div class="ic">crisis_alert</div><h4>No attack run yet</h4>
       <div>Pick a record and launch an attack. You'll see the original and attacked decisions side by side.</div></div>""")
    st.stop()

atk, r0, r1 = out["atk"], out["rec0"], out["rec1"]
if not atk.success:
    callout("No candidate satisfied the constraints for this record. Try a larger budget, more candidates, or "
            "another record.", warn=True)
    st.stop()

thr = r0["threshold"]
section("Outcome")
kpi_row([
    kpi_html("Decision", names[r1["pred"]],
             "unchanged" if r0["pred"] == r1["pred"] else f"flipped from {names[r0['pred']]}", "safe" if r0["pred"] == r1["pred"] else "attack"),
    kpi_html("Probability change", f"{r1['proba'] - r0['proba']:+.3f}", f"{r0['proba']:.3f} → {r1['proba']:.3f}", "neutral"),
    kpi_html("Trust change", f"{r1['T'] - r0['T']:+.3f}", f"{r0['T']:.3f} → {r1['T']:.3f}",
             "attack" if r1["T"] < r0["T"] else "safe", "down" if r1["T"] < r0["T"] else "up", "trust"),
    kpi_html("Attribution displacement", f"{atk.displacement:.2f}σ" if atk.displacement else "—",
             f"mean input change {100 * atk.perturbation:.1f}%", "review", "", "sigma"),
])

a, b = st.columns(2, gap="medium")
for col, rec, title, tone in ((a, r0, "Original", "neutral"), (b, r1, "Attacked", "attack")):
    with col:
        with st.container(key=f"card-{'attack' if tone == 'attack' else 'orig'}-side"):
            md(f"<div style='display:flex;justify-content:space-between;align-items:center'>"
               f"<h4 style='margin:0'>{title}</h4>{verdict_badge(rec['verdict'])}</div>"
               f"<div class='muted' style='margin-top:6px'>Prediction <b>{names[rec['pred']]}</b> · p = "
               f"<span class='mono'>{rec['proba']:.3f}</span> · T = <span class='mono'>{rec['T']:.3f}</span> · "
               f"C {rec['C']:.2f} · A {rec['A']:.2f} · S {rec['S']:.2f}</div>")
            charts.show(charts.deviation_bars(rec["top_deviations"], R.label_map(rt), 190), key=f"dev_{title}")

if r0["verdict"] == "Accept" and r1["verdict"] == "Review":
    callout(f"<b>Caught.</b> The decision stayed <b>{names[r1['pred']]}</b>, but SEAF saw that the explanation no "
            f"longer looks like normal reasoning and routed it to review (trust {r1['T']:.3f} < {thr:.3f}).")
elif r1["verdict"] == "Accept":
    callout(f"<b>Missed at the current threshold</b> ({thr:.3f}). Detection is probabilistic; the paper reports "
            "AUCs of 0.6-0.87, which supports triage rather than automatic rejection. Reviewer feedback can raise "
            "the threshold.", warn=True)
else:
    callout("Both decisions fall below the threshold: the original was already low-trust.", warn=True)

c1, c2 = st.columns([1.2, 1], gap="medium")
with c1:
    section("SHAP before vs after", "SHAP")
    charts.show(charts.shap_compare(out["phi0"], out["phi1"], R.labels(rt)), key="atk_cmp")
with c2:
    section("Trust signals before vs after", "trust")
    charts.show(charts.trust_compare(r0, r1, thr), key="atk_trust")
    if atk.trace:
        section("Random search progress")
        charts.show(charts.search_trace(atk.trace), key="atk_trace")

section("What changed")
md(R.diff_table(rt, atk.x_orig, atk.x_adv, out["phi0"], out["phi1"]))
st.write("")
if out["logged"] is None:
    if st.button("Send attacked decision to the audit log / review queue", icon=":material/move_to_inbox:"):
        res = R.score(rt, atk.x_adv[None])
        out["logged"] = R.log_records(rt, res, "attack", [f"sim-{out['idx']}"], ["attack"])[0]
        st.toast(f"Logged as audit #{out['logged']}" + (" · pending review" if res.flagged[0] else ""),
                 icon=":material/inbox:")
        st.rerun()
else:
    md(f"{badge('Logged as audit #' + str(out['logged']), 'accent')}")
