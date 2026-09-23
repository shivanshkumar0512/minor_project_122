"""Home: hero, animated pipeline, domain cards, key stats and the guided demo."""
from __future__ import annotations

import json

import numpy as np
import streamlit as st

from seaf.config import DOMAINS, RESULTS_DIR
from ui import charts, layout, state
from ui import record as R
from ui.components import badge, callout, esc, fmt_pct, kpi_html, kpi_row, md, section, tip, verdict_panel

if not layout.guard():
    st.stop()


@st.cache_data(show_spinner=False)
def experiments() -> dict | None:
    p = RESULTS_DIR / "experiments.json"
    return json.loads(p.read_text()) if p.exists() else None


# --------------------------------------------------------------------------- #
# Hero + pipeline
# --------------------------------------------------------------------------- #
STAGES = [
    ("01", "Input", "Loan, patient or employee record"),
    ("02", "Model", "Random Forest makes the decision"),
    ("03", "SHAP", "Exact per-feature receipt of <i>why</i>"),
    ("04", "Validation", "Isolation Forest vs normal reasoning"),
    ("05", "Trust", "Confidence · anomaly · stability"),
]
pipe = []
for i, (n, h, d) in enumerate(STAGES):
    pipe.append(f'<div class="stage"><div class="n">{n}</div><div class="h">{h}</div><div class="d">{d}</div></div>')
    pipe.append('<div class="link"></div>')
pipe.append('<div class="stage final"><div class="n">06</div><div class="h">Decision</div>'
            f'<div class="split">{badge("Accept", "safe")}{badge("Review", "review")}</div></div>')

md(f"""<div class="hero">
  <span class="eyebrow"><span class="material-symbols-rounded" style="font-size:15px">shield</span>
  Explanation-space anomaly detection</span>
  <h1>Don't just explain AI decisions. <span class="grad">Verify the explanation.</span></h1>
  <p>SEAF treats every SHAP explanation as a security signal: if a decision's stated reasons don't look like the
  model's normal reasoning, it is routed to a human instead of being silently accepted.</p>
  <div class="pipeline">{''.join(pipe)}</div>
</div>""")

c1, c2, c3, _ = st.columns([1.2, 1, 1, 2])
with c1:
    if st.button("Start guided demo", type="primary", icon=":material/play_circle:", width="stretch"):
        st.session_state["demo_step"] = 0
        st.session_state.pop("demo", None)
with c2:
    st.page_link("app_pages/monitor.py", label="Live Monitor", icon=":material/monitor_heart:")
with c3:
    st.page_link("app_pages/attack.py", label="Attack Simulator", icon=":material/crisis_alert:")


# --------------------------------------------------------------------------- #
# Guided demo
# --------------------------------------------------------------------------- #
STEPS = ["Normal decision", "Attack", "SEAF flags it", "Reviewer confirms", "Threshold updates"]


def run_demo() -> None:
    rt = state.runtime()
    step = st.session_state.get("demo_step", 0)
    demo = st.session_state.setdefault("demo", {})
    md('<div class="section-title">Guided demo · ' + esc(rt.title) + '</div>')
    md('<div class="stepper">' + "".join(
        f'<div class="step {"done" if i < step else "now" if i == step else ""}"><b>STEP {i + 1}</b>{s}</div>'
        for i, s in enumerate(STEPS)) + "</div>")

    idx = rt.bundle.get("demo_idx")
    pool = rt.bundle["attack_pool"]
    prow = pool.index[(pool.eval_idx == idx) & pool.success]
    if idx is None or not len(prow):
        callout("No demo pair available for this domain - try another domain.", warn=True)
        return
    if "clean" not in demo:
        with st.spinner("Scoring the demo pair with SEAF…"):
            x0, x1 = rt.x_clean(idx), rt.x_adv(int(prow[0]))
            r = R.score(rt, np.vstack([x0, x1]))
            demo.update(clean=r.record(0), attack=r.record(1), x0=x0, x1=x1,
                        phi0=r.phi[0], phi1=r.phi[1], threshold=r.threshold)
    c, a = demo["clean"], demo["attack"]
    labs = R.labels(rt)
    names = rt.schema.class_names
    thr = demo["threshold"]

    with st.container(key="card-demo"):
        if step == 0:
            st.markdown(f"#### A legitimate {rt.subtitle.lower()} request arrives")
            l, r = st.columns([1, 1.3])
            with l:
                charts.show(charts.probability_gauge(c["proba"], names, 200), key="demo_g0")
                verdict_panel(c["verdict"], c["T"], thr)
            with r:
                charts.show(charts.shap_waterfall(np.array(list(c["phi"].values())), labs,
                                                  R.display_values(rt, demo["x0"]), c["base_value"], c["proba"], top=7))
            callout(f"The model predicts <b>{names[c['pred']]}</b> ({rt.verb} {c['proba']:.3f}). "
                    f"The SHAP receipt adds up exactly: base rate {c['base_value']:.3f} + contributions = {c['proba']:.3f}. "
                    f"Trust <b>{c['T']:.3f}</b> is above the threshold, so the decision is accepted automatically.")
        elif step == 1:
            st.markdown("#### An adversary edits the inputs, keeping the decision but changing its reasons")
            dp = a["proba"] - c["proba"]
            kpi_row([
                kpi_html("Prediction", names[a["pred"]], "unchanged", "safe"),
                kpi_html("Probability change", f"{dp:+.3f}", "within the ±0.06 budget", "neutral"),
                kpi_html("Features edited", str(int(np.sum(np.abs(demo['x1'] - demo['x0']) > 1e-9))), "each within ±35%", "review"),
            ])
            charts.show(charts.shap_compare(demo["phi0"], demo["phi1"], labs, top=8))
            callout("An output-only audit sees the <b>same decision</b>. But the explanation, the recorded "
                    "justification, has been rearranged. This is a <b>prediction-preserving attack</b>." + tip("pp_attack"))
        elif step == 2:
            st.markdown("#### SEAF compares the new explanation with normal model reasoning")
            l, r = st.columns([1, 1])
            with l:
                verdict_panel(a["verdict"], a["T"], thr)
                charts.show(charts.trust_compare(c, a, thr, 230))
            with r:
                md('<div class="muted" style="margin-bottom:4px">Top deviating features (σ from normal)' + tip("sigma") + '</div>')
                charts.show(charts.deviation_bars(a["top_deviations"], R.label_map(rt), 220))
            if "logged_id" not in demo:
                demo["logged_id"] = R.log_records(rt, R.score(rt, demo["x1"][None]), "demo",
                                                  ["demo-attack"], ["attack"])[0]
            callout(f"Trust fell from <b>{c['T']:.3f}</b> to <b>{a['T']:.3f}</b>. The decision has been placed in the "
                    f"review queue (audit #{demo['logged_id']}) with the per-feature evidence above.")
        elif step == 3:
            st.markdown("#### A human reviewer examines the audit record")
            md(f"<div class='muted'>Audit record #{demo.get('logged_id')} · {esc(rt.title)} · trust "
               f"<span class='mono'>{a['T']:.3f}</span></div>")
            charts.show(charts.deviation_bars(a["top_deviations"], R.label_map(rt), 200))
            done = demo.get("confirmed")
            if not done:
                if st.button("Confirm attack", type="primary", icon=":material/gpp_bad:"):
                    state.db().review(demo["logged_id"], "confirmed_attack", "guided demo")
                    old, cal = state.recalibrate_domain(rt, "guided demo: confirmed attack")
                    old_det, old_fpr = state.heldout_rates(rt, old)
                    demo.update(confirmed=True, old_t=old, cal=cal.as_dict(), old_det=old_det, old_fpr=old_fpr)
                    st.toast("Attack confirmed - threshold recalibrated", icon=":material/verified_user:")
                    st.session_state["demo_step"] = 4
                    st.rerun()
            else:
                callout("Confirmed. Continue to see the effect on the detection threshold.")
        else:
            cal = demo.get("cal")
            if not cal:
                callout("Confirm the attack in step 4 first.", warn=True)
            else:
                st.markdown("#### The confirmed outcome recalibrates the threshold")
                dt = cal["threshold"] - demo["old_t"]
                kpi_row([
                    kpi_html("Threshold", f"{cal['threshold']:.3f}", f"{dt:+.3f} vs {demo['old_t']:.3f}", "accent",
                             "up" if dt > 0 else "down" if dt < 0 else "", "threshold"),
                    kpi_html("Held-out detection", fmt_pct(cal["heldout_detection"]),
                             f"{100 * (cal['heldout_detection'] - demo['old_det']):+.1f} pp", "safe", "up", "detection"),
                    kpi_html("Held-out false positives", fmt_pct(cal["heldout_fpr"]),
                             f"{100 * (cal['heldout_fpr'] - demo['old_fpr']):+.1f} pp", "review", "warn", "fpr"),
                ])
                callout("The threshold is re-chosen to catch the most attacks within a 10% false-alarm budget, and the "
                        "result is measured on <b>held-out</b> data never used to choose it. The baseline of normal "
                        "reasoning itself is <b>never</b> changed automatically.")

    b1, b2, _, b3 = st.columns([1, 1, 3, 1])
    with b1:
        if st.button("Back", icon=":material/arrow_back:", disabled=step == 0, width="stretch"):
            st.session_state["demo_step"] = step - 1
            st.rerun()
    with b2:
        if step < 4 and st.button("Next", type="primary", icon=":material/arrow_forward:", width="stretch",
                                  disabled=step == 3 and not demo.get("confirmed")):
            st.session_state["demo_step"] = step + 1
            st.rerun()
    with b3:
        if st.button("Close demo", width="stretch"):
            st.session_state.pop("demo_step", None)
            st.session_state.pop("demo", None)
            st.rerun()


if "demo_step" in st.session_state:
    run_demo()

# --------------------------------------------------------------------------- #
# Domains
# --------------------------------------------------------------------------- #
section("Choose a domain")
doms = [d for d in state.available_domains()]
cols = st.columns(len(doms))
for col, key in zip(cols, doms):
    m = state.meta(key) or {}
    mt = m.get("metrics", {})
    spec = DOMAINS.get(key)
    active = key == state.current_domain()
    icon = spec.icon if spec else "upload_file"
    with col:
        md(f"""<div class="domain-card {'active' if active else ''}">
            <div class="ic">{icon}</div>
            <div class="t">{esc(state.domain_title(key))}</div>
            <div class="s">{esc(spec.subtitle if spec else m.get('subtitle', 'Custom dataset'))} · {m.get('n_rows', 0):,} records</div>
            <div class="stats"><div><b>{mt.get('accuracy', 0):.3f}</b>accuracy</div>
            <div><b>{mt.get('roc_auc', 0):.3f}</b>ROC-AUC</div>
            <div><b>{m.get('n_features', 0)}</b>features</div></div></div>""")
        st.button("Active" if active else "Use this domain", key=f"use_{key}", disabled=active, width="stretch",
                  on_click=state.set_domain, args=(key,))

# --------------------------------------------------------------------------- #
# Key stats
# --------------------------------------------------------------------------- #
section("Key results", "auc")
ex = experiments()
counts = state.db().counts()
if ex:
    cd, d = ex["cross_domain"], ex["domains"]
    fin, hc = d.get("finance", {}), d.get("healthcare", {})
    pooled = cd["feedback_pooled"]["heldout"]
    kpi_row([
        kpi_html("Prediction-preserving attacks that succeeded", f"{cd['attack_success']}/{cd['attack_attempted']}",
                 "decision unchanged, reasons displaced", "attack", "", "pp_attack"),
        kpi_html("Healthcare trust-score AUC", f"{hc.get('signal_auc', {}).get('trust_equal', 0):.3f}",
                 "equal-weight C · A · S", "accent", "", "auc"),
        kpi_html("Stability AUC spread (fixed → |deviation|)",
                 f"{cd['stability']['fixed_range']:.2f} → {cd['stability']['absolute_range']:.2f}",
                 "cross-domain range", "drift"),
        kpi_html("Held-out detection after recalibration", fmt_pct(pooled["after"]["detection"]),
                 f"from {fmt_pct(pooled['before']['detection'])} (pooled)", "safe", "up", "detection"),
        kpi_html("Decisions in audit log", f"{counts.get('total', 0):,}",
                 f"{counts.get('pending', 0)} awaiting review", "neutral"),
    ])
else:
    callout("Run <code>python scripts/run_experiments.py</code> to populate paper results.", warn=True)

with st.expander("How to read SEAF (plain language)", icon=":material/help:"):
    st.markdown(
        "- **SHAP** splits every prediction into per-feature contributions that add up *exactly* to the model's output: "
        "a receipt for the decision.\n"
        "- **Normal reasoning baseline**: SHAP receipts of clean historical decisions. An *Isolation Forest* learns "
        "what those receipts usually look like.\n"
        "- **Trust score T** averages three signals: the model's **confidence**, how **normal** the receipt looks, and "
        "how **stable** the receipt is when inputs are nudged by ±2%.\n"
        "- **Low trust → human review.** The reviewer's verdict tunes the alert threshold (never the baseline).\n"
        "- **σ deviation**: how many standard deviations a feature's contribution is from its usual value.")
