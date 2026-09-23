"""Review Queue: adjudicate flagged decisions; outcomes recalibrate the threshold."""
from __future__ import annotations

import json
import time

import streamlit as st

from seaf.config import FPR_BUDGET
from ui import charts, layout, state
from ui import record as R
from ui.components import badge, callout, esc, fmt_pct, kpi_html, kpi_row, md, page_header, tip

if not layout.guard():
    st.stop()

rt = state.runtime()
ss = st.session_state
names = rt.schema.class_names
db = state.db()

page_header(f"{rt.title} · human in the loop", "Review queue",
            f"Low-trust decisions wait here with their audit evidence. Each verdict re-tunes the alert threshold "
            f"{tip('threshold')} and is measured on held-out data. The normal-reasoning baseline is never changed "
            "automatically.", "fact_check")

budget = st.session_state.get("fpr_budget", FPR_BUDGET)


def act(decision_id: int, outcome: str) -> None:
    """Button callback: store the verdict, recalibrate, remember before/after."""
    before_t = state.threshold(rt.key)
    before_det, before_fpr = state.heldout_rates(rt, before_t)
    db.review(decision_id, outcome, "reviewer")
    reason = f"review #{decision_id}: {'confirmed attack' if outcome == 'confirmed_attack' else 'approved'}"
    _, cal = state.recalibrate_domain(rt, reason, fpr_budget=ss.get("fpr_budget", FPR_BUDGET))
    ss["review_delta"] = {"id": decision_id, "outcome": outcome, "t0": before_t, "t1": cal.threshold,
                          "d0": before_det, "d1": cal.heldout_detection, "f0": before_fpr, "f1": cal.heldout_fpr,
                          "ts": time.time()}
    st.toast(("Attack confirmed" if outcome == "confirmed_attack" else "Decision approved") +
             f" · threshold {before_t:.3f} → {cal.threshold:.3f}",
             icon=":material/gpp_bad:" if outcome == "confirmed_attack" else ":material/check_circle:")


# --------------------------------------------------------------------------- #
# KPIs with before/after deltas
# --------------------------------------------------------------------------- #
thr = state.threshold(rt.key)
det, fpr = state.heldout_rates(rt, thr)
counts = db.counts(rt.key)
d = ss.get("review_delta")
fresh = d is not None and time.time() - d["ts"] < 600


def delta(new: float, old: float, pct: bool) -> tuple[str, str]:
    diff = new - old
    if abs(diff) < 1e-12:
        return "no change", ""
    return (f"{100 * diff:+.1f} pp" if pct else f"{diff:+.3f}"), ("up" if diff > 0 else "down")


t_d, t_c = delta(d["t1"], d["t0"], False) if fresh else (None, "")
d_d, d_c = delta(d["d1"], d["d0"], True) if fresh else (None, "")
f_d, f_c = delta(d["f1"], d["f0"], True) if fresh else (None, "")
kpi_row([
    kpi_html("Pending review", f"{counts.get('pending', 0)}", f"{counts.get('total', 0)} decisions logged", "review"),
    kpi_html("Threshold", f"{thr:.3f}", t_d or "flag when T < threshold", "accent", t_c, "threshold"),
    kpi_html("Held-out detection", fmt_pct(det), d_d or "attacks caught", "safe", d_c, "detection"),
    kpi_html("Held-out false positives", fmt_pct(fpr), f_d or f"budget {fmt_pct(budget, 0)}", "attack",
             "warn" if f_c == "up" else f_c, "fpr"),
    kpi_html("Reviewed", f"{counts.get('approved', 0) + counts.get('confirmed_attack', 0)}",
             f"{counts.get('confirmed_attack', 0)} confirmed attacks", "neutral"),
])
if fresh:
    verb = "Confirmed attack" if d["outcome"] == "confirmed_attack" else "Approved"
    callout(f"<b>{verb} #{d['id']}.</b> Threshold {d['t0']:.3f} → <b>{d['t1']:.3f}</b>; held-out detection "
            f"{fmt_pct(d['d0'])} → <b>{fmt_pct(d['d1'])}</b>; false positives {fmt_pct(d['f0'])} → "
            f"<b>{fmt_pct(d['f1'])}</b>. Chosen to maximise detection within the {fmt_pct(budget, 0)} false-alarm "
            "budget on the tuning pool; measured on held-out records never used for selection.")

# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
f1, f2, f3, f4 = st.columns([1.3, 1, 1, 1.2])
with f1:
    src = st.selectbox("Source", ["All", "stream", "predict", "attack", "demo", "seed"], key="rq_src")
with f2:
    order = st.selectbox("Sort", ["Lowest trust first", "Newest first"], key="rq_sort")
with f3:
    show_truth = st.toggle("Show simulation label", value=True, key="rq_truth",
                           help="Ground truth is known only because traffic is simulated. Hide it to review blind.")
with f4:
    st.slider("False-alarm budget", 0.02, 0.30, FPR_BUDGET, 0.01, format="%.2f", key="fpr_budget",
              help="Review capacity: maximum share of clean decisions that may be flagged")

pending = db.query(domain=rt.key, status="pending", source=None if src == "All" else src, limit=500)
if order == "Lowest trust first":
    pending = pending.sort_values("T")

if pending.empty:
    md("""<div class="empty"><div class="ic">task_alt</div><h4>Queue is clear</h4>
       <div>No decisions are waiting. Run the <b>Live Monitor</b> with attacks enabled, or launch an attack in the
       <b>Attack Simulator</b>, to generate flagged decisions.</div></div>""")
    st.stop()

PAGE = 8
n_pages = (len(pending) - 1) // PAGE + 1
pg = st.number_input(f"Page (of {n_pages})", 1, n_pages, 1, key="rq_page") if n_pages > 1 else 1
view = pending.iloc[(pg - 1) * PAGE: pg * PAGE]
lab = R.label_map(rt)

for _, row in view.iterrows():
    did = int(row["id"])
    top = json.loads(row["top_json"] or "[]")
    with st.container(key=f"card-review-{did}"):
        a, b, c = st.columns([1.35, 1.6, 0.85], gap="medium")
        with a:
            truth = ""
            if show_truth and row["sim_label"]:
                truth = badge(f"sim: {row['sim_label']}", {"attack": "attack", "drift": "drift"}.get(row["sim_label"], "neutral"))
            md(f"""<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
                <span class="mono" style="font-size:15px;font-weight:600">#{did}</span>{badge('Review', 'review')}
                {badge(row['source'], 'accent', False)}{truth}</div>
                <div class="muted" style="margin-top:6px">{time.strftime('%d %b %H:%M:%S', time.localtime(row['ts']))}
                · ref <span class="mono">{esc(row['record_ref'] or '—')}</span></div>
                <div style="margin-top:10px;font-size:14px">Prediction <b>{esc(names[int(row['pred'])])}</b> ·
                p=<span class="mono">{row['proba']:.3f}</span></div>
                <div style="margin-top:6px;font-size:13px" class="mono">T {row['T']:.3f} &lt; {row['threshold']:.3f}</div>
                <div class="muted mono" style="margin-top:2px">C {row['C']:.2f} · A {row['A']:.2f} · S {row['S']:.2f}</div>""")
        with b:
            md(f"<div class='muted'>Top deviating features (σ){tip('sigma')}</div>")
            charts.show(charts.deviation_bars(top, lab, 165), key=f"rq_dev_{did}")
        with c:
            st.button("Confirm attack", key=f"rq_atk_{did}", type="primary", icon=":material/gpp_bad:",
                      width="stretch", on_click=act, args=(did, "confirmed_attack"))
            st.button("Approve", key=f"rq_ok_{did}", icon=":material/check:", width="stretch",
                      on_click=act, args=(did, "approved"))
            with st.popover("Details", icon=":material/data_object:", width="stretch"):
                full = db.get(did)
                rec = json.loads(full["record_json"]) if full else {}
                feats = rec.get("features", {})
                st.dataframe({rt.schema.label(k): [rt.schema.display_value(k, v)] for k, v in feats.items()},
                             hide_index=True)
                st.json(rec, expanded=False)
