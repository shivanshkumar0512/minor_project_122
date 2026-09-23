"""Live Monitor: simulated decision stream with attack injection and drift.

Only the ``live_panel`` fragment re-runs on each tick (``st.fragment(run_every=…)``);
the controls above it re-run the page only when changed.  Each tick scores a
small batch in one SHAP call (originals + jittered copies), so the stream stays
smooth on a free-tier CPU.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from seaf.drift import DRIFT, SPIKE, FlagRateMonitor
from seaf.stream import TrafficSimulator
from ui import charts, layout, state
from ui import record as R
from ui.components import badge, esc, kpi_html, kpi_row, md, page_header, section, tip

if not layout.guard():
    st.stop()

rt = state.runtime()
ss = st.session_state
names = rt.schema.class_names
SPEEDS = {"Slow": (2.0, 1), "Normal": (1.25, 2), "Fast": (1.0, 4)}
MAX_HIST = 400

page_header(f"{rt.title} · real-time", "Live monitor",
            "A simulated stream of decisions from the evaluation split. Inject attacks or start a gradual drift and "
            f"watch SEAF separate a sudden attack spike from slow population change {tip('drift')}.", "monitor_heart")


def reset() -> None:
    pairs = rt.attack_pairs()
    tune = rt.tune_idx
    ss.live_sim = TrafficSimulator(rt.X_eval.values[tune], [f"eval-{i}" for i in tune],
                                   rt.bundle["X_adv"][pairs.index.to_numpy()],
                                   [f"adv-{int(i)}" for i in pairs.eval_idx], seed=int(time.time()) % 100_000)
    thr = state.threshold(rt.key)
    _, fpr = state.heldout_rates(rt, thr)
    ss.live_monitor = FlagRateMonitor(expected_rate=max(fpr, 0.02))
    ss.live_hist = []
    ss.live_thr0 = thr
    ss.live_last_state = "normal"
    ss.live_burst = 0
    ss.live_running = ss.get("live_running", False)


if "live_sim" not in ss:
    reset()

# --------------------------------------------------------------------------- #
# Controls (full-page reruns only when changed)
# --------------------------------------------------------------------------- #
imp = rt.bundle["importance"]
num = rt.schema.numeric_features
num_sorted = sorted(num, key=lambda f: -imp[rt.schema.feature_names.index(f)])

with st.container(key="card-live-controls"):
    c1, c2, c3, c4 = st.columns([1.25, 1.2, 1.45, 1.6], gap="medium")
    with c1:
        st.markdown("##### Stream")
        b1, b2 = st.columns(2)
        label = "Pause" if ss.live_running else "Start"
        if b1.button(label, type="primary", icon=":material/pause:" if ss.live_running else ":material/play_arrow:",
                     width="stretch"):
            ss.live_running = not ss.live_running
            st.rerun()
        if b2.button("Reset", icon=":material/restart_alt:", width="stretch"):
            ss.live_running = False
            reset()
            st.rerun()
        speed = st.segmented_control("Speed", list(SPEEDS), default="Normal", key="live_speed_ctl") or "Normal"
    with c2:
        st.markdown("##### Attacks")
        attack_rate = st.slider("Injection rate", 0, 50, 8, 1, format="%d%%", key="live_rate_ctl",
                                help="Share of records replaced by a prediction-preserving attack") / 100
        if st.button("Inject burst ×20", icon=":material/bolt:", width="stretch",
                     help="Next 20 records are attacks - a sudden spike"):
            ss.live_burst = 20
    with c3:
        st.markdown("##### Drift")
        drift_on = st.toggle("Start drift", key="live_drift_ctl",
                             help="Gradually scales the selected features upwards (population change, not an attack)")
        drift_feats = st.multiselect("Drifting features", num, default=num_sorted[:2], key="live_drift_feats",
                                     format_func=rt.schema.label, label_visibility="collapsed")
        drift_rate = st.slider("Drift speed", 0.001, 0.01, 0.004, 0.001, format="%.3f / record", key="live_drift_rate")
    with c4:
        st.markdown("##### Performance")
        full_stab = st.toggle("Stability on every record", value=True, key="live_stab_ctl",
                              help=f"Stability costs {rt.seaf.n_jitter} extra SHAP evaluations per record. Off = measure "
                                   "every 3rd record and use the baseline median for the rest.")
        md(f"<div class='muted'>{rt.meta['metrics']['shap_ms']:.1f} ms per SHAP call · "
           f"{rt.seaf.n_jitter + 1} calls per record with stability, batched per tick.</div>")
        if st.button("Recalibrate threshold", icon=":material/tune:", width="stretch",
                     help="Re-select the threshold from all labelled outcomes (≤10% false alarms)"):
            old, cal = state.recalibrate_domain(rt, "recalibrated from Live Monitor")
            st.toast(f"Threshold {old:.3f} → {cal.threshold:.3f}", icon=":material/tune:")

interval, per_tick = SPEEDS[speed]
drift_idx = [rt.schema.feature_names.index(f) for f in drift_feats]


# --------------------------------------------------------------------------- #
# Live fragment
# --------------------------------------------------------------------------- #
def step() -> None:
    sim: TrafficSimulator = ss.live_sim
    mon: FlagRateMonitor = ss.live_monitor
    rate = 1.0 if ss.live_burst > 0 else attack_rate
    events = sim.next(per_tick, rate, drift_on, drift_idx, drift_rate)
    ss.live_burst = max(0, ss.live_burst - per_tick)
    X = np.vstack([e.x for e in events])
    n0 = len(ss.live_hist)
    mask = np.ones(len(X), bool) if full_stab else np.array([(n0 + i) % 3 == 0 for i in range(len(X))])
    res = R.score(rt, X, stability=mask)
    ids = R.log_records(rt, res, "stream", [e.ref for e in events], [e.label for e in events])
    zin = rt.seaf.input_detector.zscores(X)[:, rt.seaf.numeric_idx]
    for i, e in enumerate(events):
        snap = mon.update(bool(res.flagged[i]), zin[i], float(res.T[i]))
        ss.live_hist.append({
            "n": n0 + i + 1, "id": ids[i], "T": float(res.T[i]), "p": float(res.proba[i]),
            "pred": int(res.pred[i]), "flagged": bool(res.flagged[i]), "label": e.label,
            "verdict": "Review" if res.flagged[i] else "Accept", "threshold": float(res.threshold),
            "drift": e.drift_level, **snap,
        })
        if e.is_attack and res.flagged[i]:
            st.toast(f"Attack flagged · audit #{ids[i]} · T={res.T[i]:.3f}", icon=":material/gpp_bad:")
        if snap["state"] != ss.live_last_state:
            if snap["state"] == SPIKE:
                st.toast("Sudden spike in flag rate: possible attack burst", icon=":material/crisis_alert:")
            elif snap["state"] == DRIFT:
                st.toast("Gradual drift detected: population shifting", icon=":material/trending_up:")
            ss.live_last_state = snap["state"]
    if len(ss.live_hist) > MAX_HIST:
        ss.live_hist = ss.live_hist[-MAX_HIST:]


def live_panel() -> None:
    if ss.live_running:
        step()
    hist = pd.DataFrame(ss.live_hist)
    thr = state.threshold(rt.key)
    counts = state.db().query(domain=rt.key, source="stream", status="confirmed_attack", limit=10_000)
    st_state = hist["state"].iloc[-1] if len(hist) else "normal"
    state_badge = {"normal": badge("Normal", "safe"), SPIKE: badge("Attack spike", "attack"),
                   DRIFT: badge("Gradual drift", "drift")}[st_state]
    drift_lvl = ss.live_sim.drift_level
    extra = (badge(f"drift +{100 * drift_lvl:.0f}%", "drift") if drift_lvl > 0.005 else "") + \
            (badge("burst in progress", "attack") if ss.live_burst > 0 else "")
    md(f'<div class="statusbar"><span class="live-dot {"" if ss.live_running else "off"}"></span>'
       f'<b>{"LIVE" if ss.live_running else "PAUSED"}</b><span class="muted">· {esc(rt.title)} · {per_tick} '
       f'record(s) every {interval:.2f}s</span><span style="flex:1"></span>{state_badge}{extra}</div>')

    n = len(hist)
    flagged = int(hist["flagged"].sum()) if n else 0
    atk = hist[hist["label"] == "attack"] if n else hist
    clean = hist[hist["label"] != "attack"] if n else hist
    caught = int(atk["flagged"].sum()) if len(atk) else 0
    fa = int(clean["flagged"].sum()) if len(clean) else 0
    dthr = thr - ss.live_thr0
    kpi_row([
        kpi_html("Processed", f"{n:,}", f"{per_tick / interval:.1f} rec/s" if ss.live_running else "paused", "accent"),
        kpi_html("Flagged for review", f"{flagged:,}", f"{100 * flagged / max(n, 1):.1f}% of stream", "review"),
        kpi_html("Injected attacks caught", f"{caught}/{len(atk)}",
                 f"{100 * caught / max(len(atk), 1):.0f}% detection", "attack", "", "detection"),
        kpi_html("False alarms", f"{fa}", f"{100 * fa / max(len(clean), 1):.1f}% of clean/drifted", "neutral", "", "fpr"),
        kpi_html("Confirmed by reviewers", f"{len(counts)}", "stream decisions", "safe"),
        kpi_html("Current threshold", f"{thr:.3f}", f"{dthr:+.3f} since start" if abs(dthr) > 1e-9 else "unchanged",
                 "review", "up" if dthr > 0 else "down" if dthr < 0 else "", "threshold"),
    ])

    feed_col, chart_col = st.columns([1.25, 2], gap="medium")
    with feed_col:
        section("Decision feed")
        if not n:
            md("<div class='empty'><div class='ic'>stream</div><h4>Stream idle</h4>"
               "<div>Press <b>Start</b> to begin feeding decisions.</div></div>")
        else:
            rows = []
            for r in reversed(ss.live_hist[-13:]):
                cls = "hot" if r["flagged"] and r["label"] == "attack" else "flag" if r["flagged"] else ""
                col = "var(--review)" if r["flagged"] else "var(--safe)"
                tag = {"attack": badge("attack", "attack", False), "drift": badge("drift", "drift", False)}.get(r["label"], "")
                rows.append(f"<div class='feed-row {cls}'><span class='id'>#{r['id']}</span>"
                            f"<span>{esc(names[r['pred']])} <span class='p'>p={r['p']:.2f}</span> {tag}</span>"
                            f"<span class='tt' style='color:{col}'>{r['T']:.3f}</span>"
                            f"{badge(r['verdict'], 'review' if r['flagged'] else 'safe')}</div>")
            md(f"<div class='feed'>{''.join(rows)}</div>")
            md("<div class='muted' style='margin-top:6px'>Attack / drift tags are the simulator's ground truth, "
               "shown for the demo only; SEAF never sees them.</div>")
    with chart_col:
        section("Trust over time", "trust")
        tail = hist.tail(200)
        charts.show(charts.trust_timeline(tail, tail["threshold"].tolist() if n else []), key="live_trust")
        section("Rolling flag rate: spike vs drift")
        charts.show(charts.flag_rate_chart(tail, ss.live_monitor.expected), key="live_rate")
        md("<div class='muted'><b>Red shading = spike</b>: the last 20 decisions' flag rate or mean trust departs "
           "abruptly from the preceding window. <b>Violet shading = drift</b>: the mean of standardised inputs has "
           "moved consistently away from where the stream started. Random-sign attack edits average out; population "
           "drift does not. Bursts are easiest to see in <b>Healthcare</b>, where attacked records have clearly "
           "lower trust. In Finance and Recruitment per-record separation is weak (see Analytics).</div>")


st.fragment(run_every=interval if ss.live_running else None)(live_panel)()
