"""Analytics: reproduced paper results (vs the paper) and live audit statistics."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from seaf.config import RESULTS_DIR
from ui import charts, layout, state
from ui.components import callout, fmt_pct, kpi_html, kpi_row, md, page_header, section, tip
from ui.theme import colors

if not layout.guard():
    st.stop()

page_header("Evidence", "Analytics",
            f"Results reproduced by <code>scripts/run_experiments.py</code>, side by side with the values reported in "
            f"the paper, plus live statistics from the audit database. AUC{tip('auc')} is paired: each attacked record "
            "is compared with its own clean original.", "insights")


@st.cache_data(show_spinner=False)
def load_results(mtime: float) -> tuple[dict | None, dict | None]:
    ex, pr = RESULTS_DIR / "experiments.json", RESULTS_DIR / "paper_reference.json"
    return (json.loads(ex.read_text()) if ex.exists() else None,
            json.loads(pr.read_text()) if pr.exists() else None)


p = RESULTS_DIR / "experiments.json"
ex, paper = load_results(p.stat().st_mtime if p.exists() else 0)
tab_paper, tab_live = st.tabs([":material/science: Paper experiments", ":material/sensors: Live audit stats"])

DOM = ["finance", "healthcare", "recruitment"]
DLAB = [d.title() for d in DOM]

with tab_paper:
    if not ex:
        callout("No results yet. Run <code>python scripts/run_experiments.py</code>.", warn=True)
    else:
        D = ex["domains"]
        cd = ex["cross_domain"]
        kpi_row([
            kpi_html("Attack success", f"{cd['attack_success']}/{cd['attack_attempted']}",
                     f"paper {paper['attack']['success']}/{paper['attack']['total']}", "attack", "", "pp_attack"),
            kpi_html("Mean |Δp|", " / ".join(f"{D[d]['attack']['mean_abs_dp']:.3f}" for d in DOM),
                     "paper 0.027 / 0.031 / 0.032", "neutral"),
            kpi_html("Stability AUC range", f"{cd['stability']['fixed_range']:.3f} → {cd['stability']['absolute_range']:.3f}",
                     "paper 0.350 → 0.137", "drift"),
            kpi_html("Summation identity error", f"{max(D[d]['fig2']['identity_error'] for d in DOM):.1e}",
                     "base + Σφ = p", "safe", "", "SHAP"),
        ])

        section("1 · Explanation-space vs input-space detection")
        l, r = st.columns([1.5, 1])
        with l:
            ser = {"Explanation space (ours)": [D[d]["space_auc"]["explanation"] for d in DOM],
                   "Input space (ours)": [D[d]["space_auc"]["input"] for d in DOM],
                   "Explanation (paper)": [paper["space_auc"][d]["explanation"] for d in DOM],
                   "Input (paper)": [paper["space_auc"][d]["input"] for d in DOM]}
            errs = {"Explanation space (ours)": [D[d]["space_auc"]["explanation_ci"] for d in DOM],
                    "Input space (ours)": [D[d]["space_auc"]["input_ci"] for d in DOM]}
            charts.show(charts.grouped_bars(DLAB, ser, errors=errs, height=330, y_range=(0.3, 1.02)), key="an_space")
        with r:
            rows = []
            for d in DOM:
                s = D[d].get("seed_spread") or {}
                rows.append({"Domain": d.title(),
                             "Expl. (ours)": f"{D[d]['space_auc']['explanation']:.3f}",
                             "Input (ours)": f"{D[d]['space_auc']['input']:.3f}",
                             "Expl. 5-seed mean±sd": f"{s.get('explanation', {}).get('mean', float('nan')):.3f} ± "
                                                     f"{s.get('explanation', {}).get('sd', float('nan')):.3f}",
                             "Input 5-seed mean±sd": f"{s.get('input', {}).get('mean', float('nan')):.3f} ± "
                                                     f"{s.get('input', {}).get('sd', float('nan')):.3f}",
                             "Paper": f"{paper['space_auc'][d]['explanation']:.3f} vs {paper['space_auc'][d]['input']:.3f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            md("<div class='muted'>Error bars: 95% bootstrap CI over the 60 attack/original pairs. With 60 pairs the "
               "CIs are roughly ±0.08-0.10 wide, so single-run gaps of that size are within sampling noise.</div>")

        section("2 · Stability signal: fixed direction vs absolute deviation")
        l, r = st.columns([1.5, 1])
        with l:
            ser = {"Fixed direction (ours)": [D[d]["stability_auc"]["fixed"] for d in DOM],
                   "Absolute deviation (ours)": [D[d]["stability_auc"]["absolute"] for d in DOM],
                   "Fixed (paper)": [paper["stability_auc"][d]["fixed"] for d in DOM],
                   "Absolute (paper)": [paper["stability_auc"][d]["absolute"] for d in DOM]}
            charts.show(charts.grouped_bars(DLAB, ser, height=320, y_range=(0, 1.02)), key="an_stab")
        with r:
            hc = D["healthcare"]["stability_auc"]
            callout(f"<b>Inversion.</b> In healthcare the naive signal scores <b>{hc['fixed']:.3f}</b> (below 0.5): "
                    f"attacked records look <i>more</i> stable. Mechanism check: prediction dispersion under the same "
                    f"±2% jitter is <b>{hc['pred_dispersion_attack']:.3f}</b> for attacked vs "
                    f"<b>{hc['pred_dispersion_clean']:.3f}</b> for originals (paper 0.018 vs 0.031). The attack parks "
                    "records in flat regions of the decision surface. Scoring |deviation| from the baseline median "
                    "treats unusually stable and unusually volatile explanations alike.")
            st.dataframe(pd.DataFrame([{"Domain": d.title(), "m_ref (ours)": f"{D[d]['stability_auc']['m_ref']:.3f}",
                                        "m_ref (paper)": paper["stability_auc"][d]["m_ref"]} for d in DOM]),
                         hide_index=True, width="stretch")

        section("3 · Per-signal and combined trust AUC")
        sigs = [("confidence", "Confidence 1−C"), ("anomaly", "Anomaly A"), ("stability", "Stability S"),
                ("trust_equal", "Equal-weight trust"), ("trust_fitted", "Fitted trust (5-fold LR)")]
        l, r = st.columns([1.5, 1])
        with l:
            ser = {d.title(): [D[d]["signal_auc"][k] for k, _ in sigs] for d in DOM}
            charts.show(charts.grouped_bars([s for _, s in sigs], ser, height=330, y_range=(0, 1.02)), key="an_sig")
        with r:
            tbl = pd.DataFrame({"Signal": [s for _, s in sigs],
                                **{f"{d.title()} (ours / paper)": [f"{D[d]['signal_auc'][k]:.3f} / {paper['signal_auc'][d][k]:.3f}"
                                                                   for k, _ in sigs] for d in DOM}})
            st.dataframe(tbl, hide_index=True, width="stretch")
            w = {d.title(): [round(x, 2) for x in D[d]["fitted_weights"]] for d in DOM}
            md("<div class='muted'>Fitted logistic weights on (1−C, A, S): " +
               " · ".join(f"{k} {v}" for k, v in w.items()) + "</div>")

        section("4 · Feedback loop: threshold recalibration")
        fb_rows, before, after = [], [], []
        for d in DOM:
            f = D[d]["feedback"]
            h = f["heldout"]
            before.append(h["detection_before"]); after.append(h["detection_after"])
            fb_rows.append({"Domain": d.title(), "Threshold": f"{f['default_threshold']:.3f} → {f['recalibrated_threshold']:.3f}",
                            "Held-out detection": f"{fmt_pct(h['detection_before'])} → {fmt_pct(h['detection_after'])}",
                            "Held-out FPR": f"{fmt_pct(h['fpr_before'])} → {fmt_pct(h['fpr_after'])}"})
        pooled = cd["feedback_pooled"]["heldout"]
        fb_rows.append({"Domain": "Pooled", "Threshold": "per domain",
                        "Held-out detection": f"{fmt_pct(pooled['before']['detection'])} → {fmt_pct(pooled['after']['detection'])}",
                        "Held-out FPR": f"{fmt_pct(pooled['before']['fpr'])} → {fmt_pct(pooled['after']['fpr'])}"})
        l, r = st.columns([1.3, 1])
        with l:
            charts.show(charts.grouped_bars(DLAB + ["Pooled"], {"Default threshold": before + [pooled["before"]["detection"]],
                                                                "Recalibrated": after + [pooled["after"]["detection"]]},
                                            ref_line=None, y_title="held-out detection", height=300,
                                            y_range=(0, 1), text_fmt="{:.0%}"), key="an_fb")
        with r:
            st.dataframe(pd.DataFrame(fb_rows), hide_index=True, width="stretch")
            md("<div class='muted'>Paper: detection 10% → 40%, FPR 5% → 13% on held-out data. The default threshold is "
               "the 5th percentile of clean baseline trust; recalibration maximises detection within a 10% false-alarm "
               "budget on the tuning half.</div>")

        section("5 · Baseline contamination (dose-response)")
        l, r = st.columns([1.5, 1])
        with l:
            lv = [c["level"] for c in D["finance"]["contamination"]]
            ys = {d.title(): [c["auc"] for c in D[d]["contamination"]] for d in DOM}
            charts.show(charts.line_chart(lv, ys, "attacks added to baseline (share of baseline size)",
                                          "explanation-space AUC", 290, (0.3, 1), ".0%"), key="an_cont")
        with r:
            callout("Wrongly cleared attacks are appended to the clean baseline before refitting the detector. "
                    "Degradation is monotonic and gradual: isolated review mistakes are tolerable, a systematically "
                    "compromised review process is not. This is why baseline refits are manual and restricted to "
                    "reviewer-verified records (Admin page).")

        sens = (ex.get("sensitivity") or {}).get("recruitment_unbalanced")
        if sens:
            section("Sensitivity: recruitment without class weighting")
            m = sens["model"]
            callout(f"The paper's recruitment metrics (recall 0.106) are reproduced almost exactly <b>without</b> "
                    f"class weighting: recall {m['recall']:.3f}, AUC {m['roc_auc']:.3f} "
                    f"(TN/FP/FN/TP {m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}). On that model explanation vs input AUC is "
                    f"<b>{sens['space_auc']['explanation']:.3f}</b> vs <b>{sens['space_auc']['input']:.3f}</b> and "
                    f"equal-weight trust AUC <b>{sens['signal_auc']['trust_equal']:.3f}</b>. The deployed model uses "
                    "class_weight='balanced' as specified.")

        section("Protected model performance (Table II)")
        st.dataframe(pd.DataFrame([{"Domain": d.title(), **{k: f"{D[d]['model'][k]:.4f} ({paper['model'][d][k]:.4f})"
                                                           for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
                                    "SHAP ms/rec": f"{D[d]['model']['shap_ms']:.2f}"} for d in DOM]),
                     hide_index=True, width="stretch")
        md("<div class='muted'>Ours (paper) · held-out test set.</div>")

with tab_live:
    db = state.db()
    df = db.query(limit=20_000, order="ASC")
    if df.empty:
        md("<div class='empty'><div class='ic'>sensors</div><h4>No live data</h4><div>Use the app to create decisions.</div></div>")
        st.stop()
    dom = st.segmented_control("Domain", ["All"] + state.available_domains(), default="All",
                               format_func=lambda k: "All" if k == "All" else state.domain_title(k), key="an_dom") or "All"
    if dom != "All":
        df = df[df.domain == dom]
    c = colors()
    n = len(df)
    flagged = int((df.verdict == "Review").sum())
    reviewed = df[df.status.isin(["approved", "confirmed_attack"])]
    kpi_row([
        kpi_html("Decisions", f"{n:,}", f"{df.domain.nunique()} domain(s)", "accent"),
        kpi_html("Flag rate", fmt_pct(flagged / max(n, 1)), f"{flagged} routed to review", "review"),
        kpi_html("Confirmed attacks", f"{int((df.status == 'confirmed_attack').sum())}", "by reviewers", "attack"),
        kpi_html("Approved after review", f"{int((df.status == 'approved').sum())}", "false alarms cleared", "safe"),
        kpi_html("Mean trust", f"{df['T'].mean():.3f}", f"median {df['T'].median():.3f}", "neutral", "", "trust"),
    ])
    l, r = st.columns([1.6, 1])
    with l:
        section("Decisions over time")
        t = pd.to_datetime(df.ts, unit="s")
        span = (df.ts.max() - df.ts.min()) if n > 1 else 0
        freq = "1h" if span > 6 * 3600 else "5min" if span > 1800 else "1min"
        g = df.assign(t=t.dt.floor(freq)).groupby(["t", "verdict"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for v, col in (("Accept", c["safe"]), ("Review", c["review"])):
            if v in g:
                fig.add_trace(go.Bar(x=g.index, y=g[v], name=v, marker=dict(color=col, line=dict(color=c["surface"], width=1))))
        fig.update_layout(barmode="stack", height=280, yaxis=dict(title="decisions"), margin=dict(l=8, r=8, t=30, b=8))
        charts.show(fig, key="an_time")
    with r:
        section("Review status")
        st_counts = df.status.value_counts()
        order = [("auto_accepted", "Auto-accepted", c["safe"]), ("pending", "Pending", c["review"]),
                 ("approved", "Approved", c["accent"]), ("confirmed_attack", "Confirmed attack", c["attack"])]
        charts.show(charts.donut([o[1] for o in order], [int(st_counts.get(o[0], 0)) for o in order],
                                 [o[2] for o in order]), key="an_donut")
    l, r = st.columns(2)
    with l:
        section("Trust by reviewer outcome")
        tc = df.loc[df.status.isin(["approved", "auto_accepted"]), "T"].to_numpy()
        ta = df.loc[df.status == "confirmed_attack", "T"].to_numpy()
        thr = state.threshold(dom) if dom != "All" else float(np.median([state.threshold(k) for k in state.available_domains()]))
        charts.show(charts.distribution(tc, ta, thr), key="an_dist")
        md("<div class='muted'>Clean = auto-accepted or approved; attacked = reviewer-confirmed.</div>")
    with r:
        section("Threshold history")
        keys = [dom] if dom != "All" else state.available_domains()
        fig = go.Figure()
        for i, k in enumerate(keys):
            h = db.threshold_history(k)
            if len(h):
                fig.add_trace(go.Scatter(x=pd.to_datetime(h.ts, unit="s"), y=h.threshold, mode="lines+markers",
                                         name=state.domain_title(k), line=dict(shape="hv", width=2, color=c["series"][i % 4]),
                                         marker=dict(size=8, line=dict(color=c["surface"], width=2)),
                                         customdata=h.reason, hovertemplate="%{y:.3f}<br>%{customdata}<extra></extra>"))
        fig.update_layout(height=280, yaxis=dict(title="threshold"), margin=dict(l=8, r=8, t=30, b=8))
        charts.show(fig, key="an_thr")
    md(f"<div class='muted'>Updated {time.strftime('%H:%M:%S')} · free-tier storage is ephemeral; seeded demo data is "
       "recreated on restart.</div>")
