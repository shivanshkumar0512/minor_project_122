"""Audit Log: searchable, filterable record of every decision; CSV export."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import streamlit as st

from seaf.config import DOMAINS
from ui import charts, layout, state
from ui.components import badge, esc, md, page_header

if not layout.guard():
    st.stop()

page_header("Accountability", "Audit log",
            "Every decision SEAF has scored: prediction, trust components, verdict and review outcome, with the full "
            "attribution record kept for adjudication.", "receipt_long")

db = state.db()
df = db.query(limit=50_000)
if df.empty:
    md("<div class='empty'><div class='ic'>receipt_long</div><h4>No decisions logged yet</h4>"
       "<div>Score a record on the <b>Predict</b> page or start the <b>Live Monitor</b>.</div></div>")
    st.stop()

df["time"] = pd.to_datetime(df.ts, unit="s")
with st.container(key="card-audit-filters"):
    c1, c2, c3, c4, c5 = st.columns([1.1, 1, 1.2, 1.1, 1.4])
    doms = c1.multiselect("Domain", sorted(df.domain.unique()), default=[state.current_domain()],
                          format_func=state.domain_title)
    verdict = c2.multiselect("Verdict", ["Accept", "Review"])
    status = c3.multiselect("Status", ["auto_accepted", "pending", "approved", "confirmed_attack"])
    source = c4.multiselect("Source", sorted(df.source.unique()))
    lo, hi = df.time.min().date(), df.time.max().date()
    rng = c5.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    s1, s2 = st.columns([2, 1])
    q = s1.text_input("Search", placeholder="audit id, record ref, reviewer note…", label_visibility="collapsed")
    tmin, tmax = s2.slider("Trust range", 0.0, 1.0, (0.0, 1.0), 0.01, label_visibility="collapsed")

f = df
if doms:
    f = f[f.domain.isin(doms)]
if verdict:
    f = f[f.verdict.isin(verdict)]
if status:
    f = f[f.status.isin(status)]
if source:
    f = f[f.source.isin(source)]
if isinstance(rng, tuple) and len(rng) == 2:
    f = f[(f.time.dt.date >= rng[0]) & (f.time.dt.date <= rng[1])]
f = f[(f["T"] >= tmin) & (f["T"] <= tmax)]
if q:
    ql = q.lower().strip()
    f = f[f.id.astype(str).eq(ql.lstrip("#")) | f.record_ref.fillna("").str.lower().str.contains(ql, regex=False)
          | f.note.fillna("").str.lower().str.contains(ql, regex=False)]

top_feat = f.top_json.fillna("[]").map(lambda s: ", ".join(f"{t['feature']} {t['sigma']:+.1f}σ" for t in json.loads(s)[:3]))
view = pd.DataFrame({
    "id": f.id, "time": f.time, "domain": f.domain.map(state.domain_title), "source": f.source,
    "ref": f.record_ref, "pred": f.pred, "p": f.proba.round(4), "C": f.C.round(3), "A": f.A.round(3),
    "S": f.S.round(3), "T": f["T"].round(4), "threshold": f.threshold.round(3), "verdict": f.verdict,
    "status": f.status, "top deviations": top_feat, "note": f.note,
})

md(f"<div class='muted' style='margin:6px 0'>{len(view):,} of {len(df):,} decisions "
   f"{badge(str(int((view.verdict == 'Review').sum())) + ' flagged', 'review')}</div>")
sel = st.dataframe(
    view, hide_index=True, width="stretch", height=430, on_select="rerun", selection_mode="single-row",
    column_config={
        "id": st.column_config.NumberColumn("#", format="%d", width="small"),
        "time": st.column_config.DatetimeColumn("Time", format="DD MMM HH:mm:ss"),
        "T": st.column_config.ProgressColumn("Trust", min_value=0.0, max_value=1.0, format="%.3f"),
        "p": st.column_config.NumberColumn("p", format="%.3f"),
        "verdict": st.column_config.TextColumn("Verdict"),
    },
)
st.download_button("Export filtered rows to CSV", view.to_csv(index=False).encode(), file_name="seaf_audit_log.csv",
                   mime="text/csv", icon=":material/download:")

rows = sel.selection.rows if sel and sel.selection else []
if rows:
    did = int(view.iloc[rows[0]]["id"])
    full = db.get(did)
    rec = json.loads(full["record_json"]) if full and full.get("record_json") else {}
    st.markdown(f"#### Audit record #{did}")
    a, b = st.columns([1, 1.2])
    with a:
        md(f"<div class='card tight'><div class='muted'>{esc(state.domain_title(full['domain']))} · {esc(full['source'])} · "
           f"{dt.datetime.fromtimestamp(full['ts']).strftime('%d %b %Y %H:%M:%S')}</div>"
           f"<div style='margin-top:8px'>{badge(full['verdict'], 'review' if full['verdict'] == 'Review' else 'safe')} "
           f"{badge(full['status'], 'neutral')}</div>"
           f"<div class='mono' style='margin-top:10px'>p={full['proba']:.4f} · T={full['T']:.4f} · thr={full['threshold']:.3f}</div></div>")
        st.json(rec.get("features", {}), expanded=False)
    with b:
        spec = DOMAINS.get(full["domain"])
        lab = dict(spec.feature_labels) if spec else {}  # no model load needed
        charts.show(charts.deviation_bars(rec.get("top_deviations", []), lab, 210), key="audit_dev")
else:
    md("<div class='muted'>Select a row to inspect its full audit record.</div>")
