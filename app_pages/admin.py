"""Admin: system health, threshold controls, manual baseline refit, custom datasets."""
from __future__ import annotations

import json
import platform
import resource
import shutil
import time
from io import BytesIO

import numpy as np
import pandas as pd
import shap
import sklearn
import streamlit as st

from seaf.artifacts import RUNTIME_ARTIFACT_DIR, bundle_path, save_bundle
from seaf.config import FPR_BUDGET
from seaf.feedback import rates
from seaf.pipeline import build_domain
from seaf.preprocessing import encode_frame, id_like_columns
from ui import layout, state
from ui.components import badge, callout, esc, kpi_html, kpi_row, md, page_header, section

MAX_MB, MAX_ROWS, MAX_COLS = 5, 3000, 40

if not layout.guard():
    st.stop()

rt = state.runtime()
db = state.db()
page_header("Operations", "Admin",
            "Health checks, threshold controls, manual baseline maintenance and bring-your-own-dataset.",
            "admin_panel_settings")


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


tab_h, tab_t, tab_b, tab_u = st.tabs([":material/monitor_heart: Health", ":material/tune: Threshold",
                                      ":material/database: Baseline", ":material/upload_file: Custom dataset"])

# --------------------------------------------------------------------------- #
with tab_h:
    counts = db.counts()
    kpi_row([
        kpi_html("Status", "Healthy" if db.healthy() else "Degraded", "audit DB reachable", "safe" if db.healthy() else "attack"),
        kpi_html("Memory (RSS)", f"{rss_mb():.0f} MB", "this process", "accent"),
        kpi_html("Domains available", str(len(state.available_domains())), "lazy-loaded, ≤2 resident", "neutral"),
        kpi_html("Audit rows", f"{counts.get('total', 0):,}", str(db.path.name), "neutral"),
    ])
    rows = []
    for k in state.available_domains():
        m = state.meta(k) or {}
        p = bundle_path(k)
        rows.append({"Domain": state.domain_title(k), "Version": m.get("version"), "Built": m.get("created"),
                     "Source": "runtime refit" if p and RUNTIME_ARTIFACT_DIR in p.parents else "shipped",
                     "Size (MB)": m.get("size_mb"), "Accuracy": round((m.get("metrics") or {}).get("accuracy", 0), 4),
                     "Baseline n": m.get("n_baseline"), "Threshold": round(state.threshold(k), 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    md(f"<div class='muted mono'>Python {platform.python_version()} · streamlit {st.__version__} · "
       f"scikit-learn {sklearn.__version__} · shap {shap.__version__} · numpy {np.__version__} · pandas {pd.__version__}"
       f"<br>DB: {esc(db.path)} · health endpoint: <b>/_stcore/health</b></div>")
    callout("Free-tier hosting has <b>ephemeral storage</b>: the audit DB, reviewer decisions, refits and uploaded "
            "datasets are lost on restart. Demo data is re-seeded automatically at start-up.", warn=True)
    if st.button("Reset audit database to demo data", icon=":material/restart_alt:"):
        db.clear()
        for k in state.available_domains():
            state.seed_domain(db, k)
        for k in [k for k in st.session_state if str(k).startswith(("live_", "review_", "demo"))]:
            del st.session_state[k]
        st.toast("Audit database reset", icon=":material/restart_alt:")
        st.rerun()

# --------------------------------------------------------------------------- #
with tab_t:
    thr = state.threshold(rt.key)
    det, fpr = state.heldout_rates(rt, thr)
    default_t = rt.seaf.default_threshold
    kpi_row([kpi_html("Current threshold", f"{thr:.3f}", f"default {default_t:.3f}", "accent", "", "threshold"),
             kpi_html("Held-out detection", f"{100 * det:.1f}%", None, "safe", "", "detection"),
             kpi_html("Held-out FPR", f"{100 * fpr:.1f}%", None, "review", "", "fpr")])
    c1, c2 = st.columns(2)
    with c1:
        budget = st.slider("False-alarm budget", 0.02, 0.30, FPR_BUDGET, 0.01, key="adm_budget")
        if st.button("Recalibrate now", type="primary", icon=":material/tune:"):
            old, cal = state.recalibrate_domain(rt, f"manual recalibration (budget {budget:.2f})", budget)
            st.toast(f"Threshold {old:.3f} → {cal.threshold:.3f}")
            st.rerun()
    with c2:
        md("<div class='muted'>Restore the uncalibrated default: the 5th percentile of clean-baseline trust.</div>")
        if st.button("Reset to default", icon=":material/undo:"):
            d0, f0 = state.heldout_rates(rt, default_t)
            db.set_threshold(rt.key, default_t, d0, f0, 0, "manual reset to default")
            st.rerun()
    hist = db.threshold_history(rt.key)
    if len(hist):
        hist = hist.assign(time=pd.to_datetime(hist.ts, unit="s"))[["time", "threshold", "detection", "fpr", "n_labels", "reason"]]
        st.dataframe(hist.iloc[::-1], hide_index=True, width="stretch", height=260)

# --------------------------------------------------------------------------- #
with tab_b:
    approved = db.query(domain=rt.key, status="approved", with_record=True, limit=5000)
    approved = approved[approved.source != "seed"]
    feats = []
    for s in approved.record_json.dropna():
        rec = json.loads(s)
        if set(rec.get("features", {})) == set(rt.schema.feature_names):
            feats.append([rec["features"][f] for f in rt.schema.feature_names])
    X_extra = np.array(feats, float) if feats else np.zeros((0, len(rt.schema.feature_names)))
    kpi_row([kpi_html("Baseline records", f"{rt.seaf.baseline_['n']:,}", "clean baseline split", "accent"),
             kpi_html("Reviewer-verified clean", f"{len(X_extra)}", "approved decisions (excl. seed)", "safe"),
             kpi_html("Baseline version", str(rt.meta.get("baseline_version", 1)), rt.bundle.get("created", ""), "neutral")])
    callout("The baseline of normal reasoning is <b>never updated automatically</b>. A refit here uses only the "
            "original clean baseline plus decisions a reviewer explicitly <b>approved</b>. Wrongly cleared attacks "
            "would contaminate it (see Analytics § 5), so refits are a deliberate, manual action.")
    if st.button("Refit baseline with verified clean records", type="primary", disabled=len(X_extra) == 0,
                 icon=":material/model_training:"):
        with st.status("Refitting SEAF baseline…", expanded=True) as s:
            s.write(f"TreeSHAP + Isolation Forest + stability on {rt.seaf.baseline_['n'] + len(X_extra)} records")
            new = rt.seaf.refit_baseline(X_extra)
            s.write("Re-scoring evaluation pools for the feedback loop")
            b = dict(rt.bundle)
            sc = b["scores"].copy()
            X_eval, X_adv, pool = b["X_eval"].values, b["X_adv"], b["attack_pool"]
            ok = pool["success"].to_numpy()
            rc, ra = new.score(X_eval), new.score(X_adv[ok])
            T_new = np.r_[rc.T, ra.T]
            if len(T_new) == len(sc):
                sc["T"] = T_new
            b.update(seaf=new, scores=sc, created=time.strftime("%Y-%m-%dT%H:%M:%S"))
            b["meta"] = {**b["meta"], "baseline_version": int(b["meta"].get("baseline_version", 1)) + 1,
                         "n_baseline": int(new.baseline_["n"])}
            b["meta"]["calibration"] = {**b["meta"]["calibration"], "default_threshold": new.default_threshold}
            save_bundle(rt.key, b, RUNTIME_ARTIFACT_DIR)
            held = sc[sc.split == "heldout"]
            d0, f0 = rates(new.default_threshold, held.loc[held.kind == "clean", "T"], held.loc[held.kind == "attack", "T"])
            db.set_threshold(rt.key, new.default_threshold, d0, f0, 0, "baseline refit: reset to new default")
            s.update(label="Baseline refitted", state="complete")
        st.cache_data.clear()
        st.toast("Baseline refitted and saved as a runtime version", icon=":material/check_circle:")
        st.rerun()
    if bundle_path(rt.key) and RUNTIME_ARTIFACT_DIR in bundle_path(rt.key).parents and rt.key != "custom":
        if st.button("Revert to shipped baseline", icon=":material/undo:"):
            shutil.rmtree(RUNTIME_ARTIFACT_DIR / rt.key, ignore_errors=True)
            db.set_threshold(rt.key, (state.load_meta(rt.key) or {}).get("calibration", {}).get("default_threshold", 0.5),
                             reason="reverted to shipped artifacts")
            st.cache_data.clear()
            st.rerun()

# --------------------------------------------------------------------------- #
with tab_u:
    md(f"<div class='muted'>Upload a CSV (≤ {MAX_MB} MB) with a binary or categorical target. It is cleaned "
       f"(ids / constant columns dropped, categoricals label-encoded), capped at {MAX_ROWS:,} rows and {MAX_COLS} "
       "features, and put through the identical SEAF protocol: RF(300, depth 10) → 80/20 split → baseline / "
       "evaluation halves → attack pool.</div>")
    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if up is not None:
        if up.size > MAX_MB * 1e6:
            st.error(f"File is {up.size / 1e6:.1f} MB; the limit is {MAX_MB} MB.")
            st.stop()
        raw = up.getvalue()
        sep = ";" if raw[:2000].count(b";") > raw[:2000].count(b",") else ","
        try:
            df = pd.read_csv(BytesIO(raw), sep=sep, skipinitialspace=True)
        except Exception as e:  # noqa: BLE001 - user file
            st.error(f"Could not parse the CSV: {e}")
            st.stop()
        df.columns = [str(c).strip() for c in df.columns]
        st.dataframe(df.head(8), hide_index=True, width="stretch")
        md(f"<div class='muted'>{len(df):,} rows × {df.shape[1]} columns</div>")
        c1, c2, c3 = st.columns(3)
        cand = [c for c in df.columns if 2 <= df[c].nunique() <= 20]
        if not cand:
            st.error("No suitable target column (needs 2-20 distinct values).")
            st.stop()
        hints = ("target", "label", "class", "status", "outcome", "attrition", "churn", "default", "approved", "y")
        named = [i for i, c in enumerate(cand) if any(h == c.lower() or h in c.lower().split("_") or
                                                      c.lower().endswith(h) for h in hints)]
        binary = [i for i, c in enumerate(cand) if df[c].nunique() == 2]
        guess = named[0] if named else (binary[-1] if binary else len(cand) - 1)
        target = c1.selectbox("Target column", cand, index=guess,
                              help="Guessed from the column name / binary columns. Please confirm.")
        classes = sorted(df[target].dropna().astype(str).str.strip().unique().tolist())
        pos = c2.selectbox("Positive class (probability shown)", classes, index=len(classes) - 1)
        title = c3.text_input("Display name", value=up.name.rsplit(".", 1)[0][:30])
        drop = [c for c in id_like_columns(df.drop(columns=[target]))]
        drop = st.multiselect("Columns to drop (ids detected automatically)", [c for c in df.columns if c != target],
                              default=drop)
        balanced = st.toggle("class_weight='balanced'", value=df[target].astype(str).value_counts(normalize=True).max() > 0.7)
        if st.button("Build model + SEAF baseline", type="primary", icon=":material/construction:"):
            work = df.copy()
            for c in work.columns:
                if not pd.api.types.is_numeric_dtype(work[c]):
                    work[c] = work[c].astype(str).str.strip()
            if len(work) > MAX_ROWS:
                work = work.sample(MAX_ROWS, random_state=42)
            try:
                X, y, schema = encode_frame(work, target, pos, tuple(drop), class_names=(f"not {pos}", str(pos)))
            except Exception as e:  # noqa: BLE001
                st.error(f"Encoding failed: {e}")
                st.stop()
            if X.shape[1] > MAX_COLS:
                st.error(f"{X.shape[1]} features after encoding; the limit is {MAX_COLS}.")
                st.stop()
            if y.sum() < 20 or (len(y) - y.sum()) < 20:
                st.error("Need at least 20 records of each class.")
                st.stop()
            bar = st.progress(0.0, "Starting")
            try:
                bundle = build_domain(X, y, schema, key="custom", title=title or "Custom",
                                      class_weight="balanced" if balanced else None, attack_limit=120,
                                      progress=lambda f, m: bar.progress(min(f, 1.0), m),
                                      extra_meta={"subtitle": f"{up.name} · target {target}"})
            except Exception as e:  # noqa: BLE001
                st.error(f"Build failed: {e}")
                st.stop()
            save_bundle("custom", bundle, RUNTIME_ARTIFACT_DIR)
            db.clear("custom")
            state.seed_domain(db, "custom")
            st.cache_data.clear()
            m = bundle["meta"]["metrics"]
            st.success(f"Built '{title}': accuracy {m['accuracy']:.3f}, ROC-AUC {m['roc_auc']:.3f}, "
                       f"{bundle['meta']['attack_success']}/{bundle['meta']['attack_total']} attacks succeeded.")
            st.button("Switch to this domain", type="primary", on_click=state.set_domain, args=("custom",))
    if bundle_path("custom"):
        m = state.meta("custom") or {}
        md(f"<div style='margin-top:10px'>{badge('custom domain loaded: ' + str(m.get('title', 'Custom')), 'accent')}</div>")
        if st.button("Remove custom domain", icon=":material/delete:"):
            shutil.rmtree(RUNTIME_ARTIFACT_DIR / "custom", ignore_errors=True)
            db.clear("custom")
            if st.session_state.get("domain") == "custom":
                st.session_state["domain"] = "finance"
            st.cache_data.clear()
            st.rerun()
