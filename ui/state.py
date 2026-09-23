"""Shared app state: cached artifact loading, audit DB, thresholds, feedback.

Memory discipline for free tiers: a domain bundle (model + SEAF + SHAP
explainer) is loaded lazily on first use and ``st.cache_resource`` keeps at
most two resident, so all three explainers are never in memory at once.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from seaf.artifacts import bundle_path, load_bundle, load_meta, load_seed_records
from seaf.audit import AuditLog
from seaf.config import DOMAINS, FPR_BUDGET
from seaf.feedback import Calibration, rates, recalibrate

DOMAIN_KEYS = list(DOMAINS)


# --------------------------------------------------------------------------- #
# Domain runtime
# --------------------------------------------------------------------------- #
@dataclass
class DomainRuntime:
    key: str
    title: str
    subtitle: str
    icon: str
    verb: str
    bundle: dict

    @property
    def schema(self):
        return self.bundle["schema"]

    @property
    def model(self):
        return self.bundle["model"]

    @property
    def seaf(self):
        return self.bundle["seaf"]

    @property
    def meta(self) -> dict:
        return self.bundle["meta"]

    @property
    def X_eval(self) -> pd.DataFrame:
        return self.bundle["X_eval"]

    @property
    def tune_idx(self) -> np.ndarray:
        return self.bundle["tune_idx"]

    @property
    def scores(self) -> pd.DataFrame:
        return self.bundle["scores"]

    def attack_pairs(self) -> pd.DataFrame:
        """Successful pre-computed attacks whose original is in the tuning split."""
        pool = self.bundle["attack_pool"]
        return pool[pool["success"] & pool["eval_idx"].isin(self.tune_idx)]

    def x_clean(self, eval_idx: int) -> np.ndarray:
        return self.X_eval.values[int(eval_idx)].astype(float)

    def x_adv(self, pool_row: int) -> np.ndarray:
        return self.bundle["X_adv"][int(pool_row)].astype(float)

    def labels(self) -> tuple[str, str]:
        return self.schema.class_names


def _token(key: str) -> str:
    p = bundle_path(key)
    return f"{p}:{p.stat().st_mtime_ns}" if p else "missing"


@st.cache_resource(max_entries=2, show_spinner=False)
def _load(key: str, token: str) -> DomainRuntime:
    bundle = load_bundle(key)
    spec = DOMAINS.get(key)
    meta = bundle["meta"]
    rt = DomainRuntime(
        key=key,
        title=spec.title if spec else meta.get("title", key),
        subtitle=spec.subtitle if spec else meta.get("subtitle", "Custom dataset"),
        icon=spec.icon if spec else "upload_file",
        verb=spec.decision_verb if spec else "positive-class probability",
        bundle=bundle,
    )
    _ = rt.seaf.explainer.expected_value  # build TreeExplainer once, inside the cache
    return rt


def runtime(key: str | None = None) -> DomainRuntime:
    key = key or current_domain()
    return _load(key, _token(key))


def available_domains() -> list[str]:
    keys = [k for k in DOMAIN_KEYS if bundle_path(k)]
    if bundle_path("custom"):
        keys.append("custom")
    return keys


@st.cache_data(show_spinner=False, ttl=600)
def domain_meta(key: str, token: str) -> dict | None:
    return load_meta(key)


def meta(key: str) -> dict | None:
    return domain_meta(key, _token(key))


def domain_title(key: str) -> str:
    if key in DOMAINS:
        return DOMAINS[key].title
    m = meta(key) or {}
    return m.get("title", key.title())


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
def current_domain() -> str:
    doms = available_domains()
    d = st.session_state.get("domain")
    if d not in doms:
        d = doms[0] if doms else "finance"
        st.session_state["domain"] = d
    return d


def set_domain(key: str) -> None:
    """Switch domain.  Call from a widget callback (runs before the sidebar renders)."""
    st.session_state["domain_select"] = key
    if st.session_state.get("domain") != key:
        st.session_state["domain"] = key
        # live-monitor state is per domain
        for k in [k for k in st.session_state if str(k).startswith("live_")]:
            del st.session_state[k]


# --------------------------------------------------------------------------- #
# Audit DB (shared across sessions) + demo seeding
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def db() -> AuditLog:
    log = AuditLog()
    for key in available_domains():
        seed_domain(log, key)
    return log


def seed_domain(log: AuditLog, key: str) -> None:
    """Populate an empty DB so Review Queue / Analytics are never blank."""
    if log.meta_get(f"seeded:{key}"):
        return
    m = load_meta(key) or {}
    default_t = (m.get("calibration") or {}).get("default_threshold")
    if default_t is not None and log.current_threshold(key) is None:
        cal = m["calibration"]
        log.set_threshold(key, default_t, cal.get("default_heldout_detection"),
                          cal.get("default_heldout_fpr"), 0, "initial: 5th pct of baseline trust")
    records = load_seed_records(key)
    rng = np.random.default_rng(11)
    now = time.time()
    rows = []
    order = rng.permutation(len(records))
    for n, i in enumerate(order):
        rec = records[i]
        ts = now - 24 * 3600 + (n + 1) * (23 * 3600 / max(len(records), 1)) + rng.uniform(-600, 600)
        status = None
        if rec["verdict"] == "Review":
            # a realistic mix: most flagged items reviewed, a few left pending
            if rng.random() < 0.55:
                status = "confirmed_attack" if rec["sim_label"] == "attack" else "approved"
        row = AuditLog.make_row(key, "seed", rec, sim_label=rec["sim_label"], record_ref=rec["ref"],
                                ts=ts, status=status)
        if status:
            row["reviewed_ts"] = ts + 900
        rows.append(row)
    if rows:
        log.log(rows)
    log.meta_set(f"seeded:{key}", str(now))


# --------------------------------------------------------------------------- #
# Threshold + feedback loop
# --------------------------------------------------------------------------- #
def threshold(key: str | None = None) -> float:
    key = key or current_domain()
    t = db().current_threshold(key)
    if t is None:
        m = meta(key) or {}
        t = (m.get("calibration") or {}).get("default_threshold", 0.5)
    return float(t)


def labelled_pools(rt: DomainRuntime) -> dict[str, np.ndarray]:
    """Tuning = calibration pool (+ new reviews); held-out never used to select."""
    sc = rt.scores
    sel = lambda kind, split: sc.loc[(sc.kind == kind) & (sc.split == split), "T"].to_numpy()  # noqa: E731
    reviewed = db().query(domain=rt.key, status=["approved", "confirmed_attack"])
    reviewed = reviewed[reviewed.source != "seed"]
    new_clean = reviewed.loc[reviewed.status == "approved", "T"].to_numpy()
    new_attack = reviewed.loc[reviewed.status == "confirmed_attack", "T"].to_numpy()
    return {
        "tune_clean": np.r_[sel("clean", "tune"), new_clean],
        "tune_attack": np.r_[sel("attack", "tune"), new_attack],
        "held_clean": sel("clean", "heldout"),
        "held_attack": sel("attack", "heldout"),
        "n_new": np.array([len(new_clean), len(new_attack)]),
    }


def heldout_rates(rt: DomainRuntime, t: float) -> tuple[float, float]:
    p = labelled_pools(rt)
    return rates(t, p["held_clean"], p["held_attack"])


def recalibrate_domain(rt: DomainRuntime, reason: str, fpr_budget: float = FPR_BUDGET) -> tuple[float, Calibration]:
    """Select a new threshold from labelled outcomes; store and return it."""
    old = threshold(rt.key)
    p = labelled_pools(rt)
    cal = recalibrate(p["tune_clean"], p["tune_attack"], p["held_clean"], p["held_attack"],
                      fpr_budget=fpr_budget, fallback=old)
    db().set_threshold(rt.key, cal.threshold, cal.heldout_detection, cal.heldout_fpr,
                       int(len(p["tune_clean"]) + len(p["tune_attack"])), reason)
    return old, cal


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
