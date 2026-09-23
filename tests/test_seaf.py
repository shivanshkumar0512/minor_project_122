"""Core library tests: the properties the paper's method depends on."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from seaf import SEAF
from seaf.artifacts import bundle_path, load_bundle
from seaf.attack import evasion_attack, prediction_preserving_attack_batch
from seaf.audit import AuditLog
from seaf.drift import SPIKE, FlagRateMonitor
from seaf.explainer import _positive_slice
from seaf.feedback import rates, select_threshold
from seaf.model import positive_proba
from seaf.preprocessing import prepare_domain
from seaf.stability import StabilityReference, jitter_copies


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def test_shap_summation_identity(toy):
    """base value + sum(phi) == predicted probability (efficiency axiom)."""
    seaf, X = toy["seaf"], toy["X_eval"][:50]
    phi, base = seaf.explain(X)
    p = positive_proba(toy["model"], X)
    assert np.allclose(base + phi.sum(axis=1), p, atol=1e-8)


@pytest.mark.skipif(bundle_path("finance") is None, reason="artifacts not built")
def test_shap_summation_identity_on_shipped_model():
    b = load_bundle("finance")
    X = b["X_eval"].values[:25]
    phi, base = b["seaf"].explain(X)
    assert np.allclose(base + phi.sum(1), positive_proba(b["model"], X), atol=1e-8)
    assert abs(base - 0.3775) < 5e-4          # paper Fig. 2 base rate


def test_explainer_handles_both_shap_layouts():
    a = np.arange(24, dtype=float).reshape(4, 3, 2)          # new: (n, features, classes)
    old = [a[:, :, 0], a[:, :, 1]]                            # old: list per class
    assert np.array_equal(_positive_slice(a, 1, 3), a[:, :, 1])
    assert np.array_equal(_positive_slice(old, 1, 3), a[:, :, 1])


# --------------------------------------------------------------------------- #
# Trust score
# --------------------------------------------------------------------------- #
def test_trust_in_unit_interval(toy):
    res = toy["seaf"].score(toy["X_eval"])
    for arr in (res.T, res.C, res.A, res.S):
        assert np.all((arr >= 0) & (arr <= 1))
    assert np.allclose(res.T, (res.C + (1 - res.A) + (1 - res.S)) / 3)


def test_trust_on_extreme_inputs_still_bounded(toy):
    X = toy["X_eval"][:10] * 50.0        # far outside anything seen
    res = toy["seaf"].score(X)
    assert np.all((res.T >= 0) & (res.T <= 1))


def test_normalisation_uses_clean_baseline_only(toy):
    seaf = toy["seaf"]
    base = seaf.baseline_
    assert seaf.trust.anomaly_norm.lo_ == pytest.approx(np.percentile(base["anomaly_raw"], 5))
    assert seaf.trust.anomaly_norm.hi_ == pytest.approx(np.percentile(base["anomaly_raw"], 95))
    assert seaf.trust.stability_norm.hi_ == pytest.approx(np.percentile(base["stability_raw"], 95))
    assert np.allclose(seaf.detector.standardiser.mean_, seaf.explainer.shap_values(toy["X_base"]).mean(0))
    # scoring (including wildly abnormal data) must never move the reference
    before = (seaf.trust.anomaly_norm.lo_, seaf.trust.anomaly_norm.hi_, seaf.stability.m_ref_)
    seaf.score(toy["X_eval"] * 10)
    assert before == (seaf.trust.anomaly_norm.lo_, seaf.trust.anomaly_norm.hi_, seaf.stability.m_ref_)


def test_audit_record_has_top_sigma_deviations(toy):
    rec = toy["seaf"].score(toy["X_eval"][:1]).record(0)
    assert len(rec["top_deviations"]) == 5
    sig = [abs(t["sigma"]) for t in rec["top_deviations"]]
    assert sig == sorted(sig, reverse=True)


# --------------------------------------------------------------------------- #
# Stability
# --------------------------------------------------------------------------- #
def test_jitter_only_touches_numeric_columns_and_is_small(toy):
    X = toy["X_eval"][:5]
    J = jitter_copies(X, np.array([0, 1, 2, 3]), k=8, scale=0.02).reshape(5, 8, -1)
    assert np.array_equal(J[:, :, 4:], np.repeat(X[:, None, 4:], 8, axis=1))
    rel = np.abs(J[:, :, :4] / X[:, None, :4] - 1)
    assert rel.max() <= 0.02 + 1e-12


def test_jitter_is_deterministic_per_record(toy):
    X = toy["X_eval"][:6]
    a = jitter_copies(X, np.array([0, 1]), k=4)[4 * 3:4 * 4]
    b = jitter_copies(X[3:4], np.array([0, 1]), k=4)
    assert np.array_equal(a, b)


def test_stability_signal_is_absolute_deviation_not_directional():
    ref = StabilityReference().fit(np.array([0.1, 0.2, 0.3]))
    assert ref.m_ref_ == pytest.approx(0.2)
    assert ref.signal(np.array([0.15]))[0] == pytest.approx(ref.signal(np.array([0.25]))[0])


# --------------------------------------------------------------------------- #
# Attacks
# --------------------------------------------------------------------------- #
def test_prediction_preserving_attack_preserves_class_and_budget(toy):
    seaf, X = toy["seaf"], toy["X_eval"][:30]
    atk = prediction_preserving_attack_batch(toy["model"], seaf.explainer, seaf.detector.standardiser.scale_,
                                             X, seaf.numeric_idx, n_candidates=40, budget=0.35, tau=0.06)
    ok = [a for a in atk if a.success]
    assert len(ok) >= 20
    for a in ok:
        p0, p1 = positive_proba(toy["model"], np.vstack([a.x_orig, a.x_adv]))
        assert (p0 >= 0.5) == (p1 >= 0.5)
        assert abs(p1 - p0) <= 0.06 + 1e-12
        changed = np.flatnonzero(~np.isclose(a.x_orig, a.x_adv))
        assert set(changed) <= set(seaf.numeric_idx)                    # numeric features only
        rel = np.abs(a.x_adv[seaf.numeric_idx] / a.x_orig[seaf.numeric_idx] - 1)
        assert rel.max() <= 0.35 + 1e-9
        assert a.displacement > 0


def test_evasion_attack_flips_decision(toy):
    x = toy["X_eval"][0]
    a = evasion_attack(toy["model"], x, toy["seaf"].numeric_idx)
    if a.success:
        p0, p1 = positive_proba(toy["model"], np.vstack([x, a.x_adv]))
        assert (p0 >= 0.5) != (p1 >= 0.5)


# --------------------------------------------------------------------------- #
# Detector, feedback, audit, drift
# --------------------------------------------------------------------------- #
def test_isolation_forest_is_invariant_to_per_feature_standardisation():
    rng = np.random.default_rng(1)
    X = rng.lognormal(size=(300, 4)) * [1, 10, 1e3, 1e5]
    Q = rng.lognormal(size=(40, 4)) * [1, 10, 1e3, 1e5]
    m, s = X.mean(0), X.std(0)
    a = IsolationForest(n_estimators=100, random_state=0).fit(X).score_samples(Q)
    b = IsolationForest(n_estimators=100, random_state=0).fit((X - m) / s).score_samples((Q - m) / s)
    assert np.allclose(a, b)


def test_threshold_selection_respects_fpr_budget():
    rng = np.random.default_rng(3)
    clean, attack = rng.beta(6, 2, 400), rng.beta(3, 3, 120)
    t = select_threshold(clean, attack, 0.10)
    det, fpr = rates(t, clean, attack)
    assert fpr <= 0.10
    # no higher threshold within budget catches more
    for t2 in np.linspace(t, 1, 50):
        d2, f2 = rates(t2, clean, attack)
        if f2 <= 0.10:
            assert d2 <= det + 1e-12


def test_audit_log_roundtrip(tmp_path, toy):
    log = AuditLog(tmp_path / "a.db")
    res = toy["seaf"].score(toy["X_eval"][:3])
    ids = log.log([AuditLog.make_row("toy", "test", res.record(i)) for i in range(3)])
    log.review(ids[0], "confirmed_attack")
    assert log.get(ids[0])["status"] == "confirmed_attack"
    assert log.counts("toy")["total"] == 3
    log.set_threshold("toy", 0.42)
    assert log.current_threshold("toy") == pytest.approx(0.42)


def test_drift_monitor_flags_sudden_spike():
    mon = FlagRateMonitor(short=10, long=60, expected_rate=0.05)
    for _ in range(60):
        mon.update(False, np.zeros(3))
    states = [mon.update(True, np.zeros(3))["state"] for _ in range(8)]
    assert SPIKE in states


# --------------------------------------------------------------------------- #
# Data protocol
# --------------------------------------------------------------------------- #
def test_domain_preparation_matches_paper():
    X, y, s = prepare_domain("finance")
    assert X.shape == (4269, 11)
    X, y, s = prepare_domain("healthcare")
    assert X.shape == (4269, 11)
    X, y, s = prepare_domain("recruitment")
    assert len(X) == 1470 and len(s.categorical_maps) == 7
    assert not {"EmployeeCount", "Over18", "StandardHours"} & set(X.columns)


def test_refit_baseline_is_a_new_object(toy):
    seaf = toy["seaf"]
    new = seaf.refit_baseline(toy["X_eval"][:20])
    assert new is not seaf and new.baseline_["n"] == seaf.baseline_["n"] + 20
    assert isinstance(new, SEAF)
