"""SEAF: the public, model-agnostic entry point.

    >>> seaf = SEAF().fit(model, X_baseline, numeric_features=[...])
    >>> result = seaf.score(X_new)
    >>> result.T, result.flagged, result.record(0)["top_deviations"]

``fit`` estimates everything from clean baseline data only.  ``score`` runs the
full inference path: prediction -> TreeSHAP -> isolation anomaly -> stability
-> trust -> verdict.  The protected model is never modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import IFOREST_ESTIMATORS, JITTER_SAMPLES, JITTER_SCALE, SEED
from .detector import IsolationDetector
from .explainer import TreeShapExplainer
from .model import positive_proba
from .stability import StabilityReference, dispersion_from_attributions, jitter_copies
from .trust import TrustScorer, top_deviations

ACCEPT, REVIEW = "Accept", "Review"


@dataclass
class SEAFResult:
    """Vectorised scoring output for ``n`` records."""

    feature_names: list[str]
    X: np.ndarray
    proba: np.ndarray
    pred: np.ndarray
    phi: np.ndarray
    base_value: float
    anomaly_raw: np.ndarray
    dispersion: np.ndarray
    stability_raw: np.ndarray
    stability_measured: np.ndarray
    C: np.ndarray
    A: np.ndarray
    S: np.ndarray
    T: np.ndarray
    z: np.ndarray
    threshold: float
    flagged: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.flagged = self.T < self.threshold

    def __len__(self) -> int:
        return len(self.T)

    def verdicts(self) -> list[str]:
        return [REVIEW if f else ACCEPT for f in self.flagged]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "proba": self.proba, "pred": self.pred, "C": self.C, "A": self.A,
            "S": self.S, "T": self.T, "anomaly_raw": self.anomaly_raw,
            "dispersion": self.dispersion, "stability_raw": self.stability_raw,
            "stability_measured": self.stability_measured, "flagged": self.flagged,
        })

    def record(self, i: int, k: int = 5) -> dict[str, Any]:
        """JSON-serialisable audit record for one decision."""
        return {
            "features": {f: float(v) for f, v in zip(self.feature_names, self.X[i])},
            "proba": float(self.proba[i]),
            "pred": int(self.pred[i]),
            "base_value": float(self.base_value),
            "phi": {f: float(v) for f, v in zip(self.feature_names, self.phi[i])},
            "C": float(self.C[i]), "A": float(self.A[i]), "S": float(self.S[i]),
            "T": float(self.T[i]),
            "anomaly_raw": float(self.anomaly_raw[i]),
            "dispersion": float(self.dispersion[i]),
            "stability_measured": bool(self.stability_measured[i]),
            "threshold": float(self.threshold),
            "verdict": REVIEW if self.flagged[i] else ACCEPT,
            "top_deviations": top_deviations(self.z[i], self.feature_names, k),
        }


class SEAF:
    """Secure Explainable AI Framework around a fitted tree-ensemble classifier."""

    def __init__(
        self,
        n_jitter: int = JITTER_SAMPLES,
        jitter_scale: float = JITTER_SCALE,
        if_estimators: int = IFOREST_ESTIMATORS,
        seed: int = SEED,
    ) -> None:
        self.n_jitter = n_jitter
        self.jitter_scale = jitter_scale
        self.if_estimators = if_estimators
        self.seed = seed
        self.fitted_ = False

    # ------------------------------------------------------------------ #
    def fit(
        self,
        model: Any,
        X_clean: np.ndarray | pd.DataFrame,
        numeric_features: Sequence[str] | Sequence[int] | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "SEAF":
        """Estimate the baseline of normal model reasoning from clean data."""
        if isinstance(X_clean, pd.DataFrame):
            feature_names = list(feature_names or X_clean.columns)
        X = np.asarray(X_clean, dtype=float)
        d = X.shape[1]
        self.feature_names = list(feature_names or [f"x{i}" for i in range(d)])
        self.numeric_idx = self._resolve_idx(numeric_features, d)
        self.model = model
        self.explainer = TreeShapExplainer(model, d)

        phi = self.explainer.shap_values(X)
        self.detector = IsolationDetector(self.if_estimators, self.seed).fit(phi)
        # Input-space detector is fitted on the identical baseline; it is used
        # only for the paper's explanation- vs input-space comparison.
        self.input_detector = IsolationDetector(self.if_estimators, self.seed).fit(X)

        m = self._dispersion(X)
        self.stability = StabilityReference().fit(m)
        anomaly = self.detector.score(phi)
        stab = self.stability.signal(m)
        self.trust = TrustScorer().fit(anomaly, stab)

        p = positive_proba(model, X)
        comps = self.trust.components(p, anomaly, stab)
        self.baseline_ = {
            "n": int(len(X)),
            "T": comps["T"], "C": comps["C"], "A": comps["A"], "S": comps["S"],
            "anomaly_raw": anomaly, "dispersion": m, "stability_raw": stab,
            "phi_mean": self.detector.standardiser.mean_,
            "phi_std": self.detector.standardiser.scale_,
        }
        self.X_baseline_ = X
        # Neutral value for records whose stability was not measured (fast path)
        self.S_neutral_ = float(np.median(comps["S"]))
        # Default threshold: 5th percentile of baseline trust (~5 % clean FPR).
        self.default_threshold = float(np.percentile(comps["T"], 5))
        self.threshold = self.default_threshold
        self.fitted_ = True
        return self

    # ------------------------------------------------------------------ #
    def explain(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        return self.explainer.shap_values(X), self.explainer.expected_value

    def score(
        self,
        X: np.ndarray | pd.DataFrame,
        stability: bool | np.ndarray = True,
        threshold: float | None = None,
    ) -> SEAFResult:
        """Score records.  ``stability`` may be a bool or a per-row mask.

        Rows without measured stability receive the baseline-median S (neutral)
        and are marked ``stability_measured=False``.  All SHAP work for a call
        (originals + jittered copies) happens in a single batched call.
        """
        self._check()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = len(X)
        mask = np.full(n, bool(stability)) if np.isscalar(stability) else np.asarray(stability, bool)
        k = self.n_jitter
        jit = jitter_copies(X[mask], self.numeric_idx, k, self.jitter_scale, self.seed) if mask.any() else np.zeros((0, X.shape[1]))
        phi_all = self.explainer.shap_values(np.vstack([X, jit]))
        phi = phi_all[:n]

        m = np.full(n, np.nan)
        if mask.any():
            m[mask] = dispersion_from_attributions(phi_all[n:], int(mask.sum()), k,
                                                   self.detector.standardiser.scale_)
        p = positive_proba(self.model, X)
        anomaly = self.detector.score(phi)
        stab = np.where(mask, self.stability.signal(np.nan_to_num(m)), np.nan)
        comps = self.trust.components(p, anomaly, np.nan_to_num(stab))
        S = np.where(mask, comps["S"], self.S_neutral_)
        T = (comps["C"] + (1 - comps["A"]) + (1 - S)) / 3.0
        return SEAFResult(
            feature_names=self.feature_names, X=X, proba=p, pred=(p >= 0.5).astype(int),
            phi=phi, base_value=self.explainer.expected_value, anomaly_raw=anomaly,
            dispersion=m, stability_raw=stab, stability_measured=mask,
            C=comps["C"], A=comps["A"], S=S, T=T, z=self.detector.zscores(phi),
            threshold=self.threshold if threshold is None else float(threshold),
        )

    def refit_baseline(self, X_extra_clean: np.ndarray) -> "SEAF":
        """New SEAF on baseline + reviewer-verified clean records (manual only)."""
        self._check()
        X = np.vstack([self.X_baseline_, np.atleast_2d(np.asarray(X_extra_clean, dtype=float))])
        new = SEAF(self.n_jitter, self.jitter_scale, self.if_estimators, self.seed)
        return new.fit(self.model, X, numeric_features=list(self.numeric_idx), feature_names=self.feature_names)

    # ------------------------------------------------------------------ #
    def _dispersion(self, X: np.ndarray) -> np.ndarray:
        rows = jitter_copies(X, self.numeric_idx, self.n_jitter, self.jitter_scale, self.seed)
        return dispersion_from_attributions(self.explainer.shap_values(rows), len(X), self.n_jitter,
                                            self.detector.standardiser.scale_)

    def _resolve_idx(self, cols: Sequence | None, d: int) -> np.ndarray:
        if cols is None:
            return np.arange(d)
        cols = list(cols)
        if cols and isinstance(cols[0], str):
            return np.array([self.feature_names.index(c) for c in cols], dtype=int)
        return np.array(cols, dtype=int)

    def _check(self) -> None:
        if not self.fitted_:
            raise RuntimeError("SEAF is not fitted; call SEAF.fit(model, X_clean) first.")
