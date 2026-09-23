"""Trust and Risk Scoring Module.

    C = max(p, 1 - p)                              confidence
    A = norm(isolation anomaly of attributions)    anomaly
    S = norm(|m(x) - m_ref|)                       stability deviation
    T = (C + (1 - A) + (1 - S)) / 3                equal-weight trust

``norm`` maps with the 5th / 95th percentiles of the **clean baseline only**
and clips to [0, 1].  Normalising over pooled clean + adversarial ranges would
leak the (deployment-unavailable) attack distribution into the scaling.
"""
from __future__ import annotations

import numpy as np

from .config import NORM_HI, NORM_LO


def confidence(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.maximum(p, 1.0 - p)


class PercentileNormaliser:
    """Min-max scaling between two percentiles of a clean reference sample."""

    def __init__(self, lo: float = NORM_LO, hi: float = NORM_HI) -> None:
        self.lo, self.hi = lo, hi

    def fit(self, clean_values: np.ndarray) -> "PercentileNormaliser":
        v = np.asarray(clean_values, dtype=float)
        self.lo_, self.hi_ = (float(x) for x in np.percentile(v, [self.lo, self.hi]))
        if self.hi_ - self.lo_ < 1e-12:
            self.hi_ = self.lo_ + 1e-12
        return self

    def transform(self, v: np.ndarray) -> np.ndarray:
        return np.clip((np.asarray(v, dtype=float) - self.lo_) / (self.hi_ - self.lo_), 0.0, 1.0)


class TrustScorer:
    """Combines the three signals with equal weights (paper's primary config)."""

    def fit(self, anomaly_clean: np.ndarray, stability_clean: np.ndarray) -> "TrustScorer":
        self.anomaly_norm = PercentileNormaliser().fit(anomaly_clean)
        self.stability_norm = PercentileNormaliser().fit(stability_clean)
        return self

    def components(self, p: np.ndarray, anomaly_raw: np.ndarray, stability_raw: np.ndarray) -> dict[str, np.ndarray]:
        C = confidence(p)
        A = self.anomaly_norm.transform(anomaly_raw)
        S = self.stability_norm.transform(stability_raw)
        T = (C + (1.0 - A) + (1.0 - S)) / 3.0
        return {"C": C, "A": A, "S": S, "T": T}


def top_deviations(z: np.ndarray, feature_names: list[str], k: int = 5) -> list[dict]:
    """Largest per-feature attribution deviations (in baseline sigma) for one record."""
    z = np.asarray(z, dtype=float).ravel()
    order = np.argsort(-np.abs(z))[:k]
    return [{"feature": feature_names[i], "sigma": float(z[i])} for i in order]
