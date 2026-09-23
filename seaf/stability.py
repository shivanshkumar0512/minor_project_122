"""Explanation-stability signal.

For each instance, ``k`` copies are drawn with +-2 % multiplicative jitter on
the numeric (continuous) columns only - an order of magnitude below the
adversarial budget.  All ``n * k`` jittered rows are explained in **one**
batched SHAP call.  Dispersion ``m(x)`` is the mean over features of the
per-feature standard deviation of attributions across the ``k`` copies.

Paper Section 4.4 shows the naive *directional* signal (``m(x)`` itself)
inverts across domains, so SEAF uses the absolute deviation from the baseline
reference: ``S_dev = |m(x) - median_baseline(m)|``.
"""
from __future__ import annotations

import zlib

import numpy as np

from .config import JITTER_SAMPLES, JITTER_SCALE, SEED


def jitter_copies(
    X: np.ndarray,
    numeric_idx: np.ndarray,
    k: int = JITTER_SAMPLES,
    scale: float = JITTER_SCALE,
    seed: int = SEED,
) -> np.ndarray:
    """Return ``(n * k, d)`` jittered rows, grouped per instance.

    Each row's jitter stream is seeded from ``seed`` and a CRC of the row's
    bytes, so a record always receives the same jitter (and therefore the same
    stability score) regardless of which batch it is scored in.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    rep = np.repeat(X, k, axis=0)
    if len(numeric_idx):
        factors = np.empty((len(X), k, len(numeric_idx)))
        for i, row in enumerate(X):
            rng = np.random.default_rng([seed, zlib.crc32(row.tobytes())])
            factors[i] = rng.uniform(-scale, scale, size=(k, len(numeric_idx)))
        rep[:, numeric_idx] *= 1.0 + factors.reshape(len(rep), len(numeric_idx))
    return rep


def dispersion_from_attributions(phi_jit: np.ndarray, n: int, k: int,
                                 scale: np.ndarray | None = None) -> np.ndarray:
    """Mean per-feature std of attributions across each instance's k copies.

    With ``scale`` (the baseline attribution std per feature) the std is
    expressed in baseline sigma units, so low-magnitude features count as much
    as dominant ones.  The paper's reported reference dispersions (0.15-0.57)
    are on this standardised scale.
    """
    phi_jit = np.asarray(phi_jit).reshape(n, k, -1)
    if scale is not None:
        phi_jit = phi_jit / np.asarray(scale, dtype=float)
    return phi_jit.std(axis=1).mean(axis=1)


def dispersion(
    explainer,
    X: np.ndarray,
    numeric_idx: np.ndarray,
    k: int = JITTER_SAMPLES,
    scale: float = JITTER_SCALE,
    seed: int = SEED,
    phi_scale: np.ndarray | None = None,
) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    rows = jitter_copies(X, numeric_idx, k, scale, seed)
    return dispersion_from_attributions(explainer.shap_values(rows), len(X), k, phi_scale)


class StabilityReference:
    """Stores the baseline median dispersion and produces the deviation signal."""

    def fit(self, m_baseline: np.ndarray) -> "StabilityReference":
        m_baseline = np.asarray(m_baseline, dtype=float)
        self.m_ref_ = float(np.median(m_baseline))
        self.baseline_dispersion_ = m_baseline
        return self

    def signal(self, m: np.ndarray) -> np.ndarray:
        """Absolute-deviation stability signal (NOT directional)."""
        return np.abs(np.asarray(m, dtype=float) - self.m_ref_)

    @staticmethod
    def directional(m: np.ndarray) -> np.ndarray:
        """Naive fixed-direction form, kept only for the paper's comparison."""
        return np.asarray(m, dtype=float)
