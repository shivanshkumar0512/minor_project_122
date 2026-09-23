"""Security Validation Layer: isolation-based detection in a vector space.

Attribution magnitudes differ by over an order of magnitude across features
(credit score dominates the finance model ~20x).  The paper standardises each
dimension with mean/std estimated on the **baseline split only** (Section 3.7)
and the same standardised space is used for the per-feature sigma deviations
in the audit record, the attack's displacement objective and the stability
dispersion.

Note: scikit-learn's Isolation Forest draws each split uniformly between the
node's min and max of one feature, which is invariant to per-feature affine
rescaling - standardisation leaves its scores unchanged (verified in
``tests/test_seaf.py``).  It matters for every *distance*-based quantity above.

The same class is reused on raw inputs to provide the input-space comparison
detector of Table IV.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from .config import IFOREST_ESTIMATORS, SEED

_EPS = 1e-12


class Standardiser:
    """Per-dimension z-scoring with parameters frozen at fit time."""

    def fit(self, V: np.ndarray) -> "Standardiser":
        V = np.asarray(V, dtype=float)
        self.mean_ = V.mean(axis=0)
        std = V.std(axis=0)
        # A dimension that never varies on the baseline gets unit scale so any
        # later deviation is still measurable (and no division by zero).
        self.scale_ = np.where(std > _EPS, std, 1.0)
        return self

    def transform(self, V: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(np.asarray(V, dtype=float)) - self.mean_) / self.scale_


class IsolationDetector:
    """Standardise, then score with an Isolation Forest.

    ``score`` returns the *anomaly* score ``-score_samples(x)``: the negated
    normalised expected path length, so higher means easier to isolate and
    therefore more deviant.
    """

    def __init__(self, n_estimators: int = IFOREST_ESTIMATORS, seed: int = SEED) -> None:
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, V: np.ndarray) -> "IsolationDetector":
        self.standardiser = Standardiser().fit(V)
        self.forest = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=self.seed,
            n_jobs=1,
        ).fit(self.standardiser.transform(V))
        self.n_baseline_ = int(len(V))
        return self

    def zscores(self, V: np.ndarray) -> np.ndarray:
        return self.standardiser.transform(V)

    def score(self, V: np.ndarray) -> np.ndarray:
        return -self.forest.score_samples(self.standardiser.transform(V))
