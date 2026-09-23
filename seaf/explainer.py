"""Explainability Engine: a thin, version-tolerant TreeSHAP wrapper.

SHAP has returned tree attributions in two layouts over its history:

* old: ``list`` with one ``(n, features)`` array per class
* new: a single ``(n, features, classes)`` array

Both are reduced to the ``(n, features)`` attribution of the positive class.
The underlying ``shap.TreeExplainer`` is not pickled; it is rebuilt lazily
after loading, which keeps artifacts small and avoids SHAP version lock-in.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _positive_slice(values: Any, pos: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):                       # old API
        return np.asarray(values[pos], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 3:                                  # (n, features, classes)
        if arr.shape[1] == n_features:
            return arr[:, :, pos]
        return arr[pos]                                # (classes, n, features)
    if arr.ndim == 2:                                  # single-output model
        return arr
    if arr.ndim == 1:
        return arr[None, :]
    raise ValueError(f"Unexpected SHAP output shape {arr.shape}")


def _positive_expected(expected: Any, pos: int) -> float:
    arr = np.atleast_1d(np.asarray(expected, dtype=float))
    return float(arr[pos] if arr.size > 1 else arr[0])


class TreeShapExplainer:
    """Deterministic TreeSHAP attributions for the positive class."""

    def __init__(self, model: Any, n_features: int, chunk_size: int = 2048) -> None:
        self.model = model
        self.n_features = int(n_features)
        self.chunk_size = int(chunk_size)
        classes = list(getattr(model, "classes_", [0, 1]))
        self.positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        self._explainer = None
        self._expected: float | None = None

    # -- lazy construction -------------------------------------------------- #
    @property
    def explainer(self):
        if self._explainer is None:
            import shap  # imported lazily: shap import is ~1-2 s

            self._explainer = shap.TreeExplainer(self.model)
            self._expected = _positive_expected(self._explainer.expected_value, self.positive_index)
        return self._explainer

    @property
    def expected_value(self) -> float:
        _ = self.explainer
        return float(self._expected)  # type: ignore[arg-type]

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Attribution matrix ``(n, features)`` for the positive class."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = []
        for start in range(0, len(X), self.chunk_size):
            chunk = X[start:start + self.chunk_size]
            vals = self.explainer.shap_values(chunk, check_additivity=False)
            out.append(_positive_slice(vals, self.positive_index, self.n_features))
        return np.vstack(out) if out else np.zeros((0, self.n_features))

    # -- pickling ----------------------------------------------------------- #
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_explainer"] = None
        return state
