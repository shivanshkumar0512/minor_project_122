"""The protected classifier (AI Decision Layer).

A Random Forest is used because TreeSHAP returns its attributions in
probability space, so base value + sum(phi) == predicted probability exactly
(paper Section 3.4).  SEAF itself never modifies or retrains this model.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import MAX_DEPTH, N_ESTIMATORS, SEED


def build_model(
    n_estimators: int = N_ESTIMATORS,
    max_depth: int = MAX_DEPTH,
    class_weight: str | None = None,
    seed: int = SEED,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=n_jobs,
    )


def train_model(X: np.ndarray, y: np.ndarray, **kwargs: Any) -> tuple[RandomForestClassifier, float]:
    """Fit on a plain array (no feature names -> no sklearn name warnings)."""
    model = build_model(**kwargs)
    t0 = time.perf_counter()
    model.fit(np.asarray(X, dtype=float), y)
    elapsed = time.perf_counter() - t0
    # Inference is single-record in the app; avoid thread-pool overhead there.
    model.set_params(n_jobs=1)
    return model, elapsed


def positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Probability of the positive class (label 1)."""
    proba = model.predict_proba(np.asarray(X, dtype=float))
    classes = list(getattr(model, "classes_", [0, 1]))
    return proba[:, classes.index(1)]


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> dict:
    """Table II / III style metrics on held-out data."""
    X = np.asarray(X, dtype=float)
    t0 = time.perf_counter()
    p = positive_proba(model, X)
    infer_ms = (time.perf_counter() - t0) / len(X) * 1000
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "inference_ms": float(infer_ms),
    }
