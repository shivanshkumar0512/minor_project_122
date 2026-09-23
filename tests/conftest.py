import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seaf import SEAF  # noqa: E402


@pytest.fixture(scope="session")
def toy():
    """Small synthetic problem: fast, deterministic, 6 features (4 numeric)."""
    rng = np.random.default_rng(0)
    n = 900
    X = np.c_[rng.lognormal(10, 0.5, n), rng.normal(600, 90, n), rng.uniform(1, 30, n),
              rng.normal(50, 10, n), rng.integers(0, 2, n), rng.integers(0, 4, n)].astype(float)
    logit = 0.004 * (X[:, 1] - 600) - 0.08 * (X[:, 2] - 15) + 0.6 * X[:, 4] + rng.normal(0, 0.5, n)
    y = (logit > 0).astype(int)
    model = RandomForestClassifier(60, max_depth=8, random_state=42, n_jobs=1).fit(X[:600], y[:600])
    seaf = SEAF(if_estimators=100).fit(model, X[600:750], numeric_features=[0, 1, 2, 3],
                                       feature_names=["income", "score", "tenure", "age", "flag", "level"])
    return {"X": X, "y": y, "model": model, "seaf": seaf, "X_base": X[600:750], "X_eval": X[750:]}
