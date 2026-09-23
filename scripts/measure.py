"""Measure load time / memory per domain and the n_estimators trade-off.

    python scripts/measure.py            -> results/deployment_metrics.json
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seaf.artifacts import bundle_path, load_bundle  # noqa: E402
from seaf.config import DOMAINS, RESULTS_DIR  # noqa: E402
from seaf.model import evaluate_model, train_model  # noqa: E402
from seaf.explainer import TreeShapExplainer  # noqa: E402
from seaf.preprocessing import make_splits, prepare_domain  # noqa: E402


def rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return float("nan")


def main() -> None:
    out: dict = {"rss_start_mb": rss_mb()}
    import shap  # noqa: F401  (import cost counted separately)
    out["rss_after_imports_mb"] = rss_mb()
    loads = {}
    for k in DOMAINS:
        t0 = time.perf_counter()
        b = load_bundle(k)
        _ = b["seaf"].explainer.expected_value
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        b["seaf"].score(b["X_eval"].values[:1])
        t_score = time.perf_counter() - t0
        loads[k] = {"bundle_mb": round(bundle_path(k).stat().st_size / 1e6, 2), "load_s": round(t_load, 2),
                    "score_one_with_stability_ms": round(1000 * t_score, 1), "rss_after_mb": round(rss_mb(), 1)}
    out["domains"] = loads

    trade = {}
    for k in ("finance", "healthcare", "recruitment"):
        X, y, _ = prepare_domain(k)
        sp = make_splits(X, y)
        cw = "balanced" if k == "recruitment" else None
        trade[k] = {}
        for n in (100, 200, 300):
            m, _ = train_model(sp.X_train.values, sp.y_train, n_estimators=n, class_weight=cw)
            e = evaluate_model(m, sp.X_test.values, sp.y_test)
            buf = io.BytesIO()
            joblib.dump(m, buf, compress=("zlib", 3))
            ex = TreeShapExplainer(m, X.shape[1])
            t0 = time.perf_counter()
            ex.shap_values(sp.X_eval.values[:100])
            trade[k][n] = {"accuracy": round(e["accuracy"], 4), "roc_auc": round(e["roc_auc"], 4),
                           "model_mb": round(buf.tell() / 1e6, 2),
                           "shap_ms": round((time.perf_counter() - t0) * 10, 2)}
    out["trees_tradeoff"] = trade
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "deployment_metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
