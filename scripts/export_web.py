"""Export trained artifacts for the in-browser SEAF build (web/).

    python scripts/export_web.py      -> web/data/<domain>.json, web/data/results.json

Each domain file carries the Random Forest and the explanation-space Isolation
Forest as flat base64 arrays (float64 thresholds so splits match exactly), the
fitted SEAF reference values, the evaluation records the UI exposes (tuning
half only), pre-computed attacks for the live stream, the labelled trust pools
for threshold recalibration and the demo seed decisions.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from seaf.artifacts import load_bundle, load_seed_records  # noqa: E402
from seaf.config import DOMAINS, RESULTS_DIR  # noqa: E402

OUT = ROOT / "web" / "data"


def b64(a: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def forest(trees, feature_maps=None, pos_index=1) -> dict:
    feat, thr, left, right, value, cover, nsamp, offsets = [], [], [], [], [], [], [], []
    base = 0
    for t_i, est in enumerate(trees):
        t = est.tree_
        n = t.node_count
        f = t.feature.astype(np.int64)
        if feature_maps is not None:
            fm = np.asarray(feature_maps[t_i])
            f = np.where(f >= 0, fm[np.clip(f, 0, None)], -1)
        f = np.where(f >= 0, f, -1)
        l, r = t.children_left.astype(np.int64), t.children_right.astype(np.int64)
        feat.append(f); thr.append(t.threshold)
        left.append(np.where(l >= 0, l + base, -1)); right.append(np.where(r >= 0, r + base, -1))
        v = t.value[:, 0, :]
        v = v / np.maximum(v.sum(axis=1, keepdims=True), 1e-300)
        value.append(v[:, pos_index] if v.shape[1] > 1 else t.value[:, 0, 0])
        cover.append(t.weighted_n_node_samples); nsamp.append(t.n_node_samples)
        offsets.append(base)
        base += n
    cat = np.concatenate
    return {"n_trees": len(trees), "n_nodes": int(base), "offsets": offsets,
            "feat": b64(cat(feat), "<i2"), "thr": b64(cat(thr), "<f8"),
            "left": b64(cat(left), "<i4"), "right": b64(cat(right), "<i4"),
            "value": b64(cat(value), "<f8"), "cover": b64(cat(cover), "<f8"),
            "nsamp": b64(cat(nsamp), "<i4")}


def domain(key: str) -> dict:
    b = load_bundle(key)
    model, seaf, schema, meta = b["model"], b["seaf"], b["schema"], b["meta"]
    classes = list(model.classes_)
    rf = forest(model.estimators_, pos_index=classes.index(1))
    iso = seaf.detector.forest
    iso_f = forest(iso.estimators_, feature_maps=iso.estimators_features_)
    iso_f["max_samples"] = int(iso.max_samples_)
    for k in ("value", "cover"):
        iso_f.pop(k)

    X_eval, y_eval = b["X_eval"].values, b["y_eval"]
    tune = [int(i) for i in b["tune_idx"]]
    pool = b["attack_pool"]
    pairs = pool[pool["success"] & pool["eval_idx"].isin(tune)]
    sc = b["scores"]

    def pool_T(kind: str, split: str) -> list[float]:
        return [round(float(v), 6) for v in sc.loc[(sc.kind == kind) & (sc.split == split), "T"]]

    base = seaf.baseline_
    spec = DOMAINS[key]
    return {
        "key": key, "title": spec.title, "subtitle": spec.subtitle, "verb": spec.decision_verb,
        "icon": spec.icon, "class_names": list(schema.class_names),
        "features": schema.feature_names,
        "labels": {f: schema.label(f) for f in schema.feature_names},
        "numeric_idx": [int(i) for i in seaf.numeric_idx],
        "integer_idx": [int(i) for i in schema.integer_idx],
        "categorical": schema.categorical_maps,
        "value_labels": {k: {str(a): v for a, v in d.items()} for k, d in schema.value_labels.items()},
        "display_units": schema.display_units,
        "choices": {k: [float(v) for v in vals] for k, vals in schema.choices.items()},
        "ranges": {k: [float(a), float(b_)] for k, (a, b_) in schema.ranges.items()},
        "importance": [float(v) for v in b["importance"]],
        "meta": {k: meta[k] for k in ("n_rows", "n_features", "n_train", "n_baseline", "n_eval", "metrics",
                                      "attack_success", "attack_total", "calibration")},
        "seaf": {
            "expected_value": float(seaf.explainer.expected_value),
            "phi_mean": base["phi_mean"].tolist(), "phi_scale": base["phi_std"].tolist(),
            "input_mean": seaf.input_detector.standardiser.mean_.tolist(),
            "input_scale": seaf.input_detector.standardiser.scale_.tolist(),
            "m_ref": seaf.stability.m_ref_,
            "A_lo": seaf.trust.anomaly_norm.lo_, "A_hi": seaf.trust.anomaly_norm.hi_,
            "S_lo": seaf.trust.stability_norm.lo_, "S_hi": seaf.trust.stability_norm.hi_,
            "S_neutral": seaf.S_neutral_, "default_threshold": seaf.default_threshold,
            "k": seaf.n_jitter, "jitter": seaf.jitter_scale,
            "baseline_median": {k: float(np.median(base[k])) for k in ("C", "A", "S", "T")},
        },
        "rf": rf, "iso": iso_f,
        "records": {"idx": tune, "X": X_eval[tune].round(6).tolist(), "y": [int(y_eval[i]) for i in tune]},
        "attacks": {"eval_idx": [int(i) for i in pairs.eval_idx],
                    "X": b["X_adv"][pairs.index.to_numpy()].round(6).tolist()},
        "pools": {"tune_clean": pool_T("clean", "tune"), "tune_attack": pool_T("attack", "tune"),
                  "held_clean": pool_T("clean", "heldout"), "held_attack": pool_T("attack", "heldout")},
        "demo_idx": b.get("demo_idx"),
        "seed": load_seed_records(key),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key in DOMAINS:
        d = domain(key)
        p = OUT / f"{key}.json"
        p.write_text(json.dumps(d, separators=(",", ":"), default=float))
        print(f"{key}: {p.stat().st_size / 1e6:.2f} MB ({d['rf']['n_nodes']} RF nodes)")
    res = {"experiments": json.loads((RESULTS_DIR / "experiments.json").read_text()),
           "paper": json.loads((RESULTS_DIR / "paper_reference.json").read_text()),
           "monitor": json.loads((RESULTS_DIR / "monitor_simulation.json").read_text()),
           "deployment": json.loads((RESULTS_DIR / "deployment_metrics.json").read_text())}
    for dm in res["experiments"]["domains"].values():   # drop large raw arrays
        dm["feedback"].pop("_T", None)
    (OUT / "results.json").write_text(json.dumps(res, separators=(",", ":")))
    print("results.json", round((OUT / "results.json").stat().st_size / 1e3), "kB")


if __name__ == "__main__":
    main()
