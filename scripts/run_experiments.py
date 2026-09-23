"""Reproduce the paper's experiments from the trained artifacts.

    python scripts/run_experiments.py            # all domains -> results/*.json

Outputs
-------
results/experiments.json      every number below, per domain + cross-domain
results/paper_reference.json  the values reported in the paper, for comparison

Experiments (paper section in brackets)
* protected-model metrics and timings                          [4.1, Tables II-III]
* SHAP summation identity on a representative record          [4.2, Fig. 2]
* prediction-preserving attack success, mean |delta p|        [4.3]
* explanation-space vs input-space detection AUC (paired)     [4.3, Table IV]
* fixed-direction vs absolute-deviation stability AUC          [4.4, Table V]
* mechanism check: prediction dispersion under jitter          [4.4]
* per-signal, equal-weight and fitted trust AUC + weights      [4.5, Table VI]
* threshold recalibration, tuning vs held-out                  [4.6]
* baseline-contamination dose-response                         [4.6]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from seaf.artifacts import load_bundle  # noqa: E402
from seaf.attack import prediction_preserving_attack_batch  # noqa: E402
from seaf.config import DOMAINS, JITTER_SAMPLES, N_ATTACKED_PAPER, RESULTS_DIR, SEED  # noqa: E402
from seaf.detector import IsolationDetector  # noqa: E402
from seaf.feedback import rates, recalibrate  # noqa: E402
from seaf.model import positive_proba  # noqa: E402
from seaf.stability import jitter_copies  # noqa: E402

CONTAMINATION_LEVELS = [0.0, 0.018, 0.059, 0.176, 0.468, 0.995]

PAPER = {
    "model": {
        "finance": {"accuracy": 0.9824, "precision": 0.9873, "recall": 0.9659, "f1": 0.9765, "roc_auc": 0.9975},
        "healthcare": {"accuracy": 0.7447, "precision": 0.7790, "recall": 0.6800, "f1": 0.7261, "roc_auc": 0.8101},
        "recruitment": {"accuracy": 0.8333, "precision": 0.4167, "recall": 0.1064, "f1": 0.1695, "roc_auc": 0.7873},
    },
    "attack": {"success": 179, "total": 180, "mean_abs_dp": {"finance": 0.027, "healthcare": 0.031, "recruitment": 0.032}},
    "space_auc": {
        "finance": {"explanation": 0.7366, "input": 0.6153},
        "healthcare": {"explanation": 0.7295, "input": 0.8119},
        "recruitment": {"explanation": 0.6349, "input": 0.5577},
    },
    "stability_auc": {
        "finance": {"fixed": 0.6114, "absolute": 0.5636, "m_ref": 0.152},
        "healthcare": {"fixed": 0.2619, "absolute": 0.7005, "m_ref": 0.567},
        "recruitment": {"fixed": 0.5142, "absolute": 0.5758, "m_ref": 0.243},
        "range": {"fixed": 0.3495, "absolute": 0.1369},
    },
    "signal_auc": {
        "finance": {"confidence": 0.707, "anomaly": 0.658, "stability": 0.579, "trust_equal": 0.635, "trust_fitted": 0.621},
        "healthcare": {"confidence": 0.520, "anomaly": 0.820, "stability": 0.721, "trust_equal": 0.870, "trust_fitted": 0.851},
        "recruitment": {"confidence": 0.591, "anomaly": 0.547, "stability": 0.645, "trust_equal": 0.614, "trust_fitted": 0.635},
    },
    "fitted_weights": {
        "finance": [0.350, 1.042, 0.548], "healthcare": [-0.336, 2.313, 1.629], "recruitment": [-0.108, -0.016, 1.164],
    },
    "feedback": {"detection_before": 0.10, "detection_after": 0.40, "fpr_before": 0.05, "fpr_after": 0.13},
    "contamination": {"levels": CONTAMINATION_LEVELS, "auc": [0.8050, 0.8043, 0.8027, 0.8020, 0.7937, 0.7827]},
}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def paired_ci(s0: np.ndarray, s1: np.ndarray, n_boot: int = 1000, seed: int = SEED) -> list[float]:
    """95 % bootstrap CI of the paired AUC (resampling attack/original pairs)."""
    rng = np.random.default_rng(seed)
    n, out = len(s0), []
    y = np.r_[np.zeros(n), np.ones(n)]
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        out.append(roc_auc_score(y, np.r_[s0[i], s1[i]]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def seed_spread(b: dict, seeds: range = range(1, 6)) -> dict:
    """Re-draw the 60 attacked instances and attack seed; report AUC spread."""
    model, seaf, schema = b["model"], b["seaf"], b["schema"]
    X_eval = b["X_eval"].values
    vals: dict[str, list[float]] = {"explanation": [], "input": [], "trust_equal": [], "stability_abs": []}
    for sd in seeds:
        rows = np.random.default_rng(sd).permutation(len(X_eval))[:N_ATTACKED_PAPER]
        atk = prediction_preserving_attack_batch(model, seaf.explainer, seaf.detector.standardiser.scale_,
                                                 X_eval[rows], seaf.numeric_idx, schema.integer_idx, seed=sd)
        ok = np.array([a.success for a in atk])
        X0, X1 = X_eval[rows][ok], np.vstack([a.x_adv for a in atk])[ok]
        r0, r1 = seaf.score(X0), seaf.score(X1)
        y = np.r_[np.zeros(len(r0)), np.ones(len(r1))]
        vals["explanation"].append(auc(y, np.r_[r0.anomaly_raw, r1.anomaly_raw]))
        vals["input"].append(auc(y, np.r_[seaf.input_detector.score(X0), seaf.input_detector.score(X1)]))
        vals["trust_equal"].append(auc(y, 1 - np.r_[r0.T, r1.T]))
        vals["stability_abs"].append(auc(y, np.r_[r0.stability_raw, r1.stability_raw]))
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)), "min": float(np.min(v)),
                "max": float(np.max(v)), "values": v} for k, v in vals.items()}


def run_domain(key: str, b: dict | None = None, full: bool = True) -> dict:
    t0 = time.perf_counter()
    b = b or load_bundle(key)
    model, seaf, schema, meta = b["model"], b["seaf"], b["schema"], b["meta"]
    X_eval = b["X_eval"].values
    rng = np.random.default_rng(SEED)
    rows = rng.permutation(len(X_eval))[:N_ATTACKED_PAPER]

    # ---- attack ---------------------------------------------------------- #
    atk = prediction_preserving_attack_batch(model, seaf.explainer, seaf.detector.standardiser.scale_,
                                             X_eval[rows], seaf.numeric_idx, schema.integer_idx, seed=SEED)
    ok = np.array([a.success for a in atk])
    X_orig = X_eval[rows][ok]
    X_adv = np.vstack([a.x_adv for a in atk])[ok]
    dps = np.array([abs(a.delta_p) for a in atk])[ok]
    same_class = all((a.p_orig >= .5) == (a.p_adv >= .5) for a in atk if a.success)

    # ---- paired scoring -------------------------------------------------- #
    r0, r1 = seaf.score(X_orig), seaf.score(X_adv)
    y = np.r_[np.zeros(len(r0)), np.ones(len(r1))]
    cat = lambda a, b: np.r_[a, b]  # noqa: E731
    anomaly = cat(r0.anomaly_raw, r1.anomaly_raw)
    inp = cat(seaf.input_detector.score(X_orig), seaf.input_detector.score(X_adv))
    disp = cat(r0.dispersion, r1.dispersion)
    stab = cat(r0.stability_raw, r1.stability_raw)
    C, A, S, T = (cat(getattr(r0, k), getattr(r1, k)) for k in "CAST")

    # mechanism check: prediction dispersion under the identical jitter
    def pred_disp(X: np.ndarray) -> float:
        rows_j = jitter_copies(X, seaf.numeric_idx, JITTER_SAMPLES, seaf.jitter_scale, SEED)
        return float(positive_proba(model, rows_j).reshape(len(X), -1).std(axis=1).mean())

    # fitted trust: 5-fold CV logistic regression on suspicion components
    F = np.c_[1 - C, A, S]
    lr = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(lr, F, y, cv=cv, method="predict_proba")[:, 1]
    weights = lr.fit(F, y).coef_[0].tolist()

    # ---- feedback (full evaluation pool, tuning vs held-out) ------------- #
    sc = b["scores"]
    sel = lambda kind, split: sc.loc[(sc.kind == kind) & (sc.split == split), "T"].to_numpy()  # noqa: E731
    tc, ta, hc, ha = sel("clean", "tune"), sel("attack", "tune"), sel("clean", "heldout"), sel("attack", "heldout")
    d0_t, f0_t = rates(seaf.default_threshold, tc, ta)
    d0_h, f0_h = rates(seaf.default_threshold, hc, ha)
    cal = recalibrate(tc, ta, hc, ha, fallback=seaf.default_threshold)

    # ---- contamination dose-response ------------------------------------- #
    # Wrongly cleared attacks are *appended* to the clean baseline; the level is
    # the number of added attacks relative to the clean baseline size.
    contamination: list[dict] = []
    phi_eval = np.vstack([r0.phi, r1.phi])
    if full:
        X_base = seaf.X_baseline_
        atk_b = prediction_preserving_attack_batch(model, seaf.explainer, seaf.detector.standardiser.scale_,
                                                   X_base, seaf.numeric_idx, schema.integer_idx, seed=SEED + 1)
        ok_b = np.array([a.success for a in atk_b])
        phi_clean_base = seaf.explainer.shap_values(X_base)
        phi_adv_base = seaf.explainer.shap_values(np.vstack([a.x_adv for a in atk_b]))
        order = np.random.default_rng(SEED).permutation(np.flatnonzero(ok_b))
        for lvl in CONTAMINATION_LEVELS:
            bad = order[:min(int(round(lvl * len(X_base))), len(order))]
            det = IsolationDetector(seaf.if_estimators, SEED).fit(np.vstack([phi_clean_base, phi_adv_base[bad]]))
            contamination.append({"level": float(len(bad) / len(X_base)), "auc": auc(y, det.score(phi_eval))})

    # ---- Fig. 2: summation identity on one record ------------------------ #
    x0 = X_eval[rows[0]][None]
    phi0, base = seaf.explain(x0)
    p0 = float(positive_proba(model, x0)[0])
    top = np.argsort(-np.abs(phi0[0]))
    fig2 = {"base_value": base, "sum_phi": float(phi0[0].sum()), "p": p0,
            "identity_error": float(abs(base + phi0[0].sum() - p0)),
            "features": [schema.feature_names[i] for i in top],
            "phi": [float(phi0[0][i]) for i in top]}

    res = {
        "model": meta["metrics"],
        "fig2": fig2,
        "attack": {"attempted": int(len(atk)), "success": int(ok.sum()), "mean_abs_dp": float(dps.mean()),
                   "max_abs_dp": float(dps.max()), "class_preserved": bool(same_class),
                   "mean_displacement": float(np.mean([a.displacement for a in atk if a.success])),
                   "mean_survivors": float(np.mean([a.n_survivors for a in atk]))},
        "space_auc": {"explanation": auc(y, anomaly), "input": auc(y, inp),
                      "explanation_ci": paired_ci(r0.anomaly_raw, r1.anomaly_raw),
                      "input_ci": paired_ci(inp[:len(r0)], inp[len(r0):])},
        "stability_auc": {
            "fixed": auc(y, disp), "absolute": auc(y, stab), "m_ref": float(seaf.stability.m_ref_),
            "dispersion_clean_mean": float(r0.dispersion.mean()),
            "dispersion_attack_mean": float(r1.dispersion.mean()),
            "pred_dispersion_clean": pred_disp(X_orig), "pred_dispersion_attack": pred_disp(X_adv),
        },
        "signal_auc": {"confidence": auc(y, 1 - C), "anomaly": auc(y, A), "stability": auc(y, S),
                       "trust_equal": auc(y, 1 - T), "trust_fitted": auc(y, oof),
                       "trust_equal_ci": paired_ci(1 - r0.T, 1 - r1.T)},
        "fitted_weights": weights,
        "feedback": {
            "default_threshold": seaf.default_threshold, "recalibrated_threshold": cal.threshold,
            "tune": {"detection_before": d0_t, "fpr_before": f0_t,
                     "detection_after": cal.tune_detection, "fpr_after": cal.tune_fpr},
            "heldout": {"detection_before": d0_h, "fpr_before": f0_h,
                        "detection_after": cal.heldout_detection, "fpr_after": cal.heldout_fpr},
            "n": {"tune_clean": len(tc), "tune_attack": len(ta), "held_clean": len(hc), "held_attack": len(ha)},
            "_T": {"tune_clean": tc.tolist(), "tune_attack": ta.tolist(),
                   "held_clean": hc.tolist(), "held_attack": ha.tolist()},
        },
        "contamination": contamination,
        "seed_spread": seed_spread(b) if full else None,
        "runtime_s": time.perf_counter() - t0,
    }
    print(f"[{key}] attack {res['attack']['success']}/{res['attack']['attempted']} "
          f"|dp|={res['attack']['mean_abs_dp']:.3f} | expl AUC {res['space_auc']['explanation']:.3f} "
          f"vs input {res['space_auc']['input']:.3f} | stab fixed {res['stability_auc']['fixed']:.3f} "
          f"abs {res['stability_auc']['absolute']:.3f} | trust {res['signal_auc']['trust_equal']:.3f} "
          f"({res['runtime_s']:.0f}s)", flush=True)
    return res


def recruitment_unbalanced() -> dict:
    """Paper Table II is reproduced almost exactly *without* class weighting;
    rebuild that variant in memory and repeat the paired experiment on it."""
    from seaf.pipeline import build_domain
    from seaf.preprocessing import prepare_domain

    X, y, schema = prepare_domain("recruitment")
    b = build_domain(X, y, schema, key="recruitment", title="Recruitment", class_weight=None)
    r = run_domain("recruitment", b, full=False)
    return {k: r[k] for k in ("model", "attack", "space_auc", "stability_auc", "signal_auc")}


def cross_domain(per: dict) -> dict:
    fixed = [per[k]["stability_auc"]["fixed"] for k in per]
    absd = [per[k]["stability_auc"]["absolute"] for k in per]
    # Pooled feedback over the three domains (each domain's own thresholds)
    def pooled(split: str, when: str) -> dict:
        det_n = det_d = fp_n = fp_d = 0
        for k, r in per.items():
            f = r["feedback"]
            t = f["default_threshold"] if when == "before" else f["recalibrated_threshold"]
            c = np.array(f["_T"][f"{'tune' if split == 'tune' else 'held'}_clean"])
            a = np.array(f["_T"][f"{'tune' if split == 'tune' else 'held'}_attack"])
            det_n += int((a < t).sum()); det_d += len(a)
            fp_n += int((c < t).sum()); fp_d += len(c)
        return {"detection": det_n / max(det_d, 1), "fpr": fp_n / max(fp_d, 1)}

    return {
        "attack_success": sum(per[k]["attack"]["success"] for k in per),
        "attack_attempted": sum(per[k]["attack"]["attempted"] for k in per),
        "stability": {"fixed_range": max(fixed) - min(fixed), "absolute_range": max(absd) - min(absd),
                      "fixed_worst": min(fixed), "absolute_worst": min(absd),
                      "fixed_mean": float(np.mean(fixed)), "absolute_mean": float(np.mean(absd))},
        "feedback_pooled": {s: {w: pooled(s, w) for w in ("before", "after")} for s in ("tune", "heldout")},
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    keys = [a for a in sys.argv[1:] if not a.startswith("--")] or list(DOMAINS)
    per = {k: run_domain(k) for k in keys}
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": SEED, "domains": per,
           "cross_domain": cross_domain(per)}
    if "recruitment" in keys and "--no-sensitivity" not in sys.argv:
        out["sensitivity"] = {"recruitment_unbalanced": recruitment_unbalanced()}
    (RESULTS_DIR / "experiments.json").write_text(json.dumps(out, indent=2, default=float))
    (RESULTS_DIR / "paper_reference.json").write_text(json.dumps(PAPER, indent=2))
    cd = out["cross_domain"]
    print(f"\nattack success {cd['attack_success']}/{cd['attack_attempted']} | stability range "
          f"fixed {cd['stability']['fixed_range']:.3f} -> abs {cd['stability']['absolute_range']:.3f}")
    print("pooled feedback (held-out):", json.dumps(cd["feedback_pooled"]["heldout"]))


if __name__ == "__main__":
    main()
