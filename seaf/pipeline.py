"""End-to-end builder: data -> model -> SEAF baseline -> attack pool -> bundle.

Used by ``scripts/train_all.py`` for the three paper domains and by the Admin
page for user-uploaded CSVs, so both follow the identical protocol.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from .attack import prediction_preserving_attack_batch
from .config import SEED
from .core import SEAF
from .feedback import rates, recalibrate
from .model import evaluate_model, train_model
from .preprocessing import Schema, make_splits

Progress = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    pass


def make_seed_records(seaf: SEAF, X_eval: np.ndarray, X_adv: np.ndarray, pool: pd.DataFrame,
                      scores: pd.DataFrame, tune_idx: np.ndarray, seed: int = SEED) -> list[dict]:
    """Pre-scored demo decisions (tuning split only) for seeding the audit DB at
    start-up without loading every model or running SHAP.

    The mix is deliberately enriched with low-trust cases so the review queue has
    work in it: random clean records, the lowest-trust clean records (false
    alarms) and attacked records, half chosen among the lowest-trust ones.
    """
    rng = np.random.default_rng(seed + 7)
    tune = scores[scores["split"] == "tune"]
    clean = tune[tune.kind == "clean"].sort_values("T")
    attack = tune[tune.kind == "attack"].sort_values("T")
    low_clean = clean["eval_idx"].to_numpy()[:5]
    rand_clean = rng.choice(np.setdiff1d(clean["eval_idx"].to_numpy(), low_clean), size=min(25, len(clean) - 5), replace=False)
    a_idx = attack["eval_idx"].to_numpy()
    low_att = a_idx[:7]
    rand_att = rng.choice(np.setdiff1d(a_idx, low_att), size=min(5, max(len(a_idx) - 7, 0)), replace=False)
    ev2row = {int(e): int(r) for r, e in zip(pool.index, pool["eval_idx"]) if pool.loc[r, "success"]}
    out: list[dict] = []
    c_idx = np.r_[rand_clean, low_clean].astype(int)
    if len(c_idx):
        r = seaf.score(X_eval[c_idx])
        out += [{"ref": f"eval-{int(i)}", "sim_label": "clean", **r.record(j)} for j, i in enumerate(c_idx)]
    at = [int(i) for i in np.r_[low_att, rand_att] if int(i) in ev2row]
    if at:
        r = seaf.score(X_adv[[ev2row[i] for i in at]])
        out += [{"ref": f"adv-{i}", "sim_label": "attack", **r.record(j)} for j, i in enumerate(at)]
    return out


def build_domain(
    X: pd.DataFrame,
    y: np.ndarray,
    schema: Schema,
    *,
    key: str,
    title: str,
    class_weight: str | None = None,
    n_estimators: int = 300,
    max_depth: int = 10,
    attack_limit: int | None = None,
    progress: Progress = _noop,
    seed: int = SEED,
    extra_meta: dict | None = None,
) -> dict:
    t_start = time.perf_counter()
    splits = make_splits(X, y, seed)
    progress(0.05, "Training Random Forest")
    model, train_s = train_model(splits.X_train.values, splits.y_train, n_estimators=n_estimators,
                                 max_depth=max_depth, class_weight=class_weight, seed=seed)
    metrics = evaluate_model(model, splits.X_test.values, splits.y_test)
    metrics["train_s"] = train_s

    progress(0.20, "Fitting SEAF baseline (TreeSHAP + Isolation Forest + stability)")
    t0 = time.perf_counter()
    seaf = SEAF(seed=seed).fit(model, splits.X_base, numeric_features=schema.numeric_features)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    seaf.explainer.shap_values(splits.X_eval.values[:100])
    metrics["shap_ms"] = (time.perf_counter() - t0) / min(100, len(splits.X_eval)) * 1000

    # Evaluation split -> tuning (stream / examples / recalibration) and held-out
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(splits.X_eval))
    half = len(perm) // 2
    tune_idx, held_idx = np.sort(perm[:half]), np.sort(perm[half:])

    progress(0.45, "Generating prediction-preserving attack pool")
    X_eval = splits.X_eval.values
    atk_rows = np.arange(len(X_eval)) if attack_limit is None else perm[:attack_limit]
    atk = prediction_preserving_attack_batch(
        model, seaf.explainer, seaf.detector.standardiser.scale_, X_eval[atk_rows],
        seaf.numeric_idx, schema.integer_idx, seed=seed)
    pool = pd.DataFrame({
        "eval_idx": atk_rows,
        "success": [a.success for a in atk],
        "p_orig": [a.p_orig for a in atk],
        "p_adv": [a.p_adv for a in atk],
        "displacement": [a.displacement for a in atk],
        "perturbation": [a.perturbation for a in atk],
        "n_survivors": [a.n_survivors for a in atk],
    })
    X_adv = np.vstack([a.x_adv for a in atk])

    progress(0.75, "Scoring clean and attacked evaluation records")
    clean_res = seaf.score(X_eval)
    ok = pool["success"].to_numpy()
    adv_res = seaf.score(X_adv[ok])

    def _frame(res, kind, idx):
        f = res.frame()
        f.insert(0, "kind", kind)
        f.insert(0, "eval_idx", idx)
        f["input_anomaly"] = seaf.input_detector.score(res.X)
        return f

    scores = pd.concat([
        _frame(clean_res, "clean", np.arange(len(X_eval))),
        _frame(adv_res, "attack", atk_rows[ok]),
    ], ignore_index=True)
    scores["split"] = np.where(np.isin(scores["eval_idx"], tune_idx), "tune", "heldout")

    # Default vs recalibrated threshold (tuning -> measured on held-out)
    def _t(kind, split):
        m = (scores["kind"] == kind) & (scores["split"] == split)
        return scores.loc[m, "T"].to_numpy()

    d0, f0 = rates(seaf.default_threshold, _t("clean", "heldout"), _t("attack", "heldout"))
    cal = recalibrate(_t("clean", "tune"), _t("attack", "tune"), _t("clean", "heldout"),
                      _t("attack", "heldout"), fallback=seaf.default_threshold)
    calibration = {"default_threshold": seaf.default_threshold, "default_heldout_detection": d0,
                   "default_heldout_fpr": f0, **cal.as_dict()}

    # Guided-demo pair: tuning record accepted when clean, flagged when attacked.
    demo_idx = None
    cand = scores[(scores["split"] == "tune")]
    c = cand[cand.kind == "clean"].set_index("eval_idx")["T"]
    a = cand[cand.kind == "attack"].set_index("eval_idx")["T"]
    both = pd.DataFrame({"clean": c, "attack": a}).dropna()
    good = both[(both.clean >= max(cal.threshold, seaf.default_threshold) + 0.02)
                & (both.attack < seaf.default_threshold)]
    if len(good):
        demo_idx = int((good.clean - good.attack).idxmax())
    elif len(both):
        demo_idx = int((both.clean - both.attack).idxmax())

    # Global importance (mean |phi| on the baseline) for charts / drift defaults
    importance = np.abs(seaf.explainer.shap_values(splits.X_base.values[:200])).mean(axis=0)

    seed_records = make_seed_records(seaf, X_eval, X_adv, pool, scores, tune_idx, seed)

    progress(1.0, "Done")
    meta = {
        "key": key, "title": title, "n_rows": int(len(X)), "n_features": int(X.shape[1]),
        "n_train": int(len(splits.X_train)), "n_baseline": int(len(splits.X_base)),
        "n_eval": int(len(splits.X_eval)), "positive_rate": float(np.mean(y)),
        "metrics": metrics, "seaf_fit_s": fit_s, "build_s": time.perf_counter() - t_start,
        "attack_success": int(ok.sum()), "attack_total": int(len(ok)),
        "calibration": calibration, "n_estimators": n_estimators, "max_depth": max_depth,
        "class_weight": class_weight,
        **(extra_meta or {}),
    }
    return {
        "meta": meta,
        "schema": schema,
        "model": model,
        "seaf": seaf,
        "X_base": splits.X_base, "y_base": splits.y_base,
        "X_eval": splits.X_eval, "y_eval": splits.y_eval,
        "tune_idx": tune_idx, "held_idx": held_idx,
        "attack_pool": pool, "X_adv": X_adv,
        "scores": scores,
        "demo_idx": demo_idx,
        "importance": importance,
        "seed_records": seed_records,
    }
