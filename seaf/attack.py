"""Adversaries used to exercise SEAF.

(a) Prediction-preserving attack (paper Section 3.4)::

        max ||phi(x') - phi(x)||   s.t.  y_hat(x') = y_hat(x),  |p(x') - p(x)| <= tau

    Random Forests are piecewise constant, so gradients vanish; the attack is a
    constrained random search.  40 candidates at +-35 % on numeric features are
    screened with a vectorised ``predict_proba`` call and only survivors are
    explained; the survivor with the largest *standardised* attribution
    displacement is kept.

(b) Evasion attack: flip the decision with the smallest relative perturbation,
    searching over progressively larger radii.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import ATTACK_BUDGET, ATTACK_CANDIDATES, ATTACK_TAU, SEED
from .model import positive_proba


@dataclass
class AttackResult:
    kind: str
    x_orig: np.ndarray
    x_adv: np.ndarray
    success: bool
    p_orig: float
    p_adv: float
    displacement: float = 0.0          # standardised attribution displacement
    perturbation: float = 0.0          # mean |relative change| on numeric cols
    n_candidates: int = 0
    n_survivors: int = 0
    trace: list[float] = field(default_factory=list)   # best-so-far (for UI)

    @property
    def delta_p(self) -> float:
        return float(self.p_adv - self.p_orig)


def _perturb(x: np.ndarray, numeric_idx: np.ndarray, integer_idx: np.ndarray,
             n: int, budget: float, rng: np.random.Generator) -> np.ndarray:
    cand = np.repeat(np.atleast_2d(x).astype(float), n, axis=0)
    cand[:, numeric_idx] *= 1.0 + rng.uniform(-budget, budget, size=(n, len(numeric_idx)))
    if len(integer_idx):
        cand[:, integer_idx] = np.round(cand[:, integer_idx])
    return cand


def _rel_change(x: np.ndarray, X_adv: np.ndarray, numeric_idx: np.ndarray) -> np.ndarray:
    base = np.where(np.abs(x[numeric_idx]) > 1e-9, np.abs(x[numeric_idx]), 1.0)
    return (np.abs(X_adv[:, numeric_idx] - x[numeric_idx]) / base).mean(axis=1)


def prediction_preserving_attack_batch(
    model,
    explainer,
    phi_scale: np.ndarray,
    X: np.ndarray,
    numeric_idx: np.ndarray,
    integer_idx: np.ndarray | None = None,
    n_candidates: int = ATTACK_CANDIDATES,
    budget: float = ATTACK_BUDGET,
    tau: float = ATTACK_TAU,
    seed: int = SEED,
) -> list[AttackResult]:
    """Attack many records with one predict call and one SHAP call in total."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    integer_idx = np.asarray(integer_idx if integer_idx is not None else [], dtype=int)
    integer_idx = np.intersect1d(integer_idx, numeric_idx)
    rng = np.random.default_rng(seed)
    n, d = X.shape

    cands = np.vstack([_perturb(x, numeric_idx, integer_idx, n_candidates, budget, rng) for x in X])
    p0 = positive_proba(model, X)
    pc = positive_proba(model, cands).reshape(n, n_candidates)
    same = (pc >= 0.5) == (p0[:, None] >= 0.5)
    ok = same & (np.abs(pc - p0[:, None]) <= tau)

    # SHAP only for the originals plus surviving candidates
    surv_rows = cands.reshape(n, n_candidates, d)[ok]
    phi_all = explainer.shap_values(np.vstack([X, surv_rows]))
    phi0, phi_s = phi_all[:n], phi_all[n:]

    results, cursor = [], 0
    for i in range(n):
        k = int(ok[i].sum())
        if k == 0:
            results.append(AttackResult("prediction_preserving", X[i], X[i].copy(), False,
                                        float(p0[i]), float(p0[i]), 0.0, 0.0, n_candidates, 0))
            continue
        ph = phi_s[cursor:cursor + k]
        disp = np.linalg.norm((ph - phi0[i]) / phi_scale, axis=1)
        j = int(np.argmax(disp))
        x_adv = cands.reshape(n, n_candidates, d)[i][ok[i]][j]
        results.append(AttackResult(
            "prediction_preserving", X[i], x_adv, True, float(p0[i]),
            float(pc[i][ok[i]][j]), float(disp[j]),
            float(_rel_change(X[i], x_adv[None], numeric_idx)[0]),
            n_candidates, k, trace=np.maximum.accumulate(disp).tolist(),
        ))
        cursor += k
    return results


def prediction_preserving_attack(model, explainer, phi_scale, x, numeric_idx, integer_idx=None,
                                 n_candidates=ATTACK_CANDIDATES, budget=ATTACK_BUDGET,
                                 tau=ATTACK_TAU, seed=SEED) -> AttackResult:
    return prediction_preserving_attack_batch(model, explainer, phi_scale, np.atleast_2d(x), numeric_idx,
                                              integer_idx, n_candidates, budget, tau, seed)[0]


def evasion_attack(
    model,
    x: np.ndarray,
    numeric_idx: np.ndarray,
    integer_idx: np.ndarray | None = None,
    radii: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75),
    n_candidates: int = 300,
    seed: int = SEED,
) -> AttackResult:
    """Find the smallest-radius perturbation that flips the predicted class."""
    x = np.asarray(x, dtype=float)
    integer_idx = np.intersect1d(np.asarray(integer_idx if integer_idx is not None else [], dtype=int), numeric_idx)
    rng = np.random.default_rng(seed)
    p0 = float(positive_proba(model, x[None])[0])
    y0 = p0 >= 0.5
    total = 0
    for r in radii:
        cands = _perturb(x, numeric_idx, integer_idx, n_candidates, r, rng)
        pc = positive_proba(model, cands)
        total += n_candidates
        flipped = (pc >= 0.5) != y0
        if flipped.any():
            rel = _rel_change(x, cands, numeric_idx)
            rel[~flipped] = np.inf
            j = int(np.argmin(rel))
            return AttackResult("evasion", x, cands[j], True, p0, float(pc[j]), 0.0,
                                float(rel[j]), total, int(flipped.sum()))
    return AttackResult("evasion", x, x.copy(), False, p0, p0, 0.0, 0.0, total, 0)
