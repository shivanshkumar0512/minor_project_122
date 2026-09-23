"""Feedback loop: threshold recalibration from reviewer-confirmed outcomes.

Policy (paper Section 4.6): choose the trust threshold that maximises attack
detection subject to a false-positive budget (default 10 %, representing
finite review capacity).  Ties are broken towards the *lowest* threshold that
achieves the maximum, i.e. the fewest false positives.  The threshold is chosen
on a tuning set and **measured on a held-out set** never used for selection.

A record is flagged when ``T < threshold``.  The baseline distribution is never
modified here; only the decision threshold moves.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FPR_BUDGET


def rates(threshold: float, trust_clean: np.ndarray, trust_attack: np.ndarray) -> tuple[float, float]:
    """(detection rate, false-positive rate) at ``threshold``."""
    tc, ta = np.asarray(trust_clean, float), np.asarray(trust_attack, float)
    det = float((ta < threshold).mean()) if len(ta) else 0.0
    fpr = float((tc < threshold).mean()) if len(tc) else 0.0
    return det, fpr


def select_threshold(trust_clean: np.ndarray, trust_attack: np.ndarray,
                     fpr_budget: float = FPR_BUDGET, fallback: float | None = None) -> float:
    """Max detection s.t. FPR <= budget; ties -> lowest threshold."""
    tc = np.sort(np.asarray(trust_clean, float))
    ta = np.asarray(trust_attack, float)
    if len(tc) == 0:
        return float(fallback) if fallback is not None else 0.5
    # Highest admissible threshold: at most floor(budget * n) clean values below it.
    n_allowed = int(np.floor(fpr_budget * len(tc)))
    t_max = tc[n_allowed] if n_allowed < len(tc) else np.inf
    caught = ta[ta < t_max]
    if len(caught) == 0:
        # No attack can be caught inside the budget: stay at the budget edge
        # just below the next clean value (or fallback when nothing is known).
        return float(fallback if fallback is not None and fallback < t_max else min(t_max, 1.0))
    # Lowest threshold catching all of ``caught``: midpoint between the highest
    # caught attack and the next observed score above it.
    top = caught.max()
    above = np.concatenate([tc[tc > top], ta[ta > top]])
    nxt = min(above.min(), t_max) if len(above) else min(top + 1e-3, t_max)
    return float((top + nxt) / 2.0)


@dataclass
class Calibration:
    threshold: float
    tune_detection: float
    tune_fpr: float
    heldout_detection: float
    heldout_fpr: float
    n_clean: int
    n_attack: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def recalibrate(tune_clean, tune_attack, held_clean, held_attack,
                fpr_budget: float = FPR_BUDGET, fallback: float | None = None) -> Calibration:
    t = select_threshold(tune_clean, tune_attack, fpr_budget, fallback)
    td, tf = rates(t, tune_clean, tune_attack)
    hd, hf = rates(t, held_clean, held_attack)
    return Calibration(t, td, tf, hd, hf, int(len(tune_clean)), int(len(tune_attack)))
