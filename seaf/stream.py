"""Simulated live decision traffic for the Live Monitor.

Clean records are drawn from the evaluation split.  With probability
``attack_rate`` a record is replaced by a pre-computed prediction-preserving
adversarial version (generated offline so the stream stays smooth on a weak
CPU).  When drift is enabled, selected numeric features are shifted by a
factor that ramps up slowly - a gradual population change, as opposed to the
abrupt, isolated deviations produced by attacks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SEED


@dataclass
class StreamEvent:
    x: np.ndarray
    ref: str
    is_attack: bool
    drift_level: float

    @property
    def label(self) -> str:
        if self.is_attack:
            return "attack"
        return "drift" if self.drift_level > 0.02 else "clean"


class TrafficSimulator:
    def __init__(self, X_clean: np.ndarray, clean_refs: list[str], X_attack: np.ndarray,
                 attack_refs: list[str], seed: int = SEED) -> None:
        self.X_clean = np.asarray(X_clean, float)
        self.clean_refs = list(clean_refs)
        self.X_attack = np.asarray(X_attack, float)
        self.attack_refs = list(attack_refs)
        self.rng = np.random.default_rng(seed)
        self.drift_level = 0.0
        self.n_emitted = 0

    def next(
        self,
        n: int = 1,
        attack_rate: float = 0.1,
        drift_on: bool = False,
        drift_idx: np.ndarray | list[int] = (),
        drift_rate: float = 0.004,
        drift_max: float = 0.45,
    ) -> list[StreamEvent]:
        events = []
        drift_idx = np.asarray(drift_idx, dtype=int)
        for _ in range(int(n)):
            # Gradual ramp while enabled, faster recovery once disabled.
            if drift_on:
                self.drift_level = min(drift_max, self.drift_level + drift_rate)
            else:
                self.drift_level = max(0.0, self.drift_level - 4 * drift_rate)

            if len(self.X_attack) and self.rng.random() < attack_rate:
                i = int(self.rng.integers(len(self.X_attack)))
                events.append(StreamEvent(self.X_attack[i].copy(), self.attack_refs[i], True, 0.0))
            else:
                i = int(self.rng.integers(len(self.X_clean)))
                x = self.X_clean[i].copy()
                if self.drift_level > 0 and len(drift_idx):
                    x[drift_idx] *= 1.0 + self.drift_level
                events.append(StreamEvent(x, self.clean_refs[i], False, self.drift_level))
            self.n_emitted += 1
        return events
