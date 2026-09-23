"""Rolling-window monitor separating gradual drift from sudden attack bursts.

The paper (Section 5) notes that attribution-space deviation does not by
itself distinguish adversarial manipulation from distribution shift.  This
monitor adds the temporal view it calls for:

* **short / long flag rates** - an attack burst lifts the short-window rate
  abruptly above the long-window rate ("spike");
* **short-window trust drop** - per-record detection is probabilistic, so a
  burst may raise few flags; the recent *mean trust* is therefore also tested
  against the preceding window (z-score);
* **consistent input shift** - drift moves the rolling mean of standardised
  inputs in one direction (t-statistic over the long window).  Random-sign
  adversarial perturbations average out in that mean, population drift does
  not, so a shifting population is labelled "drift" rather than "spike".
"""
from __future__ import annotations

from collections import deque

import numpy as np

NORMAL, SPIKE, DRIFT = "normal", "spike", "drift"


def _mwu_z(recent: np.ndarray, ref: np.ndarray) -> float:
    """Normal-approximation z of the Mann-Whitney U statistic (tie-corrected ranks).

    Negative when ``recent`` tends to be smaller than ``ref``.
    """
    n1, n2 = len(recent), len(ref)
    allv = np.concatenate([recent, ref])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    u = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    n = n1 + n2
    tie = (counts ** 3 - counts).sum() / (n * (n - 1)) if n > 1 else 0.0
    sd = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie))
    return float((u - n1 * n2 / 2) / sd) if sd > 0 else 0.0


class FlagRateMonitor:
    """Streaming classifier of the recent past into normal / spike / drift.

    * spike  - recent flag rate jumps, or recent trust is significantly lower
               (one-sided rank test) than the preceding part of the window (self-referencing: the baseline
               split's trust is in-sample for the Isolation Forest, so it is not
               used as the reference);
    * drift  - the recent mean of standardised inputs has moved consistently
               away from a reference captured when the stream started.
    A state must hold for ``debounce`` consecutive decisions before it is
    reported, which suppresses one-off flicker.

    Honest limit: a burst is only detectable if individual attacked decisions
    have lower trust on average - strong in healthcare, weak in finance and
    absent in recruitment (see Analytics, per-signal AUC).
    """

    def __init__(self, short: int = 20, long: int = 100, expected_rate: float = 0.05,
                 shift_t: float = 3.5, trust_z: float = 2.5, min_ref: int = 30,
                 ref_size: int = 40, recent: int = 30, debounce: int = 3) -> None:
        self.short, self.long = short, long
        self.expected = expected_rate
        self.shift_t, self.trust_z = shift_t, trust_z
        self.min_ref, self.ref_size, self.recent, self.debounce = min_ref, ref_size, recent, debounce
        self.flags: deque[int] = deque(maxlen=long)
        self.trust: deque[float] = deque(maxlen=long)
        self.z_recent: deque[np.ndarray] = deque(maxlen=recent)
        self.z_ref: list[np.ndarray] = []
        self.history: list[dict] = []
        self._raw: deque[str] = deque(maxlen=debounce)
        self._state = NORMAL

    def update(self, flagged: bool, z_input: np.ndarray | None = None, trust: float | None = None) -> dict:
        self.flags.append(int(bool(flagged)))
        if trust is not None:
            self.trust.append(float(trust))
        if z_input is not None:
            z = np.asarray(z_input, float)
            if len(self.z_ref) < self.ref_size:
                self.z_ref.append(z)
            else:
                self.z_recent.append(z)
        f = np.fromiter(self.flags, float)
        short_rate, long_rate = float(f[-self.short:].mean()), float(f.mean())
        ref_ok = len(f) >= self.short + self.min_ref
        ref_rate = float(f[:-self.short].mean()) if ref_ok else long_rate

        # sudden trust drop: recent vs preceding window, one-sided Mann-Whitney U
        # (rank-based, so it stays calibrated for skewed trust distributions);
        # reported as a signed z-score (negative = recent trust lower)
        tz = 0.0
        if ref_ok and len(self.trust) == len(f):
            t = np.fromiter(self.trust, float)
            tz = _mwu_z(t[-self.short:], t[:-self.short])

        # consistent input shift vs the stream's own starting reference (Welch-style t)
        shift = 0.0
        if len(self.z_recent) >= self.recent and len(self.z_ref) >= self.ref_size:
            R, Q = np.vstack(self.z_ref), np.vstack(self.z_recent)
            se = np.sqrt(R.var(0) / len(R) + Q.var(0) / len(Q))
            shift = float(np.max(np.abs(Q.mean(0) - R.mean(0)) / np.maximum(se, 0.05)))

        raw = NORMAL
        if shift >= self.shift_t:
            raw = DRIFT
        elif (ref_ok and short_rate >= max(3 * self.expected, ref_rate + 0.2)) or tz <= -self.trust_z:
            raw = SPIKE
        self._raw.append(raw)
        if len(self._raw) == self.debounce and len(set(self._raw)) == 1:
            self._state = raw
        snap = {"short_rate": short_rate, "long_rate": long_rate, "ref_rate": ref_rate,
                "input_shift": shift, "trust_z": tz, "state": self._state}
        self.history.append(snap)
        return snap
