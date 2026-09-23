"""Evaluate the Live Monitor's spike / drift classifier on simulated streams.

    python scripts/simulate_monitor.py     -> results/monitor_simulation.json

Scenarios per domain and seed (260 decisions, default threshold):
  clean  - no attacks, no drift            (false-alarm check)
  burst  - 8 % background attacks + a burst of 20 attacks at decisions 120-139
  drift  - 8 % background attacks + gradual drift of the two most important
           numeric features from decision 60 (+0.4 % per record, max +45 %)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seaf.artifacts import load_bundle  # noqa: E402
from seaf.config import DOMAINS, RESULTS_DIR  # noqa: E402
from seaf.drift import DRIFT, SPIKE, FlagRateMonitor  # noqa: E402
from seaf.stream import TrafficSimulator  # noqa: E402

N, BURST, DRIFT_START = 260, (120, 140), 60


def run(key: str, scen: str, seed: int) -> dict:
    b = load_bundle(key)
    s, tune, pool = b["seaf"], b["tune_idx"], b["attack_pool"]
    pairs = pool[pool.success & pool.eval_idx.isin(tune)]
    drift_idx = sorted(s.numeric_idx, key=lambda j: -b["importance"][j])[:2]
    sim = TrafficSimulator(b["X_eval"].values[tune], [str(i) for i in tune],
                           b["X_adv"][pairs.index.to_numpy()], [str(i) for i in pairs.eval_idx], seed=seed)
    mon = FlagRateMonitor(expected_rate=0.05)
    states = []
    for t in range(N):
        in_burst = scen == "burst" and BURST[0] <= t < BURST[1]
        rate = 1.0 if in_burst else (0.0 if scen == "clean" else 0.08)
        ev = sim.next(1, rate, scen == "drift" and t >= DRIFT_START, drift_idx)[0]
        r = s.score(ev.x[None])
        z = s.input_detector.zscores(ev.x[None])[0, s.numeric_idx]
        states.append(mon.update(bool(r.flagged[0]), z, float(r.T[0]))["state"])
    st = np.array(states)
    spike_idx = np.flatnonzero(st == SPIKE)
    drift_first = np.flatnonzero(st == DRIFT)
    return {
        "spike_steps": int(len(spike_idx)),
        "spike_in_burst_window": bool(np.any((spike_idx >= BURST[0]) & (spike_idx < BURST[1] + 20))) if scen == "burst" else None,
        "first_spike": int(spike_idx[0]) if len(spike_idx) else None,
        "drift_steps": int((st == DRIFT).sum()),
        "first_drift": int(drift_first[0]) if len(drift_first) else None,
    }


def main() -> None:
    seeds = range(1, 5)
    out: dict = {"n": N, "burst": BURST, "drift_start": DRIFT_START, "domains": {}}
    for key in DOMAINS:
        out["domains"][key] = {scen: [run(key, scen, sd) for sd in seeds] for scen in ("clean", "burst", "drift")}
        d = out["domains"][key]
        print(f"[{key}] clean false-spike runs {sum(r['spike_steps'] > 0 for r in d['clean'])}/{len(seeds)} · "
              f"false-drift runs {sum(r['drift_steps'] > 0 for r in d['clean'])}/{len(seeds)} · "
              f"bursts detected {sum(bool(r['spike_in_burst_window']) for r in d['burst'])}/{len(seeds)} · "
              f"drift detected {sum(r['first_drift'] is not None for r in d['drift'])}/{len(seeds)}", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "monitor_simulation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
