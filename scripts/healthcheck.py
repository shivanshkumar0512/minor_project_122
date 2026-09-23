"""Lightweight health check: artifacts load, SEAF scores a record, DB is writable.

    python scripts/healthcheck.py        # exit code 0 = healthy
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from seaf.artifacts import bundle_path, load_bundle
    from seaf.audit import AuditLog
    from seaf.config import DOMAINS

    ok = True
    for key in DOMAINS:
        if bundle_path(key) is None:
            print(f"[FAIL] {key}: artifact missing")
            ok = False
            continue
        t0 = time.perf_counter()
        b = load_bundle(key)
        r = b["seaf"].score(b["X_eval"].values[:1])
        good = 0.0 <= float(r.T[0]) <= 1.0
        ok &= good
        print(f"[{'OK' if good else 'FAIL'}] {key}: T={r.T[0]:.3f} in {time.perf_counter() - t0:.2f}s")
    try:
        log = AuditLog(Path(tempfile.gettempdir()) / "seaf_health.db")
        print(f"[{'OK' if log.healthy() else 'FAIL'}] audit DB writable")
        ok &= log.healthy()
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] audit DB: {e}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
