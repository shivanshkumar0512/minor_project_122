"""Versioned, compressed artifact bundles (one per domain).

Layout::

    artifacts/<domain>/bundle.joblib     model + fitted SEAF + splits + attack pool
    artifacts/<domain>/meta.json         human-readable summary (version, metrics)

Runtime refits (Admin page) are written to ``<runtime_dir>/artifacts/<domain>/`` so
the committed artifacts are never overwritten; the loader prefers the runtime
copy when present.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib

from .config import ARTIFACT_DIR, ARTIFACT_VERSION, runtime_dir

RUNTIME_ARTIFACT_DIR = runtime_dir() / "artifacts"
BUNDLE = "bundle.joblib"
META = "meta.json"
SEED_FILE = "seed_records.json"


def save_bundle(domain: str, bundle: dict[str, Any], base_dir: Path = ARTIFACT_DIR,
                compress: int = 3) -> Path:
    out = Path(base_dir) / domain
    out.mkdir(parents=True, exist_ok=True)
    bundle = dict(bundle)
    bundle.setdefault("version", ARTIFACT_VERSION)
    bundle.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    path = out / BUNDLE
    joblib.dump(bundle, path, compress=("zlib", compress))
    meta = {k: v for k, v in bundle.get("meta", {}).items()}
    meta.update(version=bundle["version"], created=bundle["created"], domain=domain,
                size_mb=round(path.stat().st_size / 1e6, 3))
    (out / META).write_text(json.dumps(meta, indent=2, default=float))
    if bundle.get("seed_records"):
        (out / SEED_FILE).write_text(json.dumps(bundle["seed_records"], default=float))
    return path


def load_seed_records(domain: str) -> list[dict]:
    """Small JSON of pre-scored demo decisions (no model load needed)."""
    for base in (RUNTIME_ARTIFACT_DIR, ARTIFACT_DIR):
        p = Path(base) / domain / SEED_FILE
        if p.exists():
            return json.loads(p.read_text())
    return []


def bundle_path(domain: str) -> Path | None:
    for base in (RUNTIME_ARTIFACT_DIR, ARTIFACT_DIR):
        p = Path(base) / domain / BUNDLE
        if p.exists():
            return p
    return None


def load_bundle(domain: str) -> dict[str, Any]:
    p = bundle_path(domain)
    if p is None:
        raise FileNotFoundError(f"No artifact bundle for domain '{domain}'")
    return joblib.load(p)


def load_meta(domain: str) -> dict[str, Any] | None:
    p = bundle_path(domain)
    if p is None or not (p.parent / META).exists():
        return None
    return json.loads((p.parent / META).read_text())
