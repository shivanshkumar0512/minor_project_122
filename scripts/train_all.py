"""Train the protected model and fit the SEAF baseline for every domain.

    python scripts/train_all.py                  # all domains
    python scripts/train_all.py finance          # one domain
    python scripts/train_all.py --trees 100      # trade-off study (smaller artifacts)

Artifacts are written to ``artifacts/<domain>/bundle.joblib`` (zlib-compressed,
versioned) together with a readable ``meta.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seaf.artifacts import save_bundle  # noqa: E402
from seaf.config import ARTIFACT_DIR, DOMAINS, MAX_DEPTH, N_ESTIMATORS  # noqa: E402
from seaf.pipeline import build_domain  # noqa: E402
from seaf.preprocessing import prepare_domain  # noqa: E402

CLASS_WEIGHT = {"recruitment": "balanced"}


def train_domain(key: str, trees: int = N_ESTIMATORS, out_dir: Path = ARTIFACT_DIR,
                 balanced: bool = True) -> dict:
    spec = DOMAINS[key]
    X, y, schema = prepare_domain(key)
    print(f"\n[{key}] {len(X)} rows x {X.shape[1]} features | positive rate {y.mean():.3f}")
    print(f"  numeric (jitter/attack): {schema.numeric_features}")

    def progress(frac: float, msg: str) -> None:
        print(f"  {frac:4.0%}  {msg}", flush=True)

    bundle = build_domain(X, y, schema, key=key, title=spec.title,
                          class_weight=CLASS_WEIGHT.get(key) if balanced else None, n_estimators=trees,
                          max_depth=MAX_DEPTH, progress=progress)
    path = save_bundle(key, bundle, out_dir)
    m = bundle["meta"]
    mt = m["metrics"]
    print(f"  acc={mt['accuracy']:.4f} prec={mt['precision']:.4f} rec={mt['recall']:.4f} "
          f"f1={mt['f1']:.4f} auc={mt['roc_auc']:.4f} | shap {mt['shap_ms']:.2f} ms/rec")
    print(f"  attack pool {m['attack_success']}/{m['attack_total']} | threshold default "
          f"{m['calibration']['default_threshold']:.3f} -> {m['calibration']['threshold']:.3f}")
    print(f"  saved {path} ({path.stat().st_size / 1e6:.2f} MB) in {m['build_s']:.1f}s")
    return m


def reseed(key: str, out_dir: Path = ARTIFACT_DIR) -> None:
    """Regenerate only seed_records.json from an existing bundle (fast)."""
    from seaf.artifacts import SEED_FILE, load_bundle
    from seaf.pipeline import make_seed_records

    b = load_bundle(key)
    recs = make_seed_records(b["seaf"], b["X_eval"].values, b["X_adv"], b["attack_pool"], b["scores"], b["tune_idx"])
    (out_dir / key / SEED_FILE).write_text(json.dumps(recs, default=float))
    print(f"[{key}] {len(recs)} seed records, {sum(r['verdict'] == 'Review' for r in recs)} flagged")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("domains", nargs="*", default=list(DOMAINS))
    ap.add_argument("--trees", type=int, default=N_ESTIMATORS)
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    ap.add_argument("--no-balance", action="store_true",
                    help="disable class_weight='balanced' for recruitment (reproduces paper Table II)")
    ap.add_argument("--seed-only", action="store_true", help="only regenerate demo seed records")
    args = ap.parse_args()
    if args.seed_only:
        for k in args.domains:
            reseed(k, args.out)
        return
    t0 = time.time()
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for k in args.domains:
        summary[k] = train_domain(k, args.trees, args.out, balanced=not args.no_balance)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
