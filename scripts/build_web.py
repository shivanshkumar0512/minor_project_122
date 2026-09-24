"""Assemble the in-browser SEAF app: web/dist/index.html + web/dist/data/*.json.

    python scripts/export_web.py            # model + data export (once per training)
    python scripts/build_web.py             # page (Plotly from the jsDelivr CDN)
    python scripts/build_web.py --local     # page using a local Plotly copy (offline testing)
    node web/tests/verify_engine.js <ref>   # engine vs Python check

The page is self-contained HTML (inline CSS/JS) that fetches the per-domain
model files from ./data/ on demand; it can be published as a static site
(e.g. a claude.ai Artifact, GitHub Pages, any web server).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
DIST = WEB / "dist"
PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
         "&family=JetBrains+Mono:wght@400;500;600&display=swap")

TEMPLATE = """<meta charset="utf-8">
<title>SEAF Security Console</title>
<meta name="description" content="Explanation-space anomaly detection for ML decisions, running in your browser.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="@@FONTS@@">
<style>@@CSS@@</style>
<header class="topbar"><div class="topbar-in">
  <div class="brand"><div class="brand-mark">S</div><div><div class="brand-name">SEAF</div><div class="brand-sub">Secure Explainable AI Framework</div></div></div>
  <nav class="nav" id="nav" aria-label="Pages"></nav>
  <div class="domain-select"><label for="domain" class="pill-info">Domain</label><select id="domain"></select><span class="pill-info" id="pill"></span></div>
</div></header>
<main id="main" aria-live="polite"></main>
<div id="toasts" role="status"></div>
<script type="application/json" id="index-data">@@INDEX@@</script>
<script src="@@PLOTLY@@"></script>
<script id="engine">@@ENGINE@@</script>
<script>@@APP@@</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="reference ./plotly.min.js instead of the CDN")
    args = ap.parse_args()
    DIST.mkdir(parents=True, exist_ok=True)
    (DIST / "data").mkdir(exist_ok=True)
    index = {}
    for p in sorted((WEB / "data").glob("*.json")):
        shutil.copy(p, DIST / "data" / p.name)
        if p.stem == "results":
            continue
        d = json.loads(p.read_text())
        m = d["meta"]
        index[d["key"]] = {"title": d["title"], "subtitle": d["subtitle"], "n_rows": m["n_rows"],
                           "n_features": m["n_features"], "n_baseline": m["n_baseline"],
                           "accuracy": m["metrics"]["accuracy"], "roc_auc": m["metrics"]["roc_auc"],
                           "n_nodes": d["rf"]["n_nodes"]}
    order = ["finance", "healthcare", "recruitment"]
    index = {k: index[k] for k in order if k in index}
    plotly = PLOTLY_CDN
    if args.local:
        import plotly as _p
        shutil.copy(Path(_p.__file__).parent / "package_data" / "plotly.min.js", DIST / "plotly.min.js")
        plotly = "plotly.min.js"
    parts = {"FONTS": FONTS, "PLOTLY": plotly, "INDEX": json.dumps(index),
             "CSS": (WEB / "src" / "style.css").read_text(), "ENGINE": (WEB / "src" / "engine.js").read_text(),
             "APP": (WEB / "src" / "app.js").read_text()}
    html = TEMPLATE
    for k in ("FONTS", "PLOTLY", "INDEX", "CSS", "ENGINE", "APP"):   # code last, so its text is never re-scanned
        html = html.replace(f"@@{k}@@", parts[k])
    (DIST / "index.html").write_text(html)
    print(f"web/dist/index.html {len(html) / 1e3:.0f} kB · data: " +
          ", ".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted((DIST / 'data').glob('*.json'))))


if __name__ == "__main__":
    main()
