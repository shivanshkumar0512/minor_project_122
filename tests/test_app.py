"""Every dashboard page imports and renders without raising (Streamlit AppTest)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from seaf.artifacts import bundle_path

TITLES = {"home": "Verify the explanation", "predict": "Predict &amp; verify", "attack": "Attack simulator",
          "monitor": "Live monitor", "review": "Review queue", "analytics": "Analytics", "audit": "Audit log",
          "admin": "Admin"}
PAGES = list(TITLES)
needs_artifacts = pytest.mark.skipif(bundle_path("finance") is None, reason="artifacts not built")


@pytest.mark.parametrize("mod", ["ui.theme", "ui.components", "ui.charts", "ui.state", "ui.record", "ui.layout"])
def test_ui_modules_import(mod):
    importlib.import_module(mod)


@needs_artifacts
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SEAF_DB_PATH", str(tmp_path / "t.db"))
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=120)
    at.run()
    if page != "home":
        at.switch_page(f"app_pages/{page}.py")
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "Something went wrong" not in text
    assert TITLES[page] in text
