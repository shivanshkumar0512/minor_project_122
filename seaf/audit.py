"""SQLite audit log, review queue and threshold history.

Every scored decision is appended to ``decisions`` with its full audit record
(features, attributions, C/A/S/T and the top deviating features in sigma).
Low-trust decisions start with ``status='pending'`` and form the review queue.

On free hosting tiers the file system is ephemeral: the database is recreated
(and re-seeded with demo data) whenever the container restarts.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from .config import runtime_dir

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    domain       TEXT    NOT NULL,
    source       TEXT    NOT NULL,          -- predict | stream | attack | seed | demo
    record_ref   TEXT,                      -- e.g. eval row id
    pred         INTEGER, proba REAL,
    C REAL, A REAL, S REAL, T REAL,
    threshold    REAL,
    verdict      TEXT,                      -- Accept | Review
    status       TEXT,                      -- auto_accepted | pending | approved | confirmed_attack
    sim_label    TEXT,                      -- simulation ground truth: clean | attack | drift
    top_json     TEXT,                      -- top deviating features (sigma)
    record_json  TEXT,                      -- full audit record
    reviewed_ts  REAL,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS ix_dec_domain_status ON decisions(domain, status);
CREATE INDEX IF NOT EXISTS ix_dec_ts ON decisions(ts);
CREATE TABLE IF NOT EXISTS thresholds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, domain TEXT NOT NULL, threshold REAL NOT NULL,
    detection REAL, fpr REAL, n_labels INTEGER, reason TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def default_db_path() -> Path:
    """``$SEAF_DB_PATH`` or ``<runtime_dir>/seaf_audit.db``."""
    env = os.environ.get("SEAF_DB_PATH")
    return Path(env) if env else runtime_dir() / "seaf_audit.db"


class AuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with _LOCK:
            conn = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    def log(self, rows: Iterable[dict[str, Any]]) -> list[int]:
        """Insert decisions; each row is a dict from ``make_row``."""
        ids = []
        cols = ["ts", "domain", "source", "record_ref", "pred", "proba", "C", "A", "S", "T",
                "threshold", "verdict", "status", "sim_label", "top_json", "record_json",
                "reviewed_ts", "note"]
        with self._conn() as c:
            for r in rows:
                cur = c.execute(
                    f"INSERT INTO decisions ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [r.get(k) for k in cols],
                )
                ids.append(int(cur.lastrowid))
        return ids

    @staticmethod
    def make_row(domain: str, source: str, record: dict, sim_label: str | None = None,
                 record_ref: str | None = None, ts: float | None = None,
                 status: str | None = None) -> dict:
        flagged = record["verdict"] == "Review"
        return {
            "ts": ts or time.time(), "domain": domain, "source": source,
            "record_ref": record_ref, "pred": record["pred"], "proba": record["proba"],
            "C": record["C"], "A": record["A"], "S": record["S"], "T": record["T"],
            "threshold": record["threshold"], "verdict": record["verdict"],
            "status": status or ("pending" if flagged else "auto_accepted"),
            "sim_label": sim_label,
            "top_json": json.dumps(record["top_deviations"]),
            "record_json": json.dumps(record),
        }

    def review(self, decision_id: int, outcome: str, note: str | None = None) -> None:
        if outcome not in {"approved", "confirmed_attack"}:
            raise ValueError("outcome must be 'approved' or 'confirmed_attack'")
        with self._conn() as c:
            c.execute("UPDATE decisions SET status=?, reviewed_ts=?, note=? WHERE id=?",
                      (outcome, time.time(), note, int(decision_id)))

    # ------------------------------------------------------------------ #
    def query(self, domain: str | None = None, status: str | list[str] | None = None,
              source: str | None = None, limit: int | None = None,
              order: str = "DESC", with_record: bool = False) -> pd.DataFrame:
        cols = "*" if with_record else ("id, ts, domain, source, record_ref, pred, proba, C, A, S, T, "
                                        "threshold, verdict, status, sim_label, top_json, reviewed_ts, note")
        sql, args = f"SELECT {cols} FROM decisions WHERE 1=1", []
        if domain:
            sql += " AND domain=?"; args.append(domain)
        if status:
            st = [status] if isinstance(status, str) else list(status)
            sql += f" AND status IN ({','.join('?' * len(st))})"; args += st
        if source:
            sql += " AND source=?"; args.append(source)
        sql += f" ORDER BY id {'ASC' if order == 'ASC' else 'DESC'}"
        if limit:
            sql += " LIMIT ?"; args.append(int(limit))
        with self._conn() as c:
            return pd.read_sql_query(sql, c, params=args)

    def get(self, decision_id: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM decisions WHERE id=?", (int(decision_id),)).fetchone()
        return dict(row) if row else None

    def counts(self, domain: str | None = None) -> dict[str, int]:
        sql, args = "SELECT status, COUNT(*) n FROM decisions", []
        if domain:
            sql += " WHERE domain=?"; args.append(domain)
        sql += " GROUP BY status"
        with self._conn() as c:
            out = {r["status"]: int(r["n"]) for r in c.execute(sql, args)}
        out["total"] = sum(out.values())
        return out

    # ------------------------------------------------------------------ #
    def set_threshold(self, domain: str, threshold: float, detection: float | None = None,
                      fpr: float | None = None, n_labels: int | None = None, reason: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO thresholds (ts, domain, threshold, detection, fpr, n_labels, reason) "
                      "VALUES (?,?,?,?,?,?,?)",
                      (time.time(), domain, float(threshold), detection, fpr, n_labels, reason))

    def current_threshold(self, domain: str) -> float | None:
        with self._conn() as c:
            row = c.execute("SELECT threshold FROM thresholds WHERE domain=? ORDER BY id DESC LIMIT 1",
                            (domain,)).fetchone()
        return float(row["threshold"]) if row else None

    def threshold_history(self, domain: str) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query("SELECT * FROM thresholds WHERE domain=? ORDER BY id", c, params=[domain])

    def meta_get(self, key: str) -> str | None:
        with self._conn() as c:
            row = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def meta_set(self, key: str, value: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))

    def clear(self, domain: str | None = None) -> None:
        with self._conn() as c:
            if domain:
                c.execute("DELETE FROM decisions WHERE domain=?", (domain,))
                c.execute("DELETE FROM thresholds WHERE domain=?", (domain,))
                c.execute("DELETE FROM meta WHERE key=?", (f"seeded:{domain}",))
            else:
                c.executescript("DELETE FROM decisions; DELETE FROM thresholds; DELETE FROM meta;")

    def healthy(self) -> bool:
        try:
            with self._conn() as c:
                c.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False
