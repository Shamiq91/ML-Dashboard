"""Persistent run history (SQLite): datasets, training runs and the comparisons built on them.

Sessions live in memory and expire, so every finished training job is saved here with its leaderboard rows, split,
settings and small ROC/PR curves. That is what makes dataset and model comparison possible across sessions and restarts.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from app.errors import BadRequestError, NotFoundError
from app.utils.sanitize import sanitize

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL, name TEXT NOT NULL, profile TEXT NOT NULL,
    first_seen TEXT NOT NULL, last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL, label TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '', task TEXT NOT NULL, target TEXT NOT NULL,
    payload TEXT NOT NULL, curves TEXT
);
CREATE INDEX IF NOT EXISTS runs_dataset ON runs(dataset_id);
"""


def now_iso() -> str:
    """UTC timestamp, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fingerprint(df: pd.DataFrame) -> str:
    """Stable id for 'the same data' regardless of file name: shape, columns and a hash of the leading rows."""
    head = pd.util.hash_pandas_object(df.head(500).astype(str), index=False).to_numpy().tobytes()
    meta = f"{df.shape}|{'|'.join(map(str, df.columns))}".encode()
    return hashlib.sha1(meta + head).hexdigest()


def dataset_profile(name: str, raw: pd.DataFrame) -> dict[str, Any]:
    """Size and quality summary of an uploaded dataset (before any cleaning)."""
    numeric = [c for c in raw.columns if ptypes.is_numeric_dtype(raw[c]) and not ptypes.is_bool_dtype(raw[c])]
    cells = raw.shape[0] * raw.shape[1]
    corr_pair = None
    if len(numeric) >= 2:
        corr = raw[numeric].corr().abs().where(~np.eye(len(numeric), dtype=bool))
        stacked = corr.stack()
        if len(stacked):
            (a, b), v = stacked.idxmax(), float(stacked.max())
            corr_pair = {"a": a, "b": b, "r": round(v, 3)}
    return {
        "name": name, "rows": int(raw.shape[0]), "columns": int(raw.shape[1]), "numeric_columns": len(numeric),
        "categorical_columns": int(raw.shape[1]) - len(numeric),
        "missing_pct": round(float(raw.isna().sum().sum()) / cells * 100, 3) if cells else 0.0,
        "duplicate_rows": int(raw.duplicated().sum()), "top_correlation": corr_pair,
    }


def target_profile(series: pd.Series, task: str) -> dict[str, Any]:
    """Target description saved with a run: class balance (classification) or distribution (regression)."""
    clean = series.dropna()
    if task == "classification":
        counts = clean.astype(str).value_counts()
        return {
            "n_classes": int(len(counts)), "minority_ratio": round(float(counts.min() / counts.sum()), 4),
            "imbalance_ratio": round(float(counts.max() / counts.min()), 3),
        }
    values = pd.to_numeric(clean, errors="coerce").dropna()
    return {
        "mean": round(float(values.mean()), 4), "std": round(float(values.std()), 4),
        "skew": round(float(values.skew()), 3) if len(values) > 2 else 0.0, "n_unique": int(values.nunique()),
    }


class RunStore:
    """Thread-safe SQLite wrapper (one connection, serialised by a lock; all calls are short)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        """Close the connection."""
        with self._lock:
            self._db.close()

    # --- datasets ---------------------------------------------------------------------------------------------
    def upsert_dataset(self, raw: pd.DataFrame, name: str) -> str:
        """Register (or refresh) a dataset and return its id."""
        fp = fingerprint(raw)
        profile = json.dumps(sanitize(dataset_profile(name, raw)))
        with self._lock:
            row = self._db.execute("SELECT id FROM datasets WHERE fingerprint = ?", (fp,)).fetchone()
            if row:
                self._db.execute("UPDATE datasets SET name = ?, profile = ?, last_seen = ? WHERE id = ?", (name, profile, now_iso(), row["id"]))
                self._db.commit()
                return row["id"]
            did = uuid.uuid4().hex[:12]
            self._db.execute(
                "INSERT INTO datasets (id, fingerprint, name, profile, first_seen, last_seen) VALUES (?,?,?,?,?,?)",
                (did, fp, name, profile, now_iso(), now_iso()),
            )
            self._db.commit()
            return did

    def list_datasets(self) -> list[dict[str, Any]]:
        """Every dataset with its run count and best score per task."""
        with self._lock:
            rows = self._db.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM runs r WHERE r.dataset_id = d.id) AS n_runs FROM datasets d ORDER BY last_seen DESC"
            ).fetchall()
        return [{"id": r["id"], "name": r["name"], "profile": json.loads(r["profile"]), "first_seen": r["first_seen"],
                 "last_seen": r["last_seen"], "n_runs": r["n_runs"]} for r in rows]

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        """One dataset or 404."""
        for d in self.list_datasets():
            if d["id"] == dataset_id:
                return d
        raise NotFoundError(f"Dataset '{dataset_id}' does not exist.", error_code="DATASET_NOT_FOUND")

    def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset and all of its runs."""
        self.get_dataset(dataset_id)
        with self._lock:
            self._db.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            self._db.commit()

    # --- runs -------------------------------------------------------------------------------------------------
    def add_run(self, dataset_id: str, label: str, task: str, target: str, payload: dict[str, Any], curves: dict[str, Any] | None) -> str:
        """Save a finished training job."""
        rid = uuid.uuid4().hex[:12]
        with self._lock:
            self._db.execute(
                "INSERT INTO runs (id, dataset_id, created_at, label, task, target, payload, curves) VALUES (?,?,?,?,?,?,?,?)",
                (rid, dataset_id, now_iso(), label, task, target, json.dumps(sanitize(payload)), json.dumps(sanitize(curves)) if curves else None),
            )
            self._db.commit()
        return rid

    def _summary(self, r: sqlite3.Row) -> dict[str, Any]:
        p = json.loads(r["payload"])
        return {
            "id": r["id"], "dataset_id": r["dataset_id"], "dataset_name": r["dataset_name"], "created_at": r["created_at"],
            "label": r["label"], "notes": r["notes"], "task": r["task"], "target": r["target"],
            "settings": p.get("settings", {}), "top": p.get("top"), "n_models": len(p.get("leaderboard", [])),
            "rows_used": p.get("rows_used"), "tuned": any(m.get("tuned") for m in p.get("leaderboard", [])),
        }

    def list_runs(self, dataset_id: str | None = None, task: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        """Newest-first run summaries."""
        sql = "SELECT r.*, d.name AS dataset_name FROM runs r JOIN datasets d ON d.id = r.dataset_id"
        where, args = [], []
        if dataset_id:
            where.append("r.dataset_id = ?"); args.append(dataset_id)
        if task:
            where.append("r.task = ?"); args.append(task)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY r.created_at DESC, r.rowid DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, (*args, limit)).fetchall()
        return [self._summary(r) for r in rows]

    def get_run(self, run_id: str, with_curves: bool = True) -> dict[str, Any]:
        """Full run (leaderboard rows, split, settings, curves) or 404."""
        with self._lock:
            r = self._db.execute(
                "SELECT r.*, d.name AS dataset_name, d.profile AS dataset_profile FROM runs r JOIN datasets d ON d.id = r.dataset_id WHERE r.id = ?",
                (run_id,),
            ).fetchone()
        if r is None:
            raise NotFoundError(f"Run '{run_id}' does not exist.", error_code="RUN_NOT_FOUND")
        out = {**self._summary(r), **json.loads(r["payload"]), "dataset_profile": json.loads(r["dataset_profile"])}
        out["curves"] = json.loads(r["curves"]) if with_curves and r["curves"] else None
        return out

    def update_run(self, run_id: str, label: str | None, notes: str | None) -> None:
        """Rename / annotate a run."""
        self.get_run(run_id, with_curves=False)
        with self._lock:
            if label is not None:
                self._db.execute("UPDATE runs SET label = ? WHERE id = ?", (label, run_id))
            if notes is not None:
                self._db.execute("UPDATE runs SET notes = ? WHERE id = ?", (notes, run_id))
            self._db.commit()

    def delete_run(self, run_id: str) -> None:
        """Delete one run."""
        self.get_run(run_id, with_curves=False)
        with self._lock:
            self._db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            self._db.commit()

    # --- comparison -------------------------------------------------------------------------------------------
    def compare(self, run_ids: list[str]) -> dict[str, Any]:
        """Align 2+ runs: per-run summary, a model x run matrix of metrics, dataset profiles and caveats."""
        ids = list(dict.fromkeys(run_ids))
        if len(ids) < 2:
            raise BadRequestError("Pick at least two runs to compare.", error_code="NOT_ENOUGH_RUNS")
        runs = [self.get_run(i) for i in ids]
        tasks = {r["task"] for r in runs}
        warnings: list[str] = []
        if len(tasks) > 1:
            warnings.append("The runs use different tasks, so their scores are not on the same scale.")
        if len({r["dataset_id"] for r in runs}) > 1:
            warnings.append("The runs are on different datasets: differences reflect the data as well as the model.")
        splits = {json.dumps(r.get("settings", {}).get("test_size")) for r in runs}
        if len(splits) > 1:
            warnings.append("The runs use different test sizes, so test scores are not computed on equally sized hold-outs.")
        models: dict[str, dict[str, Any]] = {}
        for r in runs:
            for row in r.get("leaderboard", []):
                models.setdefault(row["abbr"], {"abbr": row["abbr"], "name": row["name"], "family": row["family"], "by_run": {}})
                models[row["abbr"]]["by_run"][r["id"]] = row
        seen_in_all = [m["abbr"] for m in models.values() if all(i in m["by_run"] for i in ids)]
        slim = [{k: v for k, v in r.items() if k not in ("leaderboard", "curves")} for r in runs]
        return {
            "runs": slim, "models": list(models.values()), "common_models": seen_in_all,
            "datasets": list({r["dataset_id"]: {"id": r["dataset_id"], "name": r["dataset_name"], "profile": r["dataset_profile"]} for r in runs}.values()),
            "curves": {r["id"]: r.get("curves") for r in runs},
            "same_task": len(tasks) == 1, "task": runs[0]["task"] if len(tasks) == 1 else None, "warnings": warnings,
        }
