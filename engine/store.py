"""SQLite persistence for saved assessments.

Kept out of the scoring path deliberately: `engine.score.assess` does no
I/O, so the app decides when to persist. Every row is stamped with the
config and model version it was produced under, because a saved assessment
read back under different weights is a different assessment.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS assessment (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    applicant_name  TEXT,
    band            TEXT,
    pd              REAL,
    config_version  TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    applicant_json  TEXT NOT NULL,
    result_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_created ON assessment(created_at DESC);
"""


def _default(obj: Any):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def dumps(obj: Any) -> str:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    return json.dumps(obj, default=_default, ensure_ascii=False)


class AssessmentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, applicant_payload: dict, result: Any) -> str:
        """Persist a draft or a completed assessment. `result` may be None
        for a save-and-resume draft."""
        row_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO assessment (id, created_at, applicant_name, band, pd, "
                "config_version, model_version, applicant_json, result_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    row_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    (applicant_payload.get("identity", {}) or {}).get("full_name"),
                    getattr(result, "band", None),
                    getattr(result, "pd", None),
                    getattr(result, "config_version", "unsaved"),
                    getattr(result, "model_version", "unsaved"),
                    dumps(applicant_payload),
                    dumps(result) if result is not None else None,
                ),
            )
        return row_id

    def list_recent(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, applicant_name, band, pd, config_version, "
                "model_version FROM assessment ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def load(self, row_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM assessment WHERE id = ?", (row_id,)
            ).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["applicant"] = json.loads(out.pop("applicant_json"))
        raw = out.pop("result_json")
        out["result"] = json.loads(raw) if raw else None
        return out

    def delete(self, row_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM assessment WHERE id = ?", (row_id,))
