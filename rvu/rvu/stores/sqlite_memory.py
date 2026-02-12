from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SQLiteMemoryRecord:
    artifact_id: str
    text: str
    source: str


class SQLiteMemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                artifact_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def upsert(self, record: SQLiteMemoryRecord) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memory(artifact_id, text, source) VALUES(?, ?, ?)",
            (record.artifact_id, record.text, record.source),
        )
        self._conn.commit()

    def purge_ids(self, ids: set[str]) -> None:
        for artifact_id in ids:
            self._conn.execute("DELETE FROM memory WHERE artifact_id = ?", (artifact_id,))
        self._conn.commit()

    def list_ids(self) -> list[str]:
        rows = self._conn.execute("SELECT artifact_id FROM memory ORDER BY artifact_id").fetchall()
        return [str(row[0]) for row in rows]

    def export(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT artifact_id, text FROM memory ORDER BY artifact_id"
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}
