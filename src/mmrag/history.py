"""Durable, parameterized SQLite storage for local question history."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel


class HistoryEntry(BaseModel):
    history_id: str
    title: str
    question: str
    payload: dict[str, object]
    pinned: bool
    created_at: str


class HistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS history (
                history_id TEXT PRIMARY KEY, title TEXT NOT NULL, question TEXT NOT NULL,
                payload TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _entry(row: sqlite3.Row) -> HistoryEntry:
        return HistoryEntry(
            history_id=str(row["history_id"]),
            title=str(row["title"]),
            question=str(row["question"]),
            payload=json.loads(row["payload"]),
            pinned=bool(row["pinned"]),
            created_at=str(row["created_at"]),
        )

    def add(self, question: str, payload: dict[str, object], *, title: str | None = None) -> HistoryEntry:
        entry = HistoryEntry(
            history_id=uuid.uuid4().hex,
            title=(title or question)[:120],
            question=question,
            payload=payload,
            pinned=False,
            created_at=datetime.now(UTC).isoformat(),
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO history VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry.history_id,
                    entry.title,
                    entry.question,
                    json.dumps(entry.payload, ensure_ascii=False),
                    int(entry.pinned),
                    entry.created_at,
                ),
            )
        return entry

    def list(self, search: str = "") -> list[HistoryEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM history WHERE question LIKE ? OR title LIKE ?
                ORDER BY pinned DESC, created_at DESC""",
                (f"%{search}%", f"%{search}%"),
            ).fetchall()
        return [self._entry(row) for row in rows]

    def get(self, history_id: str) -> HistoryEntry | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM history WHERE history_id = ?", (history_id,)).fetchone()
        return self._entry(row) if row else None

    def update(self, history_id: str, *, title: str | None = None, pinned: bool | None = None) -> HistoryEntry | None:
        current = self.get(history_id)
        if current is None:
            return None
        next_title = title[:120] if title is not None else current.title
        next_pinned = pinned if pinned is not None else current.pinned
        with self._connect() as connection:
            connection.execute(
                "UPDATE history SET title = ?, pinned = ? WHERE history_id = ?",
                (next_title, int(next_pinned), history_id),
            )
        return self.get(history_id)

    def delete(self, history_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM history WHERE history_id = ?", (history_id,))
        return cursor.rowcount > 0

    def clear(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM history")
        return cursor.rowcount
