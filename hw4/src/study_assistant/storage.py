"""SQLite persistence for study notes."""

from __future__ import annotations

import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class NoteStore:
    """Store and search study notes in a local SQLite database."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        """Create the store, defaulting to ``study_notes.db`` in the project root."""
        self.database_path = (
            Path(database_path)
            if database_path is not None
            else Path(__file__).resolve().parents[2] / "study_notes.db"
        )
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connection() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS notes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
        except OSError as exc:
            raise RuntimeError(
                f"Could not prepare the notes database at {self.database_path}."
            ) from exc

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a transaction and consistently close its SQLite connection."""
        try:
            with closing(sqlite3.connect(self.database_path, timeout=5)) as connection:
                connection.row_factory = sqlite3.Row
                with connection:
                    yield connection
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"Could not access the notes database at {self.database_path}."
            ) from exc

    def save_note(self, title: str, content: str) -> dict[str, int | str]:
        """Save a non-empty note and return its identifier and creation time."""
        title = title.strip()
        content = content.strip()
        if not title:
            raise ValueError("Note title must not be empty.")
        if not content:
            raise ValueError("Note content must not be empty.")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO notes (title, content, created_at) VALUES (?, ?, ?)",
                (title, content, created_at),
            )
            note_id = cursor.lastrowid
        if note_id is None:
            raise RuntimeError("The note was saved without receiving an identifier.")
        return {"id": note_id, "title": title, "created_at": created_at}

    def search_notes(self, query: str, limit: int = 10) -> list[dict[str, int | str]]:
        """Find notes whose title or content contains the query."""
        query = query.strip()
        if not query:
            raise ValueError("Search query must not be empty.")
        if not 1 <= limit <= 100:
            raise ValueError("Search limit must be between 1 and 100.")

        pattern = f"%{query}%"
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT id, title, content, created_at
                FROM notes
                WHERE title LIKE ? OR content LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_note(self, note_id: int) -> dict[str, int | str] | None:
        """Return one note by identifier, or ``None`` if it does not exist."""
        if note_id < 1:
            raise ValueError("Note ID must be a positive integer.")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, title, content, created_at FROM notes WHERE id = ?",
                (note_id,),
            ).fetchone()
        return dict(row) if row is not None else None
