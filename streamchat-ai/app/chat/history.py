"""StreamChat AI — Conversation History (SQLite)."""

from __future__ import annotations

import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


class ConversationHistory:
    """SQLite-backed conversation history storage.

    Provides persistent storage for chat sessions with
    message retrieval, session management, and cleanup.
    """

    def __init__(self, db_path: str | None = None):
        settings = get_settings()
        self.db_path = db_path or settings.db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model TEXT,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session
                ON messages(session_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    title TEXT
                )
            """)
        logger.info(f"Initialized conversation history at {self.db_path}")

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Add a message to a session's history.

        Returns:
            Row ID of the inserted message.
        """
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Upsert session
            conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, last_active)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET last_active = ?
                """,
                (session_id, now, now, now),
            )

            cursor = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, model, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    model,
                    now,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            return cursor.lastrowid

    def get_history(
        self, session_id: str, limit: int | None = None
    ) -> list[dict]:
        """Get message history for a session.

        Args:
            session_id: Session identifier.
            limit: Max number of messages (most recent).

        Returns:
            List of message dicts.
        """
        query = """
            SELECT role, content, model, timestamp, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
        """
        params: list = [session_id]

        if limit:
            query = """
                SELECT * FROM (
                    SELECT role, content, model, timestamp, metadata
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) ORDER BY rowid ASC
            """
            params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "model": row["model"],
                "timestamp": row["timestamp"],
                "metadata": (
                    json.loads(row["metadata"]) if row["metadata"] else None
                ),
            }
            for row in rows
        ]

    def list_sessions(self) -> list[dict]:
        """List all sessions with metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT s.session_id, s.created_at, s.last_active, s.title,
                       COUNT(m.id) as message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.last_active DESC
                """
            ).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "created_at": row["created_at"],
                "last_active": row["last_active"],
                "title": row["title"],
                "message_count": row["message_count"],
            }
            for row in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Deleted session: {session_id}")
        return deleted

    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return result[0] if result else 0
