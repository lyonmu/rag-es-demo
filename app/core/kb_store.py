"""SQLite-backed knowledge base metadata store.

Stores KB name, description, created_at, kb_id, index_name, doc_count.
ES index operations remain in es_client.py; this stores human-readable metadata.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# SQLite DB lives next to other data files
DB_PATH = Path("data/kb_store.db")


def _get_conn() -> sqlite3.Connection:
    """Get a connection with row_factory set."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the knowledge_bases table if it doesn't exist."""
    conn = _get_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                kb_id        TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                description  TEXT DEFAULT '',
                index_name   TEXT NOT NULL,
                doc_count    INTEGER DEFAULT 0,
                created_at   TEXT NOT NULL
            )
            """
        )
        conn.commit()
        logger.info("SQLite KB store initialized at %s", DB_PATH)
    finally:
        conn.close()


def create_kb(kb_id: str, name: str, description: str, index_name: str) -> dict:
    """Insert a new KB record. Returns the record dict."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        conn.execute(
            """
            INSERT INTO knowledge_bases (kb_id, name, description, index_name, doc_count, created_at)
            VALUES (?, ?, ?, ?, 0, ?)
            """,
            (kb_id, name, description, index_name, now),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM knowledge_bases WHERE kb_id = ?", (kb_id,)).fetchone()
        logger.info("KB created in SQLite: kb_id=%s name=%s", kb_id, name)
        return dict(row)
    finally:
        conn.close()


def get_kb(kb_id: str) -> Optional[dict]:
    """Get a single KB record by kb_id. Returns None if not found."""
    conn = _get_conn()
    try:
        row = conn.execute("SELECT * FROM knowledge_bases WHERE kb_id = ?", (kb_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_kbs() -> list[dict]:
    """List all KBs ordered by created_at descending."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT * FROM knowledge_bases ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_doc_count(kb_id: str, increment: int = 1) -> int:
    """Increment (or set) doc_count for a KB. Returns new count."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE knowledge_bases SET doc_count = doc_count + ? WHERE kb_id = ?",
            (increment, kb_id),
        )
        conn.commit()
        row = conn.execute("SELECT doc_count FROM knowledge_bases WHERE kb_id = ?", (kb_id,)).fetchone()
        return row["doc_count"] if row else 0
    finally:
        conn.close()


def set_doc_count(kb_id: str, count: int) -> None:
    """Explicitly set doc_count (e.g. after recalculation)."""
    conn = _get_conn()
    try:
        conn.execute(
            "UPDATE knowledge_bases SET doc_count = ? WHERE kb_id = ?",
            (count, kb_id),
        )
        conn.commit()
    finally:
        conn.close()


def delete_kb(kb_id: str) -> bool:
    """Delete a KB record. Returns True if a row was deleted."""
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("KB deleted from SQLite: kb_id=%s", kb_id)
        return deleted
    finally:
        conn.close()
