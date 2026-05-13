"""SQLite tracking database for indexed literature documents.

Prevents duplicate indexing and supports incremental updates.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .models import DocumentSource, IndexingStatus

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS indexed_documents (
    doc_id        TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    title         TEXT DEFAULT '',
    indexed_at    TEXT NOT NULL,
    chunk_count   INTEGER DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'indexed'
);

CREATE TABLE IF NOT EXISTS parent_chunks (
    parent_id   TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    section     TEXT DEFAULT 'other',
    text        TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}',
    FOREIGN KEY (doc_id) REFERENCES indexed_documents(doc_id)
);

CREATE INDEX IF NOT EXISTS idx_parents_doc ON parent_chunks(doc_id);
"""


class TrackingDB:
    """Thin wrapper around SQLite for document tracking and parent-chunk storage."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("TrackingDB opened at %s", self._path)

    # ------------------------------------------------------------------
    # Document tracking
    # ------------------------------------------------------------------
    def is_indexed(self, doc_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM indexed_documents WHERE doc_id = ? AND status = 'indexed'",
            (doc_id,),
        ).fetchone()
        return row is not None

    def mark_indexed(
        self,
        doc_id: str,
        source: DocumentSource,
        title: str = "",
        chunk_count: int = 0,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO indexed_documents
               (doc_id, source, title, indexed_at, chunk_count, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (doc_id, source.value, title, now, chunk_count, IndexingStatus.INDEXED.value),
        )
        self._conn.commit()

    def mark_failed(self, doc_id: str, source: DocumentSource) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT OR REPLACE INTO indexed_documents
               (doc_id, source, title, indexed_at, chunk_count, status)
               VALUES (?, ?, '', ?, 0, ?)""",
            (doc_id, source.value, now, IndexingStatus.FAILED.value),
        )
        self._conn.commit()

    def get_all_indexed_ids(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT doc_id FROM indexed_documents WHERE status = 'indexed'"
        ).fetchall()
        return {r["doc_id"] for r in rows}

    def count_indexed(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM indexed_documents WHERE status = 'indexed'"
        ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Parent chunks
    # ------------------------------------------------------------------
    def store_parent_chunks(self, chunks: Sequence[tuple[str, str, str, str, str]]) -> None:
        """Bulk-insert parent chunks.  Each tuple: (parent_id, doc_id, section, text, metadata_json)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO parent_chunks (parent_id, doc_id, section, text, metadata) VALUES (?, ?, ?, ?, ?)",
            chunks,
        )
        self._conn.commit()

    def get_parent_chunk(self, parent_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM parent_chunks WHERE parent_id = ?", (parent_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_parent_chunks_batch(self, parent_ids: Sequence[str]) -> list[dict]:
        if not parent_ids:
            return []
        placeholders = ",".join("?" for _ in parent_ids)
        rows = self._conn.execute(
            f"SELECT * FROM parent_chunks WHERE parent_id IN ({placeholders})",
            list(parent_ids),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> TrackingDB:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
